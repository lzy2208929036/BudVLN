"""
StreamVLN SFT-Only Training Script

🎯 Simplified training: Only SFT update on failed greedy episodes.

Key Logic:
1. Run greedy rollout for each episode (deterministic)
2. If oracle demo triggered (failure) → Collect demo + Train with SFT
3. If no oracle demo (success) → Skip, no training needed

This is a simplified version without GRPO:
- No group sampling (only greedy)
- No advantage computation
- No TDR (Time-Decayed Reward)
- Pure SFT on oracle demonstrations

Data for SFT (when failed):
- Oracle demo: RGB + actions (supervised labels)
- Policy history: greedy's RGB before rollback (as memory context)

Author: StreamVLN Team (SFT-Only Implementation)
"""

import os
import sys
import copy
import json
import time
import random
import argparse
import logging
import warnings
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

# Suppress verbose logging before importing habitat
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
os.environ["GLOG_minloglevel"] = "2"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import tqdm

# LoRA imports
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Suppress habitat logging
logging.getLogger("habitat").setLevel(logging.ERROR)
logging.getLogger("habitat_sim").setLevel(logging.ERROR)
logging.getLogger("habitat_baselines").setLevel(logging.ERROR)

import habitat
import transformers
from omegaconf import OmegaConf
from PIL import Image
import yaml
import quaternion

from habitat_baselines.config.default import get_config as get_habitat_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from depth_camera_filtering import filter_depth

# Import habitat extensions to register custom measurements
from streamvln.habitat_extensions import measures as habitat_measures

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from streamvln.rewards.vln_reward import VLNRewardFunction, RewardConfig, RewardPhase
from streamvln.utils.utils import (
    dict_to_cuda, DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
)
from streamvln.utils.dist import get_rank, get_world_size, init_distributed_mode
from streamvln.dataset.vln_action_dataset import preprocess_qwen

# IGNORE_INDEX for label masking
IGNORE_INDEX = -100

# Try importing wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


@dataclass
class SFTOnlyTrainingConfig:
    """Configuration for SFT-only training."""
    # Model and paths
    model_path: str = ""
    habitat_config_path: str = "config/vln_r2r.yaml"
    output_path: str = "./results/sft_only_training"
    
    # 🔥 Failure episodes file (optional)
    # If provided, only train on these episodes instead of random traversal
    failure_episodes_path: str = ""
    
    # Training parameters
    num_updates: int = 500
    num_episodes_per_update: int = 4
    learning_rate: float = 1e-6
    max_grad_norm: float = 0.5
    
    # 🔥 Gradient accumulation: accumulate gradients over multiple segments before update
    gradient_accumulation_steps: int = 1  # 1 = no accumulation, >1 = accumulate N batches
    
    # SFT Demo parameters (aligned with offline training)
    num_frames: int = 32
    num_future_steps: int = 4
    num_history: int = 8
    model_max_length: int = 4096
    
    # Environment
    max_steps_per_episode: int = 300
    
    # Logging
    log_interval: int = 1
    save_interval: int = 100
    
    # LoRA settings
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    
    # Expert Intervention (Off-track Detection & Recovery)
    enable_recovery: bool = True
    offtrack_dist_thresh: float = 3.0
    offtrack_heading_thresh_deg: float = 90.0
    offtrack_patience: int = 2
    lookahead_k: int = 1
    recovery_dist_thresh: float = 2.0
    recovery_heading_thresh_deg: float = 30.0
    recovery_max_steps: int = 40
    
    # Goal Zone Protection
    goal_radius: float = 3.0
    oracle_goal_radius: float = 1.0
    heading_guard_dist: float = 1.0
    goal_grace_steps: int = 5
    goal_stop_patience: int = 5
    
    # Progress Stall Detection
    no_progress_patience: int = 60
    no_progress_enable: bool = True


class StreamVLNSFTOnlyTrainer:
    """
    SFT-Only Trainer for StreamVLN.
    
    Key features:
    1. Greedy rollout only (no exploration sampling)
    2. Skip successful episodes (no oracle demo triggered)
    3. SFT on failed episodes (oracle demo + policy history)
    4. No GRPO, no advantage computation, no TDR
    """
    
    def __init__(
        self,
        config: SFTOnlyTrainingConfig,
        model: StreamVLNForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda')
        
        # Setup optimizer
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        num_trainable_model = sum(p.numel() for p in trainable_params)
        print(f"✅ SFT-Only Trainer initialized")
        print(f"Trainable model parameters: {num_trainable_model:,}")
        print(f"Episodes per update: {config.num_episodes_per_update}")
        print(flush=True)
        
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup Habitat environment
        self.habitat_config = get_habitat_config(config.habitat_config_path)
        self._setup_habitat_config()
        
        # Logging
        self.global_step = 0
        self.episode_count = 0
        
        # Setup image processor
        base_model = model.get_base_model() if hasattr(model, 'get_base_model') else model
        self.image_processor = base_model.get_vision_tower().image_processor
        self._base_model = base_model
        
        # Camera parameters
        sim_sensors = self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors
        self._camera_height = sim_sensors.rgb_sensor.position[1]
        self._min_depth = sim_sensors.depth_sensor.min_depth
        self._max_depth = sim_sensors.depth_sensor.max_depth
        camera_fov_rad = np.deg2rad(sim_sensors.depth_sensor.hfov)
        self._fx = self._fy = sim_sensors.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        
        # Conversation template
        self.conversation = self._get_conversation_template()
        
        # Action token IDs
        self.action_token_ids = self._get_action_token_ids()
        
        # Action mapping
        self.actions2idx = {
            'STOP': [0],
            '↑': [1],
            '←': [2],
            '→': [3],
        }
        
        # Conjunctions for prompt variation
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is '
        ]
        
        # 🔥 Load failure episodes if provided
        self.failure_episode_ids = None
        self.use_failure_episodes = False
        if config.failure_episodes_path and os.path.exists(config.failure_episodes_path):
            with open(config.failure_episodes_path, 'r', encoding='utf-8') as f:
                failure_data = json.load(f)
            
            # 🔥 Extract episode_ids with type conversion (Habitat uses string episode_id)
            self.failure_episode_ids = []
            for item in failure_data:
                if isinstance(item, dict) and 'episode_id' in item:
                    # Convert to string to match Habitat's episode_id type
                    self.failure_episode_ids.append(str(item['episode_id']))
                else:
                    print(f"   ⚠️ Skipping invalid item: {item}")
            
            if len(self.failure_episode_ids) > 0:
                self.use_failure_episodes = True
                print(f"\n🔥 Loaded {len(self.failure_episode_ids)} failure episodes from:")
                print(f"   {config.failure_episodes_path}")
                print(f"   Training will only use these episodes!")
                print(f"   Sample episode IDs: {self.failure_episode_ids[:5]}")
            else:
                print(f"\n⚠️ No valid failure episodes found in {config.failure_episodes_path}")
    
    def _setup_habitat_config(self):
        """Setup Habitat environment configuration."""
        with read_write(self.habitat_config):
            self.habitat_config.habitat.task.measurements.update({
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=1024,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=True,
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=90,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            })
            
            self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 224
            self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 224
            self.habitat_config.habitat.environment.max_episode_steps = self.config.max_steps_per_episode
            
            if not hasattr(self.habitat_config.habitat, 'seed'):
                self.habitat_config.habitat.seed = 100
            print(f"🎲 Using seed for episode iterator: {self.habitat_config.habitat.seed}")
    
    def _get_conversation_template(self) -> List[Dict]:
        """Get conversation template for instruction."""
        prompt = (
            "<video>\nYou are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        return [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
    
    def actions2text(self, actions: List[int]) -> str:
        """Convert action indices to text symbols."""
        idx2actions = {0: 'STOP', 1: '↑', 2: '←', 3: '→'}
        return ''.join([idx2actions.get(a, '↑') for a in actions])
    
    def prepare_conversation(self, instruction: str, actions: List[int], 
                            num_future_steps: int = 1, has_history: bool = False) -> List[Dict]:
        """Prepare multi-turn conversation for sequence learning."""
        sources = []
        i = 0
        
        full_instruction = instruction
        if has_history:
            full_instruction += f' These are your historical observations: {DEFAULT_MEMORY_TOKEN}.'
        
        conjunction = random.choice(self.conjunctions)
        
        while i < len(actions):
            step_actions = actions[i:i+num_future_steps]
            answer = self.actions2text(step_actions)
            
            if i == 0:
                prompt = f"{full_instruction} {conjunction}{DEFAULT_IMAGE_TOKEN}."
            else:
                prompt = f"{conjunction}{DEFAULT_IMAGE_TOKEN}."
            
            sources.append({"from": "human", "value": prompt})
            sources.append({"from": "gpt", "value": answer})
            
            i += len(step_actions)
        
        return sources
    
    # ========== Helper methods for depth/pose/intrinsic processing ==========
    
    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        """Preprocess depth image to match model input size."""
        from transformers.image_utils import to_numpy_array
        target_height = self.image_processor.crop_size['height']
        target_width = self.image_processor.crop_size['width']
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
        
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        """Calculate camera intrinsic matrix from sensor config."""
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        
        return np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):
        """Adjust intrinsic matrix after image resizing."""
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]
        intrinsic[:, 1] /= ori_size[1] / target_size[1]
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2
        
        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)
        
        return intrinsic
    
    def get_axis_align_matrix(self):
        """Get axis alignment matrix for coordinate transformation."""
        return torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        """Convert xyz position and yaw to transformation matrix."""
        x, y, z = xyz
        return np.array([
            [np.cos(yaw), -np.sin(yaw), 0, x],
            [np.sin(yaw), np.cos(yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
    
    def _get_action_token_ids(self) -> torch.Tensor:
        """Get action token IDs for the 4 valid actions."""
        return torch.tensor([
            self.tokenizer.encode("STOP", add_special_tokens=False)[0],
            self.tokenizer.encode("↑", add_special_tokens=False)[0],
            self.tokenizer.encode("←", add_special_tokens=False)[0],
            self.tokenizer.encode("→", add_special_tokens=False)[0],
        ], dtype=torch.long, device=self.device)
    
    def _preprocess_qwen(self, sources: List[List[Dict]], add_system: bool = True):
        """Preprocess conversation for Qwen model."""
        roles = {"human": "user", "gpt": "assistant"}
        system_message = "You are a helpful assistant."
        
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)
        
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        
        conversations = []
        input_ids = []
        
        for source in sources:
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            
            input_id = []
            
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
            
            for conv in source:
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]
                
                role = roles.get(role, role)
                conv_msg = [{"role": role, "content": content}]
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv_msg)
                input_id += encode_id
            
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            
            input_ids.append(input_id)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations
    
    def train(self, num_updates: int, start_update: int = 0):
        """Main SFT-only training loop."""
        print(f"\n🚀 Starting SFT-Only Training")
        print(f"=" * 80)
        print(f"Updates: {start_update} -> {num_updates}")
        print(f"Episodes per update: {self.config.num_episodes_per_update}")
        print(f"Training mode: Greedy rollout → SFT on failures only")
        
        if self.use_failure_episodes:
            print(f"\n🔥 Failure Episode Mode: ENABLED")
            print(f"  Total failure episodes: {len(self.failure_episode_ids)}")
            print(f"  Will iterate through failure episodes in order")
        
        if self.config.enable_recovery:
            print(f"\n🔥 Expert Intervention: ENABLED")
            print(f"  Off-track threshold: {self.config.offtrack_dist_thresh:.1f}m / {self.config.offtrack_heading_thresh_deg:.0f}°")
            print(f"  Goal radius: {self.config.goal_radius:.1f}m")
        
        print(f"=" * 80 + "\n")
        
        # Create environment
        print("📊 Creating Habitat environment...")
        self._train_env = habitat.Env(config=self.habitat_config)
        self._episode_iterator = iter(self._train_env.episode_iterator)
        
        # 🔥 Build episode_id to episode mapping if using failure episodes
        self._episode_id_to_episode = {}
        if self.use_failure_episodes:
            print("📊 Building episode ID mapping...")
            # Iterate through all episodes to build mapping
            all_episodes = list(self._train_env.episode_iterator.episodes)
            for ep in all_episodes:
                # Store with string key for consistent lookup
                self._episode_id_to_episode[str(ep.episode_id)] = ep
            print(f"   Mapped {len(self._episode_id_to_episode)} episodes")
            print(f"   Sample dataset episode IDs: {list(self._episode_id_to_episode.keys())[:5]}")
            
            # Verify all failure episode IDs exist
            missing_ids = [eid for eid in self.failure_episode_ids if eid not in self._episode_id_to_episode]
            if missing_ids:
                print(f"   ⚠️ Warning: {len(missing_ids)} failure episode IDs not found in dataset")
                print(f"   Sample missing IDs: {missing_ids[:5]}")
            
            # Filter to only valid failure episode IDs
            self.failure_episode_ids = [eid for eid in self.failure_episode_ids if eid in self._episode_id_to_episode]
            print(f"   Valid failure episodes: {len(self.failure_episode_ids)}")
        
        # Initialize ShortestPathFollower
        self.shortest_path_follower = ShortestPathFollower(
            sim=self._train_env.sim,
            goal_radius=3.0,
            return_one_hot=False
        )
        self.shortest_path_follower.mode = 'geodesic_path'
        print(f"✅ ShortestPathFollower initialized for oracle demo collection")
        
        # Training loop
        update_id = start_update
        completed_updates = 0
        
        # Statistics
        total_episodes = 0
        successful_episodes = 0
        failed_episodes = 0
        
        # 🔥 Failure episode index (for iterating through failure episodes)
        failure_ep_idx = 0
        
        while completed_updates < num_updates - start_update:
            update_start_time = time.time()
            
            print(f"\n{'='*80}")
            print(f"Update {update_id+1}/{num_updates}")
            if self.use_failure_episodes:
                remaining = len(self.failure_episode_ids) - failure_ep_idx
                print(f"Failure episodes: {failure_ep_idx}/{len(self.failure_episode_ids)} processed, {remaining} remaining")
            print(f"{'='*80}")
            
            # Collect episodes
            print(f"\n📊 Running {self.config.num_episodes_per_update} greedy episodes...")
            all_demo_experiences = []
            update_success = 0
            update_failed = 0
            
            for ep_idx in range(self.config.num_episodes_per_update):
                print(f"\n  Episode {ep_idx+1}/{self.config.num_episodes_per_update}:")
                
                # 🔥 Get target episode ID if using failure episodes
                target_episode_id = None
                if self.use_failure_episodes:
                    if failure_ep_idx >= len(self.failure_episode_ids):
                        print(f"    ⚠️ All failure episodes processed! Wrapping around...")
                        failure_ep_idx = 0  # Wrap around
                    target_episode_id = self.failure_episode_ids[failure_ep_idx]
                    failure_ep_idx += 1
                
                # Run greedy rollout
                trajectory = self._collect_greedy_trajectory(target_episode_id=target_episode_id)
                total_episodes += 1
                
                # Check if oracle demo was triggered
                oracle_demos = trajectory.get('oracle_demonstrations', [])
                
                if len(oracle_demos) > 0:
                    # Failed: collect demo experiences for SFT
                    failed_episodes += 1
                    update_failed += 1
                    
                    instruction = trajectory['instruction']
                    for demo in oracle_demos:
                        policy_history_rgbs = demo.get('policy_history_rgbs', [])
                        
                        for step_idx, (rgb, action, weight) in enumerate(
                            zip(demo['rgbs'], demo['actions'], demo['weights'])
                        ):
                            all_demo_experiences.append({
                                'rgb': rgb,
                                'action': action,
                                'weight': weight,
                                'instruction': instruction,
                                'step_idx': step_idx,
                                'policy_history_rgbs': policy_history_rgbs,
                            })
                    
                    demo_steps = sum(len(d['rgbs']) for d in oracle_demos)
                    print(f"    ❌ Failed: {len(oracle_demos)} oracle demo(s), {demo_steps} steps → Added to SFT batch")
                else:
                    # Success: skip
                    successful_episodes += 1
                    update_success += 1
                    print(f"    ✅ Success: No oracle demo triggered → Skipping")
            
            # Summary for this update
            print(f"\n  📈 Update Summary: {update_success} success, {update_failed} failed")
            
            # SFT update if we have demo experiences
            if len(all_demo_experiences) > 0:
                print(f"\n🔧 Running SFT update ({len(all_demo_experiences)} demo steps)...")
                stats = self._sft_update(all_demo_experiences)
            else:
                print(f"\n✨ All episodes successful! Skipping SFT update.")
                stats = {'sft_loss': 0.0, 'sft_samples': 0}
            
            # Add episode stats
            stats['success_count'] = update_success
            stats['failed_count'] = update_failed
            stats['success_rate'] = update_success / self.config.num_episodes_per_update
            
            # Logging
            update_time = time.time() - update_start_time
            self._log_update(update_id, stats, update_time)
            
            # Save checkpoint
            if (completed_updates + 1) % self.config.save_interval == 0:
                self.save_checkpoint(completed_updates + start_update + 1)
            
            # Update counters
            completed_updates += 1
            update_id += 1
            self.global_step = completed_updates + start_update
            
            # Memory cleanup
            del all_demo_experiences
            gc.collect()
            torch.cuda.empty_cache()
        
        # Close environment
        if hasattr(self, '_train_env') and self._train_env is not None:
            self._train_env.close()
        
        # Final statistics
        print(f"\n🎉 Training completed!")
        print(f"  Total episodes: {total_episodes}")
        print(f"  Successful: {successful_episodes} ({100*successful_episodes/total_episodes:.1f}%)")
        print(f"  Failed: {failed_episodes} ({100*failed_episodes/total_episodes:.1f}%)")
        print(f"Final checkpoint saved to: {self.config.output_path}/checkpoint_final")
        self.save_checkpoint('final')
    
    def _collect_greedy_trajectory(self, target_episode_id: Optional[int] = None) -> Dict:
        """
        Collect a single greedy trajectory.
        
        Args:
            target_episode_id: If provided, use this specific episode instead of random
        
        Returns trajectory dict with:
        - oracle_demonstrations: List of oracle demos (empty if successful)
        - instruction: Navigation instruction
        - final_metrics: Episode metrics
        """
        env = self._train_env
        
        # 🔥 Set specific episode if target_episode_id is provided
        if target_episode_id is not None and self.use_failure_episodes:
            target_episode = self._episode_id_to_episode.get(target_episode_id)
            if target_episode is not None:
                env.current_episode = target_episode
                observations = env.reset()
            else:
                print(f"    ⚠️ Episode {target_episode_id} not found, using random episode")
                observations = env.reset()
        else:
            # Get next episode (random)
            observations = env.reset()
        
        episode = env.current_episode
        instruction = episode.instruction.instruction_text
        episode_id = episode.episode_id
        
        print(f"    Instruction (ep={episode_id}): {instruction[:60]}...")
        
        # Run greedy rollout
        self.model.eval()
        
        # Disable gradient checkpointing during rollout
        gc_was_enabled = getattr(self.model, 'gradient_checkpointing', False) or \
                         getattr(self.model, 'is_gradient_checkpointing', False)
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        
        self.model.reset_for_env(0)
        
        # Get reference path and goal
        reference_path = np.array(episode.reference_path)
        goal_position = episode.goals[0].position
        
        # Initialize trajectory storage
        trajectory = {
            'states': [],
            'actions': [],
            'instruction': instruction,
            'episode_id': episode_id,
            'oracle_demonstrations': [],
        }
        
        # Visual memory
        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsic_list = []
        time_ids = []
        
        # Get intrinsic matrix
        intrinsic_matrix = self.get_intrinsic_matrix(
            self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor
        )
        initial_height = env.sim.get_agent_state().position[1]
        
        # Generation state
        step_id = 0
        output_ids = None
        past_key_values = None
        action_seq = []
        
        is_done = False
        
        # Off-track detection state
        last_progress_idx = 0
        offtrack_count = 0
        steps_in_goal_zone = 0
        steps_after_grace = 0
        
        # Waypoint tracking
        last_waypoint_state = None
        prev_nearest_idx = -1
        waypoint_history = []
        max_reached_waypoint = -1
        steps_since_new_progress = 0
        best_waypoint_state = None
        
        with torch.no_grad():
            while not is_done and step_id < self.config.max_steps_per_episode:
                curr_distance = env.get_metrics().get('distance_to_goal', float('inf'))
                
                agent_state = env.sim.get_agent_state()
                agent_position = agent_state.position
                agent_rotation = agent_state.rotation
                
                # Off-track detection
                if self.config.enable_recovery:
                    near_goal = curr_distance <= self.config.goal_radius
                    
                    if near_goal:
                        steps_in_goal_zone += 1
                        offtrack_count = 0
                        
                        if steps_in_goal_zone <= self.config.goal_grace_steps:
                            steps_after_grace = 0
                        else:
                            steps_after_grace += 1
                    else:
                        steps_in_goal_zone = 0
                        steps_after_grace = 0
                        
                        nearest_idx = self._nearest_path_index(agent_position, reference_path)
                        last_progress_idx = max(nearest_idx, last_progress_idx)
                        
                        # Update waypoint state
                        if nearest_idx != prev_nearest_idx:
                            last_waypoint_state = {
                                'position': list(agent_position),
                                'rotation': agent_rotation,
                                'step_id': step_id,
                                'nearest_idx': nearest_idx,
                                'observations': observations.copy(),
                                'rgb_list': rgb_list.copy(),
                                'time_ids': time_ids.copy(),
                                'trajectory_length': len(trajectory['actions']),
                            }
                            prev_nearest_idx = nearest_idx
                            
                            waypoint_history.insert(0, last_waypoint_state.copy())
                            if len(waypoint_history) > 20:
                                waypoint_history = waypoint_history[:20]
                            
                            if nearest_idx > max_reached_waypoint:
                                max_reached_waypoint = nearest_idx
                                steps_since_new_progress = 0
                                best_waypoint_state = last_waypoint_state.copy()
                            else:
                                steps_since_new_progress += 1
                        else:
                            steps_since_new_progress += 1
                        
                        target_waypoint = self._select_target_waypoint(
                            reference_path, nearest_idx, last_progress_idx, goal_position
                        )
                        
                        dist_to_ref_path = self._dist_to_reference_path(
                            agent_position, reference_path, nearest_idx
                        )
                        heading_error = self._heading_error_deg(agent_rotation, agent_position, target_waypoint)
                        
                        offtrack_now = (
                            dist_to_ref_path > self.config.offtrack_dist_thresh or
                            (heading_error > self.config.offtrack_heading_thresh_deg and 
                             dist_to_ref_path > self.config.heading_guard_dist)
                        )
                        
                        if offtrack_now:
                            offtrack_count += 1
                        else:
                            offtrack_count = 0
                        
                        # Progress stall detection
                        progress_stall_detected = (
                            self.config.no_progress_enable and
                            steps_since_new_progress >= self.config.no_progress_patience and
                            not near_goal
                        )
                        
                        if progress_stall_detected:
                            print(f"      [Step {step_id}] Progress stall! No new waypoint in {steps_since_new_progress} steps")
                            
                            if not is_done:
                                rollback_state = self._find_reachable_rollback_state(
                                    env, goal_position, episode,
                                    best_waypoint_state, last_waypoint_state, waypoint_history
                                )
                                
                                rollback_step_id = rollback_state.get('step_id', len(trajectory['states']))
                                policy_history = [s['rgb'] for s in trajectory['states'][:rollback_step_id + 1]] if len(trajectory['states']) > 0 else []
                                
                                oracle_demo = self._collect_oracle_demonstration(
                                    env, rollback_state, goal_position, reference_path, policy_history
                                )
                                
                                if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                    trajectory['oracle_demonstrations'].append(oracle_demo)
                            
                            is_done = True
                            break
                        
                        # Offtrack detection
                        if offtrack_count >= self.config.offtrack_patience:
                            print(f"      [Step {step_id}] Offtrack! dist={dist_to_ref_path:.2f}m, heading={heading_error:.1f}°")
                            
                            if not is_done:
                                rollback_state = self._find_reachable_rollback_state(
                                    env, goal_position, episode,
                                    best_waypoint_state, last_waypoint_state, waypoint_history
                                )
                                
                                rollback_step_id = rollback_state.get('step_id', len(trajectory['states']))
                                policy_history = [s['rgb'] for s in trajectory['states'][:rollback_step_id + 1]] if len(trajectory['states']) > 0 else []
                                
                                oracle_demo = self._collect_oracle_demonstration(
                                    env, rollback_state, goal_position, reference_path, policy_history
                                )
                                
                                if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                    trajectory['oracle_demonstrations'].append(oracle_demo)
                            
                            is_done = True
                            break
                
                # Process observations
                rgb = observations['rgb']
                depth = observations['depth']
                x, y = observations['gps']
                camera_yaw = observations['compass'][0]
                
                depth_filtered = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                depth_filtered = depth_filtered * (self._max_depth - self._min_depth) + self._min_depth
                depth_filtered = depth_filtered * 1000
                
                image = Image.fromarray(rgb).convert('RGB')
                image_size = image.size
                image_tensor = self.image_processor.preprocess(
                    images=image, return_tensors='pt'
                )['pixel_values'][0]
                
                depth_image, resize_shape = self.preprocess_depth_image(
                    Image.fromarray(depth_filtered.astype(np.uint16), mode='I;16'),
                    do_depth_scale=True
                )
                
                height = agent_position[1] - initial_height
                camera_position = np.array([x, -y, self._camera_height + height])
                tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                
                intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                intrinsic = torch.from_numpy(intrinsic).float()
                
                rgb_list.append(image_tensor)
                depth_list.append(torch.from_numpy(depth_image).float())
                pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                intrinsic_list.append(intrinsic)
                time_ids.append(step_id)
                
                # Generate action
                if len(action_seq) == 0:
                    if output_ids is None:
                        sources = copy.deepcopy(self.conversation)
                        sources[0]["value"] = sources[0]["value"].replace(
                            ' Where should you go next to stay on track?',
                            ' Please devise an action sequence to follow the instruction '
                            'which may include turning left or right by a certain degree, '
                            'moving forward by a certain distance or stopping once the task is complete.'
                        )
                        if step_id != 0:
                            sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                        sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN + '\n', '')
                        sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
                        add_system = True
                    else:
                        sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                        add_system = False
                    
                    input_ids, _ = self._preprocess_qwen([sources], add_system=add_system)
                    if output_ids is not None:
                        input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)
                    
                    images = rgb_list[-1:]
                    depths = depth_list[-1:]
                    poses = pose_list[-1:]
                    intrinsics = intrinsic_list[-1:]
                    
                    if step_id != 0 and step_id % self.config.num_frames == 0:
                        if self.config.num_history is None:
                            history_ids = slice(0, time_ids[0], self.config.num_future_steps)
                        else:
                            history_ids = slice(0, time_ids[0], (time_ids[0] // self.config.num_history))
                        images = rgb_list[history_ids] + images
                        depths = depth_list[history_ids] + depths
                        poses = pose_list[history_ids] + poses
                        intrinsics = intrinsic_list[history_ids] + intrinsics
                    
                    input_dict = {
                        'images': torch.stack(images).unsqueeze(0),
                        'depths': torch.stack(depths).unsqueeze(0),
                        'poses': torch.stack(poses).unsqueeze(0),
                        'intrinsics': torch.stack(intrinsics).unsqueeze(0),
                        'inputs': input_ids,
                        'env_id': 0,
                        'time_ids': [time_ids],
                        'task_type': [0],
                    }
                    
                    input_dict = dict_to_cuda(input_dict, self.device)
                    for key in ['images', 'depths', 'poses', 'intrinsics']:
                        if input_dict[key] is not None:
                            input_dict[key] = input_dict[key].to(torch.bfloat16)
                    
                    # Greedy generation (do_sample=False)
                    outputs = self.model.generate(
                        **input_dict,
                        do_sample=False,
                        temperature=1.0,
                        num_beams=1,
                        max_new_tokens=10000,
                        use_cache=True,
                        return_dict_in_generate=True,
                        past_key_values=past_key_values,
                    )
                    
                    output_ids = outputs.sequences
                    past_key_values = outputs.past_key_values
                    
                    llm_output = self.tokenizer.batch_decode(
                        output_ids, skip_special_tokens=False
                    )[0].strip()
                    
                    action_seq = self._parse_actions(llm_output)
                    
                    if len(action_seq) == 0:
                        action_seq = [0]
                
                action = action_seq.pop(0)
                
                # Check for forced STOP in goal zone
                if (self.config.enable_recovery and 
                    self.config.goal_stop_patience > 0 and
                    near_goal and
                    steps_in_goal_zone > self.config.goal_grace_steps and
                    action != 0):
                    
                    if steps_after_grace >= self.config.goal_stop_patience:
                        print(f"      [Step {step_id}] Forced STOP in goal zone")
                        
                        if not is_done:
                            rollback_state = self._find_reachable_rollback_state(
                                env, goal_position, episode,
                                best_waypoint_state, last_waypoint_state, waypoint_history
                            )
                            
                            rollback_step_id = rollback_state.get('step_id', len(trajectory['states']))
                            policy_history = [s['rgb'] for s in trajectory['states'][:rollback_step_id + 1]] if len(trajectory['states']) > 0 else []
                            
                            oracle_demo = self._collect_oracle_demonstration(
                                env, rollback_state, goal_position, reference_path, policy_history
                            )
                            
                            if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                trajectory['oracle_demonstrations'].append(oracle_demo)
                        
                        is_done = True
                        break
                
                # Check for premature STOP
                if (self.config.enable_recovery and 
                    action == 0 and
                    curr_distance > self.config.goal_radius):
                    
                    print(f"      [Step {step_id}] Premature STOP! dist={curr_distance:.2f}m")
                    
                    if not is_done:
                        rollback_state = self._find_reachable_rollback_state(
                            env, goal_position, episode,
                            best_waypoint_state, last_waypoint_state, waypoint_history
                        )
                        
                        rollback_step_id = rollback_state.get('step_id', len(trajectory['states']))
                        policy_history = [s['rgb'] for s in trajectory['states'][:rollback_step_id + 1]] if len(trajectory['states']) > 0 else []
                        
                        oracle_demo = self._collect_oracle_demonstration(
                            env, rollback_state, goal_position, reference_path, policy_history
                        )
                        
                        if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                            trajectory['oracle_demonstrations'].append(oracle_demo)
                    
                    is_done = True
                    break
                
                # Save state
                trajectory['states'].append({
                    'rgb': observations['rgb'].copy(),
                    'step_id': step_id,
                    'instruction': instruction,
                })
                trajectory['actions'].append(action)
                
                # Step environment
                observations = env.step(action)
                step_id += 1
                
                # Check if done
                if action == 0:
                    is_done = True
                else:
                    is_done = env.episode_over
                
                # Reset at num_frames boundary
                if step_id % self.config.num_frames == 0:
                    self.model.reset_for_env(0)
                    output_ids = None
                    past_key_values = None
                    time_ids = []
                    action_seq = []
        
        # Store final metrics
        trajectory['final_metrics'] = env.get_metrics()
        
        # Re-enable gradient checkpointing
        if gc_was_enabled and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        return trajectory
    
    def _parse_actions(self, output: str) -> List[int]:
        """Parse action symbols from model output."""
        import re
        import itertools
        
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
    
    def _nearest_path_index(self, agent_pos: np.ndarray, reference_path: np.ndarray) -> int:
        """Find nearest point index on reference path."""
        if len(reference_path) == 0:
            return 0
        distances = np.linalg.norm(reference_path[:, [0, 2]] - agent_pos[[0, 2]], axis=1)
        return int(np.argmin(distances))
    
    def _dist_to_path_segment(self, agent_pos: np.ndarray, path_point_a: np.ndarray, 
                             path_point_b: np.ndarray) -> float:
        """Calculate perpendicular distance to path segment."""
        agent_2d = agent_pos[[0, 2]]
        a_2d = path_point_a[[0, 2]]
        b_2d = path_point_b[[0, 2]]
        
        v = b_2d - a_2d
        w = agent_2d - a_2d
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(np.linalg.norm(agent_2d - a_2d))
        
        c2 = np.dot(v, v)
        if c1 >= c2:
            return float(np.linalg.norm(agent_2d - b_2d))
        
        b = c1 / c2
        projection = a_2d + b * v
        return float(np.linalg.norm(agent_2d - projection))
    
    def _dist_to_reference_path(self, agent_pos: np.ndarray, reference_path: np.ndarray, 
                                nearest_idx: int) -> float:
        """Calculate minimum distance to reference path."""
        if len(reference_path) <= 1:
            return float(np.linalg.norm(agent_pos[[0, 2]] - reference_path[0][[0, 2]]))
        
        min_dist = float('inf')
        
        if nearest_idx > 0:
            dist = self._dist_to_path_segment(
                agent_pos, reference_path[nearest_idx - 1], reference_path[nearest_idx]
            )
            min_dist = min(min_dist, dist)
        
        if nearest_idx < len(reference_path) - 1:
            dist = self._dist_to_path_segment(
                agent_pos, reference_path[nearest_idx], reference_path[nearest_idx + 1]
            )
            min_dist = min(min_dist, dist)
        
        if nearest_idx == 0 or nearest_idx == len(reference_path) - 1:
            point_dist = float(np.linalg.norm(
                agent_pos[[0, 2]] - reference_path[nearest_idx][[0, 2]]
            ))
            min_dist = min(min_dist, point_dist)
        
        return min_dist
    
    def _select_target_waypoint(self, reference_path: np.ndarray, nearest_idx: int, 
                                last_progress_idx: int, goal_position: np.ndarray) -> np.ndarray:
        """Select target waypoint with lookahead."""
        current_idx = max(nearest_idx, last_progress_idx)
        target_idx = min(current_idx + self.config.lookahead_k, len(reference_path) - 1)
        
        if target_idx >= len(reference_path) - 1:
            return goal_position
        
        return reference_path[target_idx]
    
    def _heading_error_deg(self, agent_rotation: np.quaternion, agent_pos: np.ndarray, 
                          target_pos: np.ndarray) -> float:
        """Calculate heading error in degrees."""
        agent_pos = np.asarray(agent_pos)
        target_pos = np.asarray(target_pos)
        
        forward_vector = quaternion.as_rotation_matrix(agent_rotation) @ np.array([0, 0, -1])
        forward_2d = forward_vector[[0, 2]]
        forward_2d = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
        
        target_dir = target_pos[[0, 2]] - agent_pos[[0, 2]]
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        
        cos_angle = np.clip(np.dot(forward_2d, target_dir), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return float(np.rad2deg(angle_rad))
    
    def _collect_oracle_demonstration(
        self,
        env: habitat.Env,
        rollback_state: Dict,
        goal_position: np.ndarray,
        reference_path: np.ndarray,
        policy_history_rgbs: Optional[List[np.ndarray]] = None,
    ) -> Optional[Dict]:
        """Collect oracle demonstration from rollback point to goal."""
        print(f"      [Oracle Demo] Restoring to rollback state...")
        
        try:
            position = rollback_state['position']
            if isinstance(position, list):
                position = np.array(position, dtype=np.float32)
            elif not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)
            
            rotation = rollback_state['rotation']
            if hasattr(rotation, 'components'):
                rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
            elif isinstance(rotation, np.ndarray):
                rotation = rotation.tolist()
            
            success = env.sim.set_agent_state(
                position=position,
                rotation=rotation,
                reset_sensors=True
            )
            if not success:
                print(f"      Failed to restore agent state")
                return None
            
            env._episode_over = False
            env._elapsed_steps = rollback_state.get('step_id', 0)
            
        except Exception as e:
            print(f"      Exception during state restore: {e}")
            return None
        
        raw_observations = env.sim.get_sensor_observations()
        observations = env.sim._sensor_suite.get_observations(raw_observations)
        
        follower = ShortestPathFollower(
            env.sim,
            goal_radius=self.config.oracle_goal_radius,
            return_one_hot=False
        )
        
        demo_rgbs = []
        demo_actions = []
        demo_step_ids = []
        
        max_oracle_steps = self.config.max_steps_per_episode - rollback_state['step_id']
        oracle_step = 0
        oracle_done = False
        
        while not oracle_done and oracle_step < max_oracle_steps:
            oracle_action = follower.get_next_action(goal_position)
            
            if oracle_action is None:
                if len(demo_rgbs) > 0:
                    break
                else:
                    return None
            
            if oracle_action == 0:
                oracle_done = True
            
            demo_rgbs.append(observations['rgb'].copy())
            demo_actions.append(oracle_action)
            demo_step_ids.append(rollback_state['step_id'] + oracle_step)
            
            if env._episode_over:
                env._episode_over = False
            
            try:
                observations = env.step(oracle_action)
            except AssertionError as e:
                if "Episode over" in str(e):
                    break
                else:
                    raise e
            oracle_step += 1
            
            if observations.get('done', False):
                oracle_done = True
        
        if len(demo_rgbs) == 0:
            return None
        
        print(f"      [Oracle Demo] Collected {len(demo_rgbs)} steps")
        
        # Compute weights
        demo_weights = self._compute_demonstration_weights(len(demo_rgbs))
        
        # Sample policy history
        sampled_policy_history = []
        if policy_history_rgbs is not None and len(policy_history_rgbs) > 0:
            lookback_window = min(len(policy_history_rgbs), 32)
            history_candidates = policy_history_rgbs[-lookback_window:]
            
            if len(history_candidates) <= self.config.num_history:
                sampled_policy_history = history_candidates
            else:
                indices = np.linspace(0, len(history_candidates) - 1, self.config.num_history, dtype=int)
                sampled_policy_history = [history_candidates[i] for i in indices]
        
        return {
            'rgbs': demo_rgbs,
            'actions': demo_actions,
            'step_ids': demo_step_ids,
            'weights': demo_weights,
            'rollback_step_id': rollback_state['step_id'],
            'rollback_waypoint_idx': rollback_state['nearest_idx'],
            'policy_history_rgbs': sampled_policy_history,
        }
    
    def _compute_demonstration_weights(self, num_steps: int) -> List[float]:
        """Compute exponential decay weights for demonstration."""
        base_weight = 2.0
        min_weight = 1.0
        
        if num_steps <= 1:
            return [base_weight]
        
        decay_rate = (min_weight / base_weight) ** (1.0 / (num_steps - 1))
        
        return [base_weight * (decay_rate ** step) for step in range(num_steps)]
    
    def _check_goal_reachability(self, env: habitat.Env, rollback_state: Dict, 
                                 goal_position: np.ndarray) -> bool:
        """Check if goal is reachable from rollback point."""
        current_position = env.sim.get_agent_state().position.copy()
        current_rotation = env.sim.get_agent_state().rotation
        
        try:
            position = rollback_state['position']
            if isinstance(position, list):
                position = np.array(position, dtype=np.float32)
            elif not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)
            
            rotation = rollback_state['rotation']
            if hasattr(rotation, 'components'):
                rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
            elif isinstance(rotation, np.ndarray):
                rotation = rotation.tolist()
            
            success = env.sim.set_agent_state(
                position=position,
                rotation=rotation,
                reset_sensors=True
            )
            if not success:
                return False
            
            follower = ShortestPathFollower(
                env.sim,
                goal_radius=self.config.oracle_goal_radius,
                return_one_hot=False
            )
            test_action = follower.get_next_action(goal_position)
            
            env.sim.set_agent_state(
                position=current_position,
                rotation=current_rotation,
                reset_sensors=True
            )
            
            if test_action is None:
                return False
            
            if test_action == 0:
                dist_to_goal = np.linalg.norm(position - goal_position)
                if dist_to_goal > 1.0:
                    return False
            
            return True
            
        except Exception as e:
            return False
    
    def _find_reachable_rollback_state(
        self,
        env: habitat.Env,
        goal_position: np.ndarray,
        episode,
        best_waypoint_state: Optional[Dict],
        last_waypoint_state: Optional[Dict],
        waypoint_history: List[Dict],
    ) -> Dict:
        """Find a reachable rollback state."""
        candidates = []
        
        if best_waypoint_state is not None:
            candidates.append(('best', best_waypoint_state))
        
        if last_waypoint_state is not None and last_waypoint_state != best_waypoint_state:
            candidates.append(('last', last_waypoint_state))
        
        for i, wp_state in enumerate(waypoint_history):
            if wp_state not in [best_waypoint_state, last_waypoint_state]:
                candidates.append((f'history_{i}', wp_state))
        
        for candidate_name, candidate_state in candidates:
            if self._check_goal_reachability(env, candidate_state, goal_position):
                return candidate_state
        
        return {
            'position': list(episode.start_position),
            'rotation': episode.start_rotation,
            'step_id': 0,
            'nearest_idx': 0,
        }
    
    def _preprocess_observation_rgb_only(self, rgb: np.ndarray) -> Dict:
        """Preprocess RGB observation."""
        if rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
            raise ValueError(f"Invalid RGB data")
        
        rgb_pil = Image.fromarray(rgb)
        image_tensor = self.image_processor.preprocess(
            rgb_pil, return_tensors='pt'
        )['pixel_values'][0]
        return {'image': image_tensor}
    
    def _sft_update(self, demo_experiences: List[Dict]) -> Dict:
        """
        SFT update on oracle demonstrations.
        
        Uses multi-turn conversation format aligned with offline training.
        """
        stats = {
            'sft_loss': 0.0,
            'sft_samples': 0,
        }
        num_updates = 0
        
        print(f"     🎯 Using Multi-turn Conversation SFT")
        
        # Group demos by instruction
        from collections import defaultdict
        instruction_demo_groups = defaultdict(list)
        for demo_exp in demo_experiences:
            instruction_demo_groups[demo_exp['instruction']].append(demo_exp)
        
        print(f"     Grouped {len(demo_experiences)} demo steps into {len(instruction_demo_groups)} instruction groups")
        
        # Build training segments
        training_segments = []
        for instruction, demos in instruction_demo_groups.items():
            demos = sorted(demos, key=lambda x: x.get('step_idx', 0))
            actions_len = len(demos)
            
            num_rounds = actions_len // self.config.num_frames
            for n in range(num_rounds + 1):
                if n * self.config.num_frames == actions_len:
                    continue
                
                start_idx = n * self.config.num_frames
                end_idx = min(start_idx + self.config.num_frames, actions_len)
                
                segment_demos = demos[start_idx:end_idx]
                segment_actions = [demo['action'] for demo in segment_demos]
                segment_weights = [demo['weight'] for demo in segment_demos]
                
                if len(segment_actions) == 0:
                    continue
                
                interval = self.config.num_future_steps
                sample_indices = list(range(0, len(segment_demos), interval))
                
                segment_rgbs = []
                has_invalid = False
                for i in sample_indices:
                    rgb = segment_demos[i]['rgb']
                    if rgb is None or not isinstance(rgb, np.ndarray) or rgb.size == 0:
                        has_invalid = True
                        break
                    segment_rgbs.append(rgb)
                
                if has_invalid:
                    continue
                
                # History frames
                history_rgbs = []
                has_history = False
                use_nonzero_start = False
                
                if start_idx > 0:
                    history_interval = max(start_idx // self.config.num_history, 1)
                    history_indices = list(range(0, start_idx, history_interval))
                    history_rgbs = [demos[i]['rgb'] for i in history_indices]
                    has_history = True
                else:
                    policy_history_rgbs = demos[0].get('policy_history_rgbs', [])
                    
                    if len(policy_history_rgbs) > 0:
                        valid_policy_history = [h for h in policy_history_rgbs 
                                               if h is not None and isinstance(h, np.ndarray) and h.size > 0]
                        
                        if len(valid_policy_history) > 0:
                            if len(valid_policy_history) < self.config.num_history:
                                padding_count = self.config.num_history - len(valid_policy_history)
                                first_frame = valid_policy_history[0]
                                history_rgbs = [first_frame] * padding_count + valid_policy_history
                            else:
                                history_rgbs = valid_policy_history[:self.config.num_history]
                            
                            has_history = True
                            use_nonzero_start = True
                
                if len(segment_rgbs) == 0:
                    continue
                
                all_rgbs = history_rgbs + segment_rgbs
                
                if use_nonzero_start:
                    action_time_ids = list(range(1, end_idx - start_idx + 1))
                else:
                    action_time_ids = list(range(start_idx, end_idx))
                
                # 🔥 Get short instruction ID for logging
                instr_short = instruction[:30].replace('\n', ' ') + "..."
                
                training_segments.append({
                    'instruction': instruction,
                    'rgbs': all_rgbs,
                    'sample_rgbs_count': len(segment_rgbs),
                    'history_rgbs_count': len(history_rgbs),
                    'actions': segment_actions,
                    'weights': segment_weights,
                    'time_ids': action_time_ids,
                    'has_history': has_history,
                    'segment_info': f"{start_idx}-{end_idx-1}/{actions_len}",
                    'instruction_short': instr_short,  # 🔥 For logging
                })
        
        print(f"     Split into {len(training_segments)} training segments")
        if self.config.gradient_accumulation_steps > 1:
            print(f"     🔥 Gradient accumulation: {self.config.gradient_accumulation_steps} batches per update")
        
        # Train
        random.shuffle(training_segments)
        
        # 🔥 Initialize gradient accumulation counter and clear gradients
        accumulation_counter = 0
        self.optimizer.zero_grad()
        
        pbar = tqdm.tqdm(
            training_segments,
            desc=f"  SFT Update",
            ncols=120,
            leave=True,
            file=sys.stdout,
        )
        
        for segment in pbar:
            instruction = segment['instruction']
            all_rgbs = segment['rgbs']
            sample_rgbs_count = segment['sample_rgbs_count']
            history_rgbs_count = segment['history_rgbs_count']
            actions = segment['actions']
            weights = segment['weights']
            time_ids_list = segment['time_ids']
            has_history = segment['has_history']
            segment_info = segment['segment_info']
            
            if sample_rgbs_count == 0 or len(actions) == 0:
                continue
            
            # Build conversation
            conversation = self.prepare_conversation(
                instruction=instruction,
                actions=actions,
                num_future_steps=self.config.num_future_steps,
                has_history=has_history
            )
            
            # Tokenize
            data_dict = preprocess_qwen([conversation], self.tokenizer, has_image=True)
            input_ids = data_dict['input_ids'].to(self.device)
            labels = data_dict['labels'].to(self.device)
            
            # Prepare images
            try:
                image_tensors = []
                for rgb in all_rgbs:
                    obs_dict = self._preprocess_observation_rgb_only(rgb)
                    image_tensors.append(obs_dict['image'])
                
                images = torch.stack(image_tensors, dim=0).to(self.device, dtype=torch.bfloat16)
                images = images.unsqueeze(0)
            except Exception as e:
                print(f"\n     ⚠️ Error processing images: {e}")
                continue
            
            time_ids = [time_ids_list]
            
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    images=images,
                    labels=labels,
                    time_ids=time_ids,
                    task_type=[0],
                    depths=None,
                    poses=None,
                    intrinsics=None,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                
                sft_loss = outputs.loss
                
                weight_mean = sum(weights) / len(weights)
                weighted_loss = sft_loss * weight_mean
                
                # 🔥 Gradient Accumulation: Scale loss by accumulation steps
                # This keeps the effective gradient scale the same regardless of accumulation
                accum_steps = self.config.gradient_accumulation_steps
                scaled_loss = weighted_loss / accum_steps
                
                # Backward (accumulate gradients)
                scaled_loss.backward()
                
                # Track stats
                stats['sft_loss'] += sft_loss.item()
                stats['sft_samples'] += len(all_rgbs)
                accumulation_counter += 1
                
                # 🔥 Only update when accumulated enough batches
                if accumulation_counter >= accum_steps:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    num_updates += 1
                    accumulation_counter = 0
                
                # 🔥 Show instruction hint to distinguish different demos
                instr_hint = segment.get('instruction_short', '')[:20]
                pbar.set_postfix({
                    'loss': f"{sft_loss.item():.3f}",
                    'accum': f"{accumulation_counter}/{accum_steps}",
                    'seg': segment_info,
                    'instr': instr_hint,
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n     ⚠️ OOM error, skipping segment...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            del sft_loss, weighted_loss, scaled_loss, outputs, images
            torch.cuda.empty_cache()
        
        # 🔥 Handle remaining accumulated gradients (if any)
        if accumulation_counter > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
            num_updates += 1
        
        # Average stats
        total_segments = stats['sft_samples'] // max(1, len(training_segments)) if len(training_segments) > 0 else 0
        if len(training_segments) > 0:
            stats['sft_loss'] /= len(training_segments)
        
        # 🔥 Improved logging: show actual segments merged per update
        if self.config.gradient_accumulation_steps > 1:
            segs_per_update = len(training_segments) / max(num_updates, 1)
            accum_info = f" ({len(training_segments)} segments merged → {num_updates} optimizer step)"
        else:
            accum_info = ""
        print(f"     ✅ SFT completed: {num_updates} optimizer updates{accum_info}, avg loss = {stats['sft_loss']:.4f}")
        
        return stats
    
    def _log_update(self, update_id: int, stats: Dict, update_time: float):
        """Log update statistics."""
        print(f"\n📈 Update {update_id+1} Statistics:")
        print(f"  Episodes: {stats['success_count']} success, {stats['failed_count']} failed")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  SFT Loss: {stats['sft_loss']:.4f}")
        print(f"  SFT Samples: {stats['sft_samples']:.0f}")
        print(f"  Update Time: {update_time:.1f}s")
        
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                'update': update_id + 1,
                'sft_loss': stats['sft_loss'],
                'sft_samples': stats['sft_samples'],
                'success_count': stats['success_count'],
                'failed_count': stats['failed_count'],
                'success_rate': stats['success_rate'],
                'update_time': update_time,
            })
    
    def save_checkpoint(self, update_id):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_path, f"checkpoint_{update_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        training_state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(
            training_state,
            os.path.join(checkpoint_dir, "training_state.pt")
        )
        
        print(f"💾 Checkpoint saved: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--habitat_config_path", type=str, default="config/vln_r2r.yaml")
    parser.add_argument("--output_path", type=str, default="./results/sft_only_training")
    parser.add_argument("--num_updates", type=int, default=500)
    parser.add_argument("--num_episodes_per_update", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="streamvln_sft_only")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # 🔥 Failure episodes file
    parser.add_argument("--failure_episodes_path", type=str, default="",
                        help="Path to JSON file containing failure episode IDs. "
                             "If provided, only train on these episodes.")
    
    # Expert Intervention
    parser.add_argument("--enable_recovery", action="store_true", default=True)
    parser.add_argument("--disable_recovery", dest="enable_recovery", action="store_false")
    parser.add_argument("--offtrack_dist_thresh", type=float, default=3.0)
    parser.add_argument("--offtrack_heading_thresh_deg", type=float, default=90.0)
    parser.add_argument("--offtrack_patience", type=int, default=2)
    parser.add_argument("--lookahead_k", type=int, default=1)
    parser.add_argument("--recovery_dist_thresh", type=float, default=2.0)
    parser.add_argument("--recovery_heading_thresh_deg", type=float, default=30.0)
    parser.add_argument("--recovery_max_steps", type=int, default=40)
    parser.add_argument("--goal_radius", type=float, default=3.0)
    parser.add_argument("--oracle_goal_radius", type=float, default=1.0)
    parser.add_argument("--heading_guard_dist", type=float, default=1.0)
    parser.add_argument("--goal_grace_steps", type=int, default=5)
    parser.add_argument("--goal_stop_patience", type=int, default=5)
    
    # Progress Stall Detection
    parser.add_argument("--no_progress_patience", type=int, default=60)
    parser.add_argument("--no_progress_enable", action="store_true", default=True)
    
    # SFT parameters
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients over N batches before optimizer update. "
                             "Effective batch size = num_episodes_per_update * gradient_accumulation_steps")
    
    # LoRA
    parser.add_argument("--no_lora", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"sft_only_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Create config
    config = SFTOnlyTrainingConfig(
        model_path=args.model_path,
        habitat_config_path=args.habitat_config_path,
        output_path=args.output_path,
        failure_episodes_path=args.failure_episodes_path,
        num_updates=args.num_updates,
        num_episodes_per_update=args.num_episodes_per_update,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        num_frames=args.num_frames,
        num_future_steps=args.num_future_steps,
        num_history=args.num_history,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        enable_recovery=args.enable_recovery,
        offtrack_dist_thresh=args.offtrack_dist_thresh,
        offtrack_heading_thresh_deg=args.offtrack_heading_thresh_deg,
        offtrack_patience=args.offtrack_patience,
        lookahead_k=args.lookahead_k,
        recovery_dist_thresh=args.recovery_dist_thresh,
        recovery_heading_thresh_deg=args.recovery_heading_thresh_deg,
        recovery_max_steps=args.recovery_max_steps,
        goal_radius=args.goal_radius,
        oracle_goal_radius=args.oracle_goal_radius,
        heading_guard_dist=args.heading_guard_dist,
        goal_grace_steps=args.goal_grace_steps,
        goal_stop_patience=args.goal_stop_patience,
        no_progress_patience=args.no_progress_patience,
        no_progress_enable=args.no_progress_enable,
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=4096,
        padding_side="right",
    )
    
    model_config = transformers.AutoConfig.from_pretrained(args.model_path)
    
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=model_config,
        low_cpu_mem_usage=False,
    )
    
    model.model.num_history = 8
    model.reset(1)
    model.to('cuda')
    
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
    
    # Apply LoRA
    lora_enabled = not args.no_lora
    if lora_enabled:
        if args.resume_from:
            print(f"Loading LoRA from {args.resume_from}")
            model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        else:
            print(f"Adding LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("⚠️ LoRA DISABLED: Full model training")
        for param in model.parameters():
            param.requires_grad = True
    
    # Create trainer
    trainer = StreamVLNSFTOnlyTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
    )
    
    # Resume if needed
    start_update = 0
    if args.resume_from:
        training_state_path = os.path.join(args.resume_from, "training_state.pt")
        
        if os.path.exists(training_state_path):
            print(f"Loading training state from {training_state_path}")
            training_state = torch.load(training_state_path, map_location='cuda')
            trainer.global_step = training_state['global_step']
            trainer.episode_count = training_state['episode_count']
            trainer.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            start_update = training_state['global_step']
            print(f"  Resumed from global_step={start_update}")
    
    # Train
    trainer.train(num_updates=args.num_updates, start_update=start_update)
    
    # Close wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
