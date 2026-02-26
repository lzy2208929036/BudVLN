"""
StreamVLN GRPO Training Script

This script implements GRPO (Group Relative Policy Optimization) for Vision-and-Language Navigation.
GRPO is an alternative to PPO that does not require a reference model and uses group-relative
advantages for more stable and sample-efficient training.

Key differences from PPO:
1. No reference model needed (saves 50% memory)
2. Group-based sampling (multiple trajectories per instruction)
3. Group-relative advantage computation (baseline = group mean)
4. No KL divergence calculation or early stopping
5. More stable training with simpler implementation

Training Phases:
- Phase 1: Stop optimization
- Phase 2: SPL/efficiency optimization
- Phase 3: Instruction alignment

Author: StreamVLN Team (GRPO Implementation)
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
import quaternion  # For pose calculations

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
from streamvln.rewards.vln_reward import VLNRewardFunction, RewardConfig, RewardPhase, compute_gae
from streamvln.utils.utils import (
    dict_to_cuda, DEFAULT_MEMORY_TOKEN, DEFAULT_VIDEO_TOKEN,
    DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
)
from streamvln.utils.dist import get_rank, get_world_size, init_distributed_mode

# Try importing wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


@dataclass
class GRPOTrainingConfig:
    """Configuration for GRPO training."""
    # Model and paths
    model_path: str = ""
    habitat_config_path: str = "config/vln_r2r.yaml"
    output_path: str = "./results/grpo_training"
    
    # GRPO-specific parameters
    group_size: int = 2                     # Number of trajectories per instruction group
    sampling_temperature: float = 1.0      # Sampling temperature for diversity
    baseline_type: str = "mean"            # 'mean' or 'median' for group baseline
    advantage_distribution: str = "uniform" # 'uniform' or 'temporal' for step allocation
    use_baseline_ema: bool = True          # Use exponential moving average for baseline
    baseline_ema_decay: float = 0.9        # EMA decay factor
    
    # Training parameters
    num_updates: int = 500                 # Total number of training updates
    num_episodes_per_update: int = 2       # Number of instructions per update
    # Actual samples per update = num_episodes * group_size
    ppo_epochs: int = 4                    # Can be higher than PPO (no KL constraint)
    mini_batch_size: int = 8               # Mini-batch size
    learning_rate: float = 1e-6            # Learning rate
    max_grad_norm: float = 0.5             # Gradient clipping
    
    # PPO parameters (kept for compatibility)
    gamma: float = 0.99                    # Discount factor (for temporal advantage distribution)
    gae_lambda: float = 0.95               # GAE lambda (not used in GRPO)
    clip_range: float = 0.2                # PPO clip range
    entropy_coef: float = 0.02             # Entropy bonus (slightly higher than PPO)
    
    # Note: No KL parameters needed for GRPO!
    # init_kl_coef: REMOVED
    # target_kl: REMOVED
    
    # Training phases
    phase: str = "phase1_stop"             # Current training phase
    
    # 🔥 Hybrid Training: GRPO + SFT
    use_hybrid_training: bool = False      # Enable hybrid training (GRPO + SFT on gt_actions)
    sft_use_policy_steps: bool = True      # Include policy steps (with valid gt) in SFT training
    sft_loss_start_weight: float = 1.0     # Initial SFT loss weight (high at beginning)
    sft_loss_end_weight: float = 0.5       # Final SFT loss weight (stable, not lower than GRPO)
    sft_loss_decay_updates: int = 200      # Number of updates for weight decay
    sft_loss_decay_type: str = "cosine"    # Decay schedule: 'linear', 'cosine', 'exponential'
    
    # 🔧 Debug options
    disable_grpo_loss: bool = False        # Disable GRPO policy loss (for debugging OOM)
    
    # 🔥 SFT-only Mode: Pure expert demonstration training
    sft_only: bool = False                 # Only train on oracle data (skip GRPO)
    
    # Environment
    num_envs: int = 1                      # Number of parallel environments
    max_steps_per_episode: int = 300       # Max steps per episode
    
    # Logging
    log_interval: int = 1                  # Log every N updates
    save_interval: int = 10                # Save model every N updates
    eval_interval: int = 5                 # Evaluate every N updates
    
    # LoRA settings
    lora_enable: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    
    # Model settings
    num_frames: int = 32
    num_future_steps: int = 4
    num_history: int = 8
    model_max_length: int = 4096
    
    # Mixed training (BC + RL)
    bc_coef: float = 0.0                   # Coefficient for BC loss mixing
    
    # TDR (Time-Decayed Reward) parameters
    use_tdr: bool = True                   # Enable TDR mechanism
    tdr_weight: float = 0.2                # TDR weight coefficient for reward fusion
    tdr_gamma: float = 0.9                 # TDR decay factor
    tdr_strict_mode: bool = True           # Stop TDR on first wrong action
    tdr_use_reference_path: bool = True    # Use reference_path waypoints for GT
    tdr_waypoint_threshold: float = 3.0    # Distance threshold to switch to next waypoint (meters)
    
    # 🔥 Expert Intervention (Off-track Detection & Recovery)
    enable_recovery: bool = True           # Master switch for expert intervention
    offtrack_dist_thresh: float = 3.0      # Distance threshold to trigger intervention (meters)
    offtrack_heading_thresh_deg: float = 90.0  # Heading error threshold to trigger intervention (degrees)
    offtrack_patience: int = 2             # Number of consecutive off-track steps before intervention
    lookahead_k: int = 1                   # Lookahead steps on reference path
    recovery_dist_thresh: float = 2.0      # Distance threshold to exit recovery (meters)
    recovery_heading_thresh_deg: float = 30.0  # Heading error threshold to exit recovery (degrees)
    recovery_max_steps: int = 40           # Maximum recovery steps (safety stop)
    
    # 🔥 Goal Zone Protection (prevents recovery near goal)
    goal_radius: float = 3.0               # Goal zone radius (meters) - matches reward config
    oracle_goal_radius: float = 1.0        # Oracle demo goal radius (meters) - closer to target for better learning
    heading_guard_dist: float = 1.0        # Minimum distance for heading check (meters)
    goal_grace_steps: int = 7              # Grace steps in goal zone before forced STOP
    goal_stop_patience: int = 5            # Steps to wait before forcing STOP in goal zone (0=disabled)
    
    # 🔥 NEW: Stuck Detection (agent stuck at same waypoint for too long)
    stuck_patience: int = 60               # Number of steps stuck at same waypoint before intervention
    stuck_enable: bool = True              # Enable stuck detection


class StreamVLNGRPOTrainer:
    """
    GRPO Trainer for StreamVLN.
    
    Key features:
    1. Group-based sampling: Multiple trajectories per instruction
    2. Group-relative advantages: advantage = reward - group_baseline
    3. No reference model or KL divergence
    4. Simplified and more stable training
    """
    
    def __init__(
        self,
        config: GRPOTrainingConfig,
        model: StreamVLNForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        # Note: No ref_model needed for GRPO!
        self.device = torch.device('cuda')
        
        # Setup reward function based on phase
        if reward_config is None:
            reward_phase = RewardPhase(config.phase)
            reward_config = RewardConfig(phase=reward_phase).get_phase_config()
        self.reward_fn = VLNRewardFunction(reward_config)
        
        # NOTE: GRPO does not use a value estimator (value head removed)
        # Setup optimizer (only model parameters are trainable)
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        num_trainable_model = sum(p.numel() for p in trainable_params)
        print(f"✅ GRPO Trainer initialized (no reference model, no value estimator!)")
        print(f"Trainable model parameters: {num_trainable_model:,}")
        print(f"Group size: {config.group_size} (samples per instruction)")
        print(f"Effective batch size: {config.num_episodes_per_update * config.group_size}")
        
        # 🔥 Hybrid Training info
        if config.use_hybrid_training:
            print(f"\n🔥 Hybrid Training Enabled:")
            print(f"   SFT weight: {config.sft_loss_start_weight:.2f} → {config.sft_loss_end_weight:.2f}")
            print(f"   Decay schedule: {config.sft_loss_decay_type} over {config.sft_loss_decay_updates} updates")
            sft_sources = "Oracle only" if not config.sft_use_policy_steps else "Oracle + Policy (with valid GT)"
            print(f"   SFT data sources: {sft_sources}")
        
        print(flush=True)
        
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01
        )
        
        # === Hybrid Training State ===
        self.current_update = 0  # Track current update for dynamic weight scheduling
        
        # Setup Habitat environment
        self.habitat_config = get_habitat_config(config.habitat_config_path)
        self._setup_habitat_config()
        
        # GRPO-specific: baseline EMA (for logging and optional use)
        self.baseline_ema = 0.0
        
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
        
        # Action token IDs (same as PPO)
        self.action_token_ids = self._get_action_token_ids()
        
        # 🔥 NEW: Action mapping for parsing (same as PPO)
        self.actions2idx = {
            'STOP': [0],
            '↑': [1],
            '←': [2],
            '→': [3],
        }
        
        # Fixed conjunction to match RL exactly
        self.conjunction = 'you can see '
        
        # 🔥 TDR: Configuration for ground truth action computation
        if config.use_tdr:
            strategy = "reference_path" if config.tdr_use_reference_path else "goal_direct"
            print(f"✅ TDR enabled: weight={config.tdr_weight}, gamma={config.tdr_gamma}, strict={config.tdr_strict_mode}")
            print(f"   Strategy: {strategy}, waypoint_threshold={config.tdr_waypoint_threshold}m")
        else:
            print(f"⚡ TDR disabled: Skipping GT action computation (performance optimized)")
    
    def _setup_habitat_config(self):
        """Setup Habitat environment configuration."""
        with read_write(self.habitat_config):
            # Set task config with proper measurement configs
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
            
            # Enable RGB sensor
            self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 224
            self.habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 224
            
            # Set max episode steps
            self.habitat_config.habitat.environment.max_episode_steps = self.config.max_steps_per_episode
    
    def _get_conversation_template(self) -> List[Dict]:
        """Get conversation template for instruction."""
        # Match RL trainer exactly
        prompt = (
            "<video>\nYou are an autonomous navigation assistant. "
            "Your task is to <instruction>. "
            "Devise an action sequence to follow the instruction using the four actions: "
            "TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
            "MOVE FORWARD (↑) by 25 centimeters, or STOP."
        )
        return [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]
    
    # ========== Helper methods for depth/pose/intrinsic processing (matching eval) ==========
    
    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        """Preprocess depth image to match model input size."""
        from transformers.image_utils import to_numpy_array
        target_height = self.image_processor.crop_size['height']  # 384
        target_width  = self.image_processor.crop_size['width']  # 384
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
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):
        """Adjust intrinsic matrix after image resizing."""
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        """Get axis alignment matrix for coordinate transformation."""
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        """Convert xyz position and yaw to transformation matrix."""
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix
    
    # ========== End of helper methods ==========
    
    def _get_action_token_ids(self) -> torch.Tensor:
        """Get action token IDs for the 4 valid actions."""
        action_token_ids = torch.tensor([
            self.tokenizer.encode("STOP", add_special_tokens=False)[0],
            self.tokenizer.encode("↑", add_special_tokens=False)[0],
            self.tokenizer.encode("←", add_special_tokens=False)[0],
            self.tokenizer.encode("→", add_special_tokens=False)[0],
        ], dtype=torch.long, device=self.device)
        return action_token_ids
    
    def train(self, num_updates: int, start_update: int = 0):
        """
        Main GRPO training loop.
        
        Args:
            num_updates: Total number of updates to perform
            start_update: Starting update (for resuming)
        """
        print(f"\n🚀 Starting GRPO Training")
        print(f"=" * 80)
        print(f"Phase: {self.config.phase}")
        print(f"Updates: {start_update} -> {num_updates}")
        print(f"Group size: {self.config.group_size}")
        print(f"Episodes per update: {self.config.num_episodes_per_update}")
        print(f"Total trajectories per update: {self.config.num_episodes_per_update * self.config.group_size}")
        
        # 🔥 Print Expert Intervention Configuration
        if self.config.enable_recovery:
            print(f"\n🔥 Expert Intervention: ENABLED")
            print(f"  Off-track Detection:")
            print(f"    Distance threshold: {self.config.offtrack_dist_thresh:.1f}m")
            print(f"    Heading threshold: {self.config.offtrack_heading_thresh_deg:.0f}°")
            print(f"    Heading guard distance: {self.config.heading_guard_dist:.1f}m")
            print(f"    Patience: {self.config.offtrack_patience} steps")
            print(f"  Recovery Parameters:")
            print(f"    Lookahead: {self.config.lookahead_k} waypoint(s)")
            print(f"    Exit distance: {self.config.recovery_dist_thresh:.1f}m")
            print(f"    Exit heading: {self.config.recovery_heading_thresh_deg:.0f}°")
            print(f"    Max steps: {self.config.recovery_max_steps}")
            print(f"  Goal Zone Protection:")
            print(f"    Goal radius: {self.config.goal_radius:.1f}m")
            print(f"    Grace steps: {self.config.goal_grace_steps}")
            if self.config.goal_stop_patience > 0:
                print(f"    Force STOP patience: {self.config.goal_stop_patience} steps")
            else:
                print(f"    Force STOP: disabled")
        else:
            print(f"\n🔥 Expert Intervention: DISABLED")
        
        print(f"=" * 80 + "\n")
        
        # 🔥 Set random seed for reproducible-yet-different runs
        # Use current time to ensure different sampling order across runs
        random_seed = int(time.time() * 1000) % (2**31)
        random.seed(random_seed)
        np.random.seed(random_seed)
        print(f"🎲 Random seed: {random_seed} (based on current time)")
        print(f"   This ensures different episode sampling order across runs.\n")
        
        print("DEBUG: Creating Habitat environments...")
        sys.stdout.flush()
        
        # 🔥 FIX: Create ONE environment and reuse it across groups
        # This allows env.reset() to cycle through different episodes
        self._train_env = habitat.Env(config=self.habitat_config)
        
        # 🔥 NEW: Store all episodes and shuffle them for random sampling
        # This ensures each run has different episode order
        self._all_episodes = list(self._train_env.episode_iterator.episodes)
        random.shuffle(self._all_episodes)  # Shuffle episodes for random order
        self._episode_index = 0
        print(f"✅ Loaded {len(self._all_episodes)} episodes (shuffled for random sampling)")
        
        # 🔥 Initialize ShortestPathFollower for GT action computation (used by both TDR and Recovery)
        self.shortest_path_follower = ShortestPathFollower(
            sim=self._train_env.sim,
            goal_radius=3.0,  # 🔥 修改为3.0米，与reward_config的goal_radius一致
            return_one_hot=False
        )
        self.shortest_path_follower.mode = 'geodesic_path'
        if self.config.use_tdr:
            print(f"✅ TDR enabled: ShortestPathFollower initialized (goal_radius={3.0}m)")
        print(f"✅ Expert Intervention enabled: ShortestPathFollower ready for recovery")
        
        # 🔥 Training loop: Ensure num_updates effective updates (skip empty oracle batches)
        update_id = start_update
        completed_updates = 0
        
        while completed_updates < num_updates - start_update:
            update_start_time = time.time()
            
            # 🔥 Update current_update for dynamic SFT weight scheduling
            self.current_update = update_id
            
            print(f"\n{'='*80}")
            print(f"Update {update_id+1}/{num_updates}")
            if self.config.sft_only:
                print(f"🔥 SFT-only Mode: Pure expert demonstration training")
            elif self.config.use_hybrid_training:
                sft_weight = self._get_sft_weight()
                print(f"Hybrid Training: SFT weight = {sft_weight:.3f}")
            
            # 🔥 Reset peak memory stats at start of update
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            print(f"{'='*80}")
            
            # Step 1: Collect trajectory groups
            print(f"\n📊 Collecting {self.config.num_episodes_per_update} instruction groups...")
            all_trajectories = []
            
            for ep_idx in range(self.config.num_episodes_per_update):
                print(f"\n  Group {ep_idx+1}/{self.config.num_episodes_per_update}:")
                
                # Collect a group of trajectories for one instruction
                group_trajectories = self.collect_trajectory_group(update_id)
                
                # Print group statistics
                group_returns = [sum(traj['rewards']) for traj in group_trajectories]
                print(f"  Group returns: {[f'{r:.2f}' for r in group_returns]}")
                print(f"  Mean: {np.mean(group_returns):.2f}, Std: {np.std(group_returns):.2f}")
                
                all_trajectories.extend(group_trajectories)
            
            print(f"\n✅ Collected {len(all_trajectories)} trajectories total")
            
            # 🔥 Expert Intervention: Aggregate recovery statistics
            total_oracle_steps = sum(traj['recovery_stats']['oracle_steps'] for traj in all_trajectories)
            total_policy_steps = sum(traj['recovery_stats']['policy_steps'] for traj in all_trajectories)
            total_recovery_triggered = sum(traj['recovery_stats']['recovery_triggered'] for traj in all_trajectories)
            total_recovery_success = sum(traj['recovery_stats']['recovery_success'] for traj in all_trajectories)
            
            print(f"  Oracle steps: {total_oracle_steps}, Policy steps: {total_policy_steps}")
            if total_recovery_triggered > 0:
                print(f"  Recovery: {total_recovery_success}/{total_recovery_triggered} succeeded")
            
            # 🔥 SFT-only Mode: Skip GRPO, only train on oracle data
            if self.config.sft_only:
                # Extract oracle experiences only
                sft_experiences = []
                for traj in all_trajectories:
                    for i, action_source in enumerate(traj['action_sources']):
                        if action_source == 'oracle':
                            sft_experiences.append({
                                'state': traj['states'][i],
                                'action': traj['actions'][i],
                                'instruction': traj['instruction'],
                                'action_source': action_source,
                                'gt_action': traj['gt_actions'][i],
                            })
                
                if len(sft_experiences) == 0:
                    print(f"  ⚠️ No oracle data collected (model stayed on track), resampling...")
                    update_id += 1  # Increment attempt counter
                    continue  # Don't increment completed_updates, resample
                
                print(f"\n🔄 Performing SFT-only update ({len(sft_experiences)} oracle samples)...")
                stats = self._sft_only_update(sft_experiences)
            else:
                # Step 2: Compute GRPO advantages
                print(f"\n🎯 Computing GRPO advantages...")
                all_trajectories = self.compute_grpo_advantages_batch(all_trajectories)
                
                # Step 3: GRPO update (PPO without KL)
                print(f"\n🔄 Performing GRPO update...")
                stats = self._grpo_update(all_trajectories)
            
            # Add recovery stats to training stats
            stats['oracle_steps'] = total_oracle_steps
            stats['policy_steps'] = total_policy_steps
            stats['recovery_triggered'] = total_recovery_triggered
            stats['recovery_success'] = total_recovery_success
            
            # Step 4: Logging
            update_time = time.time() - update_start_time
            self._log_update(completed_updates + start_update, stats, update_time)
            
            # 🔥 GPU Memory Stats
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                reserved_memory = torch.cuda.memory_reserved() / 1024**3  # GB
                print(f"💾 GPU Memory: Current={current_memory:.2f}GB, Peak={peak_memory:.2f}GB, Reserved={reserved_memory:.2f}GB")
            
            # Step 5: Save checkpoint
            if (completed_updates + 1) % self.config.save_interval == 0:
                self.save_checkpoint(completed_updates + start_update + 1)
            
            # Update counters
            completed_updates += 1
            update_id += 1
            self.global_step = completed_updates + start_update
        
        # 🔥 FIX: Close environment at end of training
        if hasattr(self, '_train_env') and self._train_env is not None:
            self._train_env.close()
        
        print(f"\n🎉 Training completed!")
        print(f"Final checkpoint saved to: {self.config.output_path}/checkpoint_final")
        self.save_checkpoint('final')
    
    def collect_trajectory_group(self, update_id: int) -> List[Dict]:
        """
        Collect a group of trajectories for the same instruction.
        
        === GRPO核心机制：Group-Based Sampling ===
        这是GRPO的核心：对同一条instruction采样多条轨迹进行组内比较。
        
        === 🔥 Greedy Anchor采样策略 ===
        - 第0个样本: do_sample=False (贪婪解码) → 模型当前能力上限
        - 第1-N个样本: do_sample=True, temperature=0.5 → 探索
        
        工作流程：
        1. 从环境获取一个episode（包含起点、目标、指令）
        2. 对同一个起始状态，使用模型生成group_size条不同轨迹
        3. 每条轨迹通过不同的采样策略获得多样性（deterministic vs stochastic）
        4. 收集到的多条轨迹将用于计算group-relative advantage
        
        === 采样多样性来源 ===
        - do_sample=True时：模型从action概率分布中采样（非贪婪）
        - do_sample=False时：模型选择概率最高的action（贪婪/deterministic）
        - temperature参数：控制概率分布的平滑程度（越高越随机）
        
        注意：GRPO不依赖随机种子来产生多样性，而是通过模型的采样过程。
        
        Args:
            update_id: 当前update索引，用于exploration scheduling（早期探索，后期利用）
        
        Returns:
            List of trajectory dicts，每个dict包含：
            - states: 状态序列（RGB图像、step_id、生成历史等）
            - actions: 动作序列
            - rewards: 奖励序列
            - advantages: 优势值（后续计算）
            - gt_actions: ground truth动作（用于TDR）
        """
        # 🔥 使用共享环境并从打乱的episode列表中获取下一个episode
        env = self._train_env
        
        # 🔥 NEW: Get episode from shuffled list (循环使用)
        episode = self._all_episodes[self._episode_index % len(self._all_episodes)]
        self._episode_index += 1
        
        # 设置episode并重置环境
        env._current_episode = episode
        observations = env.reset()
        
        instruction = episode.instruction.instruction_text
        episode_id = episode.episode_id
        
        print(f"  Instruction (ep={episode_id}): {instruction[:80]}...")
        
        trajectories = []
        
        # === Group采样循环：对同一instruction收集多条轨迹 ===
        for sample_idx in range(self.config.group_size):
            # === 🔥 Greedy Anchor采样策略 ===
            # 第0个样本: 总是greedy (deterministic=True, do_sample=False)
            #   → 模型当前能力上限，提供稳定的baseline
            # 第1-N个样本: temperature=0.5探索
            #   → 平衡探索与利用，避免崩溃
            deterministic = (sample_idx == 0)  # 🔥 只有第0个样本是greedy
            
            # 🔥 重置到相同的episode（相同起点、目标、指令）
            # 这确保group内的轨迹有相同的任务条件，可以公平比较
            env._current_episode = episode
            observations = env.reset()
            
            # 注意：不设置随机种子！
            # GRPO的多样性来自模型的stochastic sampling，而非环境随机性
            
            # Collect one trajectory
            traj = self._collect_single_trajectory(
                env=env,
                instruction=instruction,
                deterministic=deterministic,  # Use exploration scheduling
                temperature=self.config.sampling_temperature,
            )
            
            trajectories.append(traj)
            
            # Quick stats
            total_reward = sum(traj['rewards'])
            num_steps = len(traj['actions'])
            success = traj.get('final_metrics', {}).get('success', 0.0)
            print(f"    Sample {sample_idx+1}: reward={total_reward:.2f}, steps={num_steps}, success={success:.0f}")
        
        # 🔥 FIX: Don't close env - we reuse it across groups
        return trajectories
    
    def _collect_single_trajectory(
        self,
        env: habitat.Env,
        instruction: str,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Dict:
        """
        收集单条轨迹：使用model.generate()进行autoregressive生成。
        
        === Autoregressive生成机制（关键！） ===
        与单步forward不同，这里使用generate()一次生成多个动作，然后逐步执行。
        这保留了模型在预训练时学到的序列规划能力。
        
        工作流程：
        1. 当action_seq缓冲为空时，调用model.generate()生成一段动作序列
        2. 将生成的动作序列解析为具体动作（↑←→ STOP）
        3. 从序列中pop一个动作，执行到环境
        4. 将执行的动作token加入generated_ids_so_far（autoregressive上下文）
        5. 重复2-4，直到缓冲为空或达到num_frames边界
        6. 到达边界时，重置生成状态，重新生成下一段动作序列
        
        === 采样控制 ===
        - deterministic=True: 使用argmax选择概率最高的token（贪婪解码）
        - deterministic=False: 从概率分布中采样token（随机解码）
        - temperature: 控制采样的随机性（仅在do_sample=True时有效）
          * temperature=1.0: 原始概率分布
          * temperature>1.0: 更平滑的分布（更随机）
          * temperature<1.0: 更尖锐的分布（更确定）
        
        === 训练-推理一致性 ===
        rollout时保存generated_ids_so_far（已生成的动作token序列），
        训练时将其作为prefix拼接到input_ids，确保训练时的context与生成时一致。
        
        Args:
            env: Habitat环境实例
            instruction: 导航指令文本
            deterministic: 是否使用deterministic解码
            temperature: 采样温度（当前版本未直接使用，由do_sample控制）
        
        Returns:
            trajectory dict包含：
            - states: 每步的状态（RGB + step_id + generated_ids_so_far）
            - actions: 动作序列
            - rewards: 奖励序列
            - gt_actions: 最优动作序列（用于TDR）
            - distances: 到目标的距离序列
        """
        # === 采样策略 ===
        # deterministic=True:  greedy decoding (do_sample=False)
        # deterministic=False: temperature sampling (do_sample=True, temperature=0.7)
        # 
        # 注意：不使用model.train()切换，因为会导致模型不稳定（dropout随机性太强）
        #       训练时始终保持model.eval()，只在采样参数上调整
        self.model.eval()  # 始终使用eval模式（推理模式）
            
        self.reward_fn.reset()
        
        # 🔥 CRITICAL: Reset model internal state before each trajectory
        # Without this, previous trajectory's state bleeds into the next one
        self._base_model.reset_for_env(0)
        
        observations = env.reset()
        episode = env.current_episode
        
        # Get reference path for reward computation
        reference_path = np.array(episode.reference_path)
        goal_position = episode.goals[0].position
        
        # 🔥 TDR混合策略: 构建完整路径 (reference_path + goal)
        # Only needed when TDR is enabled
        if self.config.use_tdr:
            full_path = np.vstack([reference_path, goal_position])
            current_waypoint_idx = 0  # 当前目标waypoint索引
        else:
            full_path = None  # Not needed when TDR disabled
            current_waypoint_idx = None
        
        # Initialize trajectory storage
        trajectory = {
            'states': [],
            'actions': [],
            'action_token_ids': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'old_log_probs': [],
            'dones': [],
            'infos': [],
            'distances': [],
            'instruction': instruction,
            'episode_id': episode.episode_id,
            'gt_actions': [],  # 🔥 TDR: Ground truth actions
            'action_sources': [],  # 🔥 Expert Intervention: 'policy' or 'oracle'
        }
        
        # Visual memory (RGB only, matching RL trainer)
        rgb_list = []
        time_ids = []
        
        # 🔥 NEW: Generation state (matching PPO and eval)
        step_id = 0
        output_ids = None
        past_key_values = None
        action_seq = []  # Action buffer from generated sequence
        generated_action_ids = []  # Track generated action tokens for autoregressive context
        
        is_done = False
        prev_distance = env.get_metrics().get('distance_to_goal', float('inf'))
        
        # 🔥 Expert Intervention: Off-track detection and recovery state
        last_progress_idx = 0
        offtrack_count = 0
        in_recovery = False
        recovery_steps = 0
        recovery_triggered_count = 0  # Statistics
        recovery_success_count = 0    # Statistics
        oracle_takeover_active = False  # 🔥 SFT-only: Once triggered, expert controls till end
        steps_in_goal_zone = 0        # Goal zone tracking
        steps_after_grace = 0         # Steps after grace period
        
        # 🔥 NEW: Waypoint state tracking for rollback learning
        last_waypoint_state = None  # State when last passing a waypoint
        prev_nearest_idx = -1       # Track waypoint transitions
        oracle_demonstrations = []  # Store oracle demos for rollback learning
        
        # 🔥 NEW: Stuck detection tracking
        steps_at_current_waypoint = 0  # Counter for steps at same waypoint
        
        # Collect trajectory with generate-based sampling
        with torch.no_grad():
            while not is_done and step_id < self.config.max_steps_per_episode:
                # Get current metrics
                curr_distance = env.get_metrics().get('distance_to_goal', float('inf'))
                
                # 🔥 Unified agent state acquisition (used by recovery + TDR)
                agent_state = env.sim.get_agent_state()
                agent_position = agent_state.position
                agent_rotation = agent_state.rotation
                
                # 🔥 Expert Intervention: Off-track detection with goal zone protection
                if self.config.enable_recovery:
                    # 🔥 Goal Zone Protection (Method A + Grace Period)
                    near_goal = curr_distance <= self.config.goal_radius
                    
                    if near_goal:
                        # In goal zone
                        steps_in_goal_zone += 1
                        offtrack_count = 0  # Reset off-track counter
                        
                        # Check grace period status
                        if steps_in_goal_zone <= self.config.goal_grace_steps:
                            # Within grace period: no forced STOP
                            steps_after_grace = 0
                        else:
                            # After grace period: start counting towards forced STOP
                            steps_after_grace += 1
                        
                        if in_recovery:
                            # 🔥 SFT-only mode: Don't exit recovery in goal zone (oracle controls till end)
                            if not oracle_takeover_active:
                                # Force exit recovery in goal zone (normal mode)
                                in_recovery = False
                                print(f"    [Step {step_id}] Near goal ({curr_distance:.2f}m), exiting recovery")
                                
                                # Clear model state
                                self._base_model.reset_for_env(0)
                                output_ids = None
                                past_key_values = None
                                action_seq = []
                                generated_action_ids = []
                            else:
                                # SFT-only: Keep oracle control, navigate to goal
                                print(f"    [Step {step_id}] Near goal ({curr_distance:.2f}m), oracle continues to goal")
                    else:
                        # Outside goal zone: reset counters and normal off-track detection
                        steps_in_goal_zone = 0
                        steps_after_grace = 0
                        
                        # Find nearest point on reference path
                        nearest_idx = self._nearest_path_index(agent_position, reference_path)
                        last_progress_idx = max(nearest_idx, last_progress_idx)
                        
                        # 🔥 NEW: Record state when passing a waypoint (for potential rollback)
                        if nearest_idx != prev_nearest_idx:
                            # Agent moved to a new waypoint - save this state
                            last_waypoint_state = {
                                'position': list(agent_position),
                                'rotation': agent_rotation,  # quaternion
                                'step_id': step_id,
                                'nearest_idx': nearest_idx,
                                'observations': observations.copy(),  # Current RGB
                                'rgb_list': rgb_list.copy(),  # Visual history
                                'time_ids': time_ids.copy(),
                                'trajectory_length': len(trajectory['actions']),
                            }
                            prev_nearest_idx = nearest_idx
                            steps_at_current_waypoint = 0  # 🔥 Reset stuck counter
                            print(f"    [Step {step_id}] Recorded waypoint {nearest_idx} state for rollback")
                        else:
                            # Still at same waypoint
                            steps_at_current_waypoint += 1
                        
                        # Select target waypoint with lookahead
                        target_waypoint = self._select_target_waypoint(
                            reference_path, nearest_idx, last_progress_idx, goal_position
                        )
                        
                        # Calculate deviation metrics
                        # 🔥 Use segment-based distance to handle sparse waypoints correctly
                        dist_to_ref_path = self._dist_to_reference_path(
                            agent_position, reference_path, nearest_idx
                        )
                        heading_error = self._heading_error_deg(agent_rotation, agent_position, target_waypoint)
                        
                        # 🔥 Improved Off-track Detection (Method B)
                        # Heading check only applies when far from path
                        offtrack_now = (
                            dist_to_ref_path > self.config.offtrack_dist_thresh or
                            (heading_error > self.config.offtrack_heading_thresh_deg and 
                             dist_to_ref_path > self.config.heading_guard_dist)
                        )
                        
                        # Update off-track counter
                        if offtrack_now:
                            offtrack_count += 1
                        else:
                            offtrack_count = 0
                        
                        # 🔥 NEW: Stuck Detection - agent stuck at same waypoint for too long
                        stuck_detected = (
                            self.config.stuck_enable and 
                            steps_at_current_waypoint >= self.config.stuck_patience and
                            not in_recovery and
                            not near_goal  # Don't trigger in goal zone
                        )
                        
                        if stuck_detected:
                            # Agent is stuck at same waypoint - trigger rollback learning
                            print(f"\n  [Step {step_id}] Stuck detected! {steps_at_current_waypoint} steps at waypoint {nearest_idx}")
                            
                            # 🔥 Check if episode is already done (e.g., max steps reached or env.episode_over)
                            if is_done or observations.get('done', False) or env.episode_over:
                                print(f"    Episode already done (step_id={step_id}, env.episode_over={env.episode_over}), skipping oracle demo collection")
                            else:
                                if last_waypoint_state is None:
                                    # No waypoint state recorded yet - use episode start
                                    print(f"    No waypoint state, using episode start for rollback")
                                    rollback_state = {
                                        'position': list(episode.start_position),
                                        'rotation': episode.start_rotation,
                                        'step_id': 0,
                                        'nearest_idx': 0,
                                    }
                                else:
                                    rollback_state = last_waypoint_state
                                    print(f"    Rolling back to waypoint {rollback_state['nearest_idx']} (step {rollback_state['step_id']})")
                                
                                # Collect oracle demonstration from rollback point
                                oracle_demo = self._collect_oracle_demonstration(
                                    env=env,
                                    rollback_state=rollback_state,
                                    goal_position=goal_position,
                                    reference_path=reference_path,
                                )
                                
                                if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                    oracle_demonstrations.append(oracle_demo)
                                    print(f"    Collected oracle demo: {len(oracle_demo['rgbs'])} steps from rollback point to goal")
                                    recovery_triggered_count += 1
                            
                            # Episode ends after stuck detection
                            print(f"    Episode ends after stuck detection\n")
                            is_done = True
                            break
                        
                        # Trigger recovery if persistent off-track
                        if not in_recovery and offtrack_count >= self.config.offtrack_patience:
                            # 🔥 NEW: Rollback learning instead of inline recovery
                            print(f"\n  [Step {step_id}] Offtrack detected! Distance={dist_to_ref_path:.2f}m, Heading={heading_error:.1f}°")
                            
                            # 🔥 Check if episode is already done (e.g., max steps reached or env.episode_over)
                            if is_done or observations.get('done', False) or env.episode_over:
                                print(f"    Episode already done (step_id={step_id}, env.episode_over={env.episode_over}), skipping oracle demo collection")
                            else:
                                if last_waypoint_state is None:
                                    # No waypoint state recorded yet - use episode start
                                    print(f"    No waypoint state, using episode start for rollback")
                                    rollback_state = {
                                        'position': list(episode.start_position),
                                        'rotation': episode.start_rotation,
                                        'step_id': 0,
                                        'nearest_idx': 0,
                                    }
                                else:
                                    rollback_state = last_waypoint_state
                                    print(f"    Rolling back to waypoint {rollback_state['nearest_idx']} (step {rollback_state['step_id']})")
                                
                                # Collect oracle demonstration from rollback point
                                oracle_demo = self._collect_oracle_demonstration(
                                    env=env,
                                    rollback_state=rollback_state,
                                    goal_position=goal_position,
                                    reference_path=reference_path,
                                )
                                
                                if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                    oracle_demonstrations.append(oracle_demo)
                                    print(f"    Collected oracle demo: {len(oracle_demo['rgbs'])} steps from rollback point to goal")
                                    recovery_triggered_count += 1
                            
                            # Episode ends after offtrack detection
                            print(f"    Episode ends after offtrack detection\n")
                            is_done = True
                            break
                        
                        # Check recovery exit conditions (old logic, now unused with rollback)
                        if in_recovery:
                            recovery_steps += 1
                            # Use the same segment-based distance for exit check
                            success_recover = (
                                dist_to_ref_path < self.config.recovery_dist_thresh and
                                heading_error < self.config.recovery_heading_thresh_deg
                            )
                            safety_stop = recovery_steps >= self.config.recovery_max_steps
                            
                            # 🔥 SFT-only mode: Don't exit recovery (oracle controls till end)
                            if success_recover and not oracle_takeover_active:
                                in_recovery = False
                                recovery_success_count += 1
                                recovery_fallback_count = 0
                                recovery_heading_history = []
                                print(f"    [Step {step_id}] Recovery succeeded after {recovery_steps} steps")
                                
                                # 🔥 Clear model state when exiting recovery (model will regenerate from clean slate)
                                self._base_model.reset_for_env(0)
                                output_ids = None
                                past_key_values = None
                                action_seq = []
                                generated_action_ids = []
                            elif safety_stop:
                                in_recovery = False
                                oracle_takeover_active = False
                                print(f"    [Step {step_id}] Recovery safety stop after {recovery_steps} steps")
                                # Force episode termination on safety stop
                                is_done = True
                                break
                
                # 🔥 TDR: Get ground truth action for current state (for reward calculation)
                # ⚡ Performance: Skip GT action computation when TDR is disabled
                if self.config.use_tdr:
                    try:
                        if self.config.tdr_use_reference_path and current_waypoint_idx < len(full_path):
                            # 混合策略: 根据与路径的距离选择GT目标
                            tdr_target = full_path[current_waypoint_idx]
                            dist_to_waypoint = np.linalg.norm(agent_position - tdr_target)
                            
                            # 接近当前waypoint，前进到下一个
                            if dist_to_waypoint < self.config.tdr_waypoint_threshold:
                                if current_waypoint_idx < len(full_path) - 1:
                                    current_waypoint_idx += 1
                                    tdr_target = full_path[current_waypoint_idx]
                            
                            # 计算到目标waypoint的GT动作
                            gt_action = self.shortest_path_follower.get_next_action(tdr_target)
                        else:
                            # Fallback: 直接到goal (当没有路径或已完成路径时)
                            gt_action = self.shortest_path_follower.get_next_action(goal_position)
                        
                        # Convert None to 0 (STOP)
                        if gt_action is None:
                            gt_action = 0
                    except Exception as e:
                        # Fallback: use STOP if path finding fails
                        gt_action = 0
                else:
                    # ⚡ TDR disabled: Skip shortest_path_follower computation
                    gt_action = -1  # Placeholder (not used in advantage/loss calculation)
                
                # Get RGB observation (matching RL trainer - RGB only)
                rgb = observations['rgb']
                
                # Process RGB only (matching RL trainer)
                image = Image.fromarray(rgb).convert('RGB')
                image_tensor = self.image_processor.preprocess(
                    images=image, return_tensors='pt'
                )['pixel_values'][0]
                
                # Append to list
                rgb_list.append(image_tensor)
                time_ids.append(step_id)
                
                # === 🔥 Expert Intervention: Action Selection ===
                action_source = "oracle" if in_recovery else "policy"
                
                if in_recovery:
                    # 🔥 Recovery: Follow reference_path to get back on track
                    # Strategy: Navigate to the next waypoint on reference_path
                    
                    # Select target: nearest waypoint ahead on reference_path
                    # Use lookahead to avoid getting stuck on already-passed waypoints
                    target_idx = min(nearest_idx + self.config.lookahead_k, len(reference_path) - 1)
                    target_waypoint_on_path = reference_path[target_idx]
                    
                    # Get action from ShortestPathFollower
                    action = self.shortest_path_follower.get_next_action(target_waypoint_on_path)
                    
                    # If current waypoint unreachable, try next waypoints (skip obstacles)
                    if action is None and target_idx + 1 < len(reference_path):
                        target_waypoint_on_path = reference_path[target_idx + 1]
                        action = self.shortest_path_follower.get_next_action(target_waypoint_on_path)
                    
                    # If still None, try one more ahead
                    if action is None and target_idx + 2 < len(reference_path):
                        target_waypoint_on_path = reference_path[target_idx + 2]
                        action = self.shortest_path_follower.get_next_action(target_waypoint_on_path)
                    
                    # Final fallback: navigate directly to goal (end of reference_path)
                    if action is None:
                        action = self.shortest_path_follower.get_next_action(goal_position)
                    
                    # Handle complete failure
                    if action is None:
                        # Goal unreachable, force STOP
                        action = 0
                        print(f"    [Step {step_id}] Recovery: All targets unreachable, forcing STOP")
                        is_done = True
                    
                    action_token_id = self.action_token_ids[action].item()
                    
                    # 🔥 Oracle steps don't participate in TDR (they're expert actions, not model predictions)
                    # TDR should only evaluate model-generated actions
                    gt_action = -1
                else:
                    # === 动作序列生成：Action Sequence Buffering ===
                    # 当action_seq缓冲为空时，调用generate()生成新的动作序列
                    # 这是autoregressive rollout的核心机制
                    if len(action_seq) == 0:
                        # 构建输入（完全匹配eval行为）
                        if output_ids is None:
                            # === 首次生成：完整prompt ===
                            # 包含系统提示、指令、视觉token等
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
                            # Continuation - empty prompt
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                        
                        # Tokenize
                        input_ids, _ = self._preprocess_qwen([sources], add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)
                        
                        # Prepare visual inputs (RGB only, matching RL trainer)
                        images = rgb_list[-1:]
                        
                        # Add history if at num_frames boundary
                        if step_id != 0 and step_id % self.config.num_frames == 0:
                            if self.config.num_history is None:
                                history_ids = slice(0, time_ids[0], self.config.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], (time_ids[0] // self.config.num_history))
                            images = rgb_list[history_ids] + images
                        
                        # Build input dict (matching RL trainer exactly)
                        input_dict = {
                            'images': torch.stack(images).unsqueeze(0),
                            'depths': None,
                            'poses': None,
                            'intrinsics': None,
                            'inputs': input_ids,
                            'env_id': 0,
                            'time_ids': [time_ids],
                            'task_type': [0],
                        }
                        
                        input_dict = dict_to_cuda(input_dict, self.device)
                        input_dict['images'] = input_dict['images'].to(torch.bfloat16)
                        
                        # === 调用LLM生成动作序列 ===
                        # 🔥 Greedy Anchor采样策略：
                        #   - deterministic=True:  greedy decoding (do_sample=False) - 用于第0个样本
                        #   - deterministic=False: temperature=0.5 sampling - 用于探索样本
                        with torch.no_grad():  # 不计算梯度（推理模式）
                            outputs = self.model.generate(
                                **input_dict,
                                # === 🔥 采样策略 ===
                                do_sample=(not deterministic),  # deterministic时greedy，否则采样
                                temperature=0.5 if not deterministic else 1.0,  # 🔥 降低温度到0.5，减少探索噪声
                                num_beams=1,                    # 不使用beam search
                                max_new_tokens=10000,           # 最大生成长度
                                # === 效率优化 ===
                                use_cache=True,                 # 使用KV cache加速
                                return_dict_in_generate=True,   # 返回详细信息
                                past_key_values=past_key_values,# 复用KV cache
                                output_scores=True,             # 记录生成scores（可选）
                                # === 其他配置 ===
                                use_progress_guidance=False,    # 不使用进度引导
                            )
                        
                        # Update generation state
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        
                        # Parse actions from output
                        llm_output = self.tokenizer.batch_decode(
                            output_ids, skip_special_tokens=False
                        )[0].strip()
                        
                        action_seq = self._parse_actions(llm_output)
                        
                        if len(action_seq) == 0:
                            action_seq = [0]  # Default to STOP if no actions parsed
                    
                    # === 从缓冲中取出一个动作执行 ===
                    # FIFO顺序：按生成顺序执行动作
                    action = action_seq.pop(0)
                    action_token_id = self.action_token_ids[action].item()
                
                # 🔥 Optional: Force STOP in goal zone (after grace period)
                # When the model doesn't stop in goal zone after grace+patience, we:
                # 1. Collect oracle demonstration from last waypoint to goal
                # 2. Force episode to end
                if (self.config.enable_recovery and 
                    self.config.goal_stop_patience > 0 and
                    near_goal and
                    steps_in_goal_zone > self.config.goal_grace_steps and
                    not in_recovery and
                    action_source == 'policy'):
                    
                    if steps_after_grace >= self.config.goal_stop_patience and action != 0:
                        # Model hasn't stopped after grace+patience expired
                        print(f"\n  [Step {step_id}] Forced STOP triggered! Model didn't stop in goal zone (grace={self.config.goal_grace_steps}, patience={self.config.goal_stop_patience})")
                        
                        # 🔥 Collect oracle demonstration from rollback point
                        if not is_done and not observations.get('done', False) and not env.episode_over:
                            if last_waypoint_state is None:
                                # No waypoint state recorded yet - use episode start
                                print(f"    No waypoint state, using episode start for rollback")
                                rollback_state = {
                                    'position': list(episode.start_position),
                                    'rotation': episode.start_rotation,
                                    'step_id': 0,
                                    'nearest_idx': 0,
                                }
                            else:
                                rollback_state = last_waypoint_state
                                print(f"    Rolling back to waypoint {rollback_state['nearest_idx']} (step {rollback_state['step_id']})")
                            
                            # Collect oracle demonstration from rollback point
                            oracle_demo = self._collect_oracle_demonstration(
                                env=env,
                                rollback_state=rollback_state,
                                goal_position=goal_position,
                                reference_path=reference_path,
                            )
                            
                            if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                                oracle_demonstrations.append(oracle_demo)
                                print(f"    Collected oracle demo: {len(oracle_demo['rgbs'])} steps from rollback point to goal")
                                recovery_triggered_count += 1
                        else:
                            print(f"    Episode already done (step_id={step_id}, env.episode_over={env.episode_over}), skipping oracle demo collection")
                        
                        # Episode ends after forced STOP
                        print(f"    Episode ends after forced STOP detection\n")
                        is_done = True
                        break
                
                # 🔥 NEW: Detect premature STOP (wrong stop outside goal zone)
                if (self.config.enable_recovery and 
                    action == 0 and 
                    action_source == 'policy' and
                    not in_recovery and
                    curr_distance > self.config.goal_radius):
                    
                    # Model stopped outside goal zone - trigger rollback learning
                    print(f"\n  [Step {step_id}] Premature STOP detected! Distance to goal: {curr_distance:.2f}m (threshold: {self.config.goal_radius:.2f}m)")
                    
                    # 🔥 Check if episode is already done (e.g., max steps reached or env.episode_over)
                    if is_done or observations.get('done', False) or env.episode_over:
                        print(f"    Episode already done (step_id={step_id}, env.episode_over={env.episode_over}), skipping oracle demo collection")
                    else:
                        if last_waypoint_state is None:
                            # No waypoint state recorded yet - use episode start
                            print(f"    No waypoint state, using episode start for rollback")
                            rollback_state = {
                                'position': list(episode.start_position),
                                'rotation': episode.start_rotation,
                                'step_id': 0,
                                'nearest_idx': 0,
                            }
                        else:
                            rollback_state = last_waypoint_state
                            print(f"    Rolling back to waypoint {rollback_state['nearest_idx']} (step {rollback_state['step_id']})")
                        
                        # Collect oracle demonstration from rollback point
                        oracle_demo = self._collect_oracle_demonstration(
                            env=env,
                            rollback_state=rollback_state,
                            goal_position=goal_position,
                            reference_path=reference_path,
                        )
                        
                        if oracle_demo is not None and len(oracle_demo['rgbs']) > 0:
                            oracle_demonstrations.append(oracle_demo)
                            print(f"    Collected oracle demo: {len(oracle_demo['rgbs'])} steps from rollback point to goal")
                            recovery_triggered_count += 1
                    
                    # Episode ends after premature STOP detection
                    print(f"    Episode ends after premature STOP detection\n")
                    is_done = True
                    break
                
                # === 保存状态用于后续训练 ===
                # 关键：保存generated_ids_so_far作为autoregressive上下文
                # 训练时会将这些token作为prefix，确保训练/推理一致
                trajectory['states'].append({
                    'rgb': observations['rgb'].copy(),           # 当前RGB观测
                    'step_id': step_id,                         # 时间步ID
                    'instruction': instruction,                  # 导航指令
                    'generated_ids_so_far': generated_action_ids.copy() if not in_recovery else [],  # 🔥 Recovery时清空
                    'action_source': action_source,             # 🔥 Record action source
                    'agent_position': np.array(env.sim.get_agent_state().position),  # 🔥 Agent position for top-down map (ensure 3D array)
                })
                trajectory['actions'].append(action)
                trajectory['action_token_ids'].append(action_token_id)
                trajectory['gt_actions'].append(gt_action)  # 🔥 TDR: Save ground truth action
                trajectory['action_sources'].append(action_source)  # 🔥 Save action source
                
                # 🔥 Add current action to generation history (only for policy-generated actions)
                if action_source == 'policy':
                    generated_action_ids.append(action_token_id)
                trajectory['distances'].append(curr_distance)
                
                # Placeholder values - will be computed in PPO update
                trajectory['log_probs'].append(0.0)
                trajectory['old_log_probs'].append(0.0)
                trajectory['values'].append(0.0)
                
                # Step environment
                observations = env.step(action)
                step_id += 1
                
                # Get new metrics
                new_metrics = env.get_metrics()
                new_distance = new_metrics.get('distance_to_goal', float('inf'))
                
                # Check if done
                if action == 0:  # STOP
                    is_done = True
                else:
                    is_done = env.episode_over
                
                # Compute reward
                reward, reward_dict = self.reward_fn.compute_step_reward(
                    prev_distance_to_goal=curr_distance,
                    curr_distance_to_goal=new_distance,
                    action=action,
                    info=new_metrics,
                    is_done=is_done,
                    reference_path=reference_path,
                    agent_position=np.array(observations['gps']),
                )
                
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(is_done)
                trajectory['infos'].append(new_metrics)
                
                prev_distance = new_distance
                
                # Reset generation state at num_frames boundary (matching eval)
                if step_id % self.config.num_frames == 0:
                    self._base_model.reset_for_env(0)
                    output_ids = None
                    past_key_values = None
                    time_ids = []
                    action_seq = []  # Force regeneration
                    generated_action_ids = []  # 🔥 Reset autoregressive context
        
        # Store final metrics
        trajectory['final_metrics'] = env.get_metrics()
        
        # 🔥 Expert Intervention: Store statistics
        trajectory['recovery_stats'] = {
            'recovery_triggered': recovery_triggered_count,
            'recovery_success': recovery_success_count,
            'oracle_steps': sum(1 for src in trajectory['action_sources'] if src == 'oracle'),
            'policy_steps': sum(1 for src in trajectory['action_sources'] if src == 'policy'),
        }
        
        # 🔥 NEW: Store oracle demonstrations for rollback learning
        trajectory['oracle_demonstrations'] = oracle_demonstrations
        
        # === 恢复eval模式（清理状态） ===
        # 确保在训练更新时模型处于正确状态
        self.model.eval()
        
        return trajectory
    
    def _parse_actions(self, output: str) -> List[int]:
        """Parse action symbols from model output (same as PPO)."""
        import re
        import itertools
        
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
    
    def _nearest_path_index(self, agent_pos: np.ndarray, reference_path: np.ndarray) -> int:
        """Find the nearest point index on reference path to agent position (2D Euclidean)."""
        if len(reference_path) == 0:
            return 0
        # Compute 2D distances (x, z plane, ignore y)
        distances = np.linalg.norm(reference_path[:, [0, 2]] - agent_pos[[0, 2]], axis=1)
        return int(np.argmin(distances))
    
    def _dist_to_path_segment(self, agent_pos: np.ndarray, path_point_a: np.ndarray, 
                             path_point_b: np.ndarray) -> float:
        """Calculate the perpendicular distance from agent to a path segment.
        
        This solves the sparse waypoint problem: when reference path points are far apart
        (e.g., 5-10m in long corridors), an agent moving between two waypoints could be
        far from both points but still on the correct path.
        
        Args:
            agent_pos: Agent position (3D, but only x,z used)
            path_point_a: Start of segment (3D)
            path_point_b: End of segment (3D)
        
        Returns:
            Perpendicular distance to the line segment (2D in x-z plane)
        """
        # Work in 2D (x, z plane)
        agent_2d = agent_pos[[0, 2]]
        a_2d = path_point_a[[0, 2]]
        b_2d = path_point_b[[0, 2]]
        
        # Vector from A to B
        v = b_2d - a_2d
        # Vector from A to agent
        w = agent_2d - a_2d
        
        # Dot products for projection
        c1 = np.dot(w, v)
        
        # Agent is before point A (use distance to A)
        if c1 <= 0:
            return float(np.linalg.norm(agent_2d - a_2d))
        
        c2 = np.dot(v, v)
        
        # Agent is after point B (use distance to B)
        if c1 >= c2:
            return float(np.linalg.norm(agent_2d - b_2d))
        
        # Agent is between A and B, compute perpendicular distance
        b = c1 / c2
        projection = a_2d + b * v
        return float(np.linalg.norm(agent_2d - projection))
    
    def _dist_to_reference_path(self, agent_pos: np.ndarray, reference_path: np.ndarray, 
                                nearest_idx: int) -> float:
        """Calculate minimum distance from agent to reference path (considering segments).
        
        This method checks the distance to the line segment between the nearest point
        and its neighbors, handling sparse waypoints correctly.
        
        Args:
            agent_pos: Agent position (3D)
            reference_path: Array of waypoints (Nx3)
            nearest_idx: Index of nearest waypoint
        
        Returns:
            Minimum distance to the path (considering line segments)
        """
        if len(reference_path) <= 1:
            # Only one point, use point distance
            return float(np.linalg.norm(agent_pos[[0, 2]] - reference_path[0][[0, 2]]))
        
        # Check distance to segments around the nearest point
        min_dist = float('inf')
        
        # Segment before nearest point (if exists)
        if nearest_idx > 0:
            dist = self._dist_to_path_segment(
                agent_pos, 
                reference_path[nearest_idx - 1], 
                reference_path[nearest_idx]
            )
            min_dist = min(min_dist, dist)
        
        # Segment after nearest point (if exists)
        if nearest_idx < len(reference_path) - 1:
            dist = self._dist_to_path_segment(
                agent_pos,
                reference_path[nearest_idx],
                reference_path[nearest_idx + 1]
            )
            min_dist = min(min_dist, dist)
        
        # If nearest_idx is first or last point, also compute direct distance
        if nearest_idx == 0 or nearest_idx == len(reference_path) - 1:
            point_dist = float(np.linalg.norm(
                agent_pos[[0, 2]] - reference_path[nearest_idx][[0, 2]]
            ))
            min_dist = min(min_dist, point_dist)
        
        return min_dist
    
    def _select_target_waypoint(self, reference_path: np.ndarray, nearest_idx: int, 
                                last_progress_idx: int, goal_position: np.ndarray) -> np.ndarray:
        """Select target waypoint with lookahead from reference path."""
        # Ensure progress doesn't go backwards
        current_idx = max(nearest_idx, last_progress_idx)
        # Apply lookahead
        target_idx = min(current_idx + self.config.lookahead_k, len(reference_path) - 1)
        
        # If near the end of path, target the goal directly
        if target_idx >= len(reference_path) - 1:
            return goal_position
        
        return reference_path[target_idx]
    
    def _heading_error_deg(self, agent_rotation: np.quaternion, agent_pos: np.ndarray, 
                          target_pos: np.ndarray) -> float:
        """Calculate heading error in degrees between agent forward and target direction."""
        # Convert inputs to numpy arrays (handles both list and array inputs)
        agent_pos = np.asarray(agent_pos)
        target_pos = np.asarray(target_pos)
        
        # Get agent's forward direction from quaternion
        forward_vector = quaternion.as_rotation_matrix(agent_rotation) @ np.array([0, 0, -1])
        forward_2d = forward_vector[[0, 2]]  # x, z plane
        forward_2d = forward_2d / (np.linalg.norm(forward_2d) + 1e-8)
        
        # Target direction (2D)
        target_dir = target_pos[[0, 2]] - agent_pos[[0, 2]]
        target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        
        # Compute angle
        cos_angle = np.clip(np.dot(forward_2d, target_dir), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.rad2deg(angle_rad)
        
        return float(angle_deg)
    
    def _collect_oracle_demonstration(
        self,
        env: habitat.Env,
        rollback_state: Dict,
        goal_position: np.ndarray,
        reference_path: np.ndarray,
    ) -> Optional[Dict]:
        """
        Collect oracle demonstration from rollback point to goal.
        
        🔥 NEW: Rollback learning mechanism
        When policy goes off-track, we:
        1. Restore environment to last waypoint state
        2. Let oracle navigate from that point to goal
        3. Collect RGB and action sequence
        4. Train policy to mimic this demonstration with exponential decay weights
        
        Args:
            env: Habitat environment
            rollback_state: Saved state dict with position, rotation, step_id
            goal_position: Goal position
            reference_path: Reference path waypoints
        
        Returns:
            Dict with oracle demonstration data, or None if collection failed
        """
        print(f"  [Oracle Demo] Restoring environment to rollback state...")
        
        # 🔥 Check if environment is in a valid state before attempting rollback
        if env.episode_over:
            print(f"    Environment episode is over, cannot collect oracle demonstration")
            return None
        
        # Restore environment to rollback state
        try:
            # Convert position to numpy array (Habitat expects np.ndarray)
            position = rollback_state['position']
            if isinstance(position, list):
                position = np.array(position, dtype=np.float32)
            elif not isinstance(position, np.ndarray):
                position = np.array(position, dtype=np.float32)
            
            # Convert rotation to proper format
            rotation = rollback_state['rotation']
            if hasattr(rotation, 'components'):
                # Already a quaternion, convert to list
                rotation = [rotation.x, rotation.y, rotation.z, rotation.w]
            elif isinstance(rotation, np.ndarray):
                rotation = rotation.tolist()
            
            success = env.sim.set_agent_state(
                position=position,
                rotation=rotation,
                reset_sensors=True
            )
            if not success:
                print(f"    Failed to restore agent state")
                return None
        except Exception as e:
            print(f"    Exception during state restore: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Get fresh observations after state restore
        # Habitat's get_sensor_observations returns raw sensor data
        # We need to process it through the sensor suite
        raw_observations = env.sim.get_sensor_observations()
        observations = env.sim._sensor_suite.get_observations(raw_observations)
        
        # Initialize oracle follower
        follower = self._init_follower(env)
        
        # Collect demonstration
        demo_rgbs = []
        demo_actions = []
        demo_step_ids = []
        demo_positions = []  # 🔥 Store agent positions for top-down map
        
        max_oracle_steps = self.config.max_steps_per_episode - rollback_state['step_id']
        oracle_step = 0
        oracle_done = False
        
        print(f"  [Oracle Demo] Starting demonstration from waypoint {rollback_state['nearest_idx']}...")
        
        while not oracle_done and oracle_step < max_oracle_steps:
            # Get oracle action
            oracle_action = follower.get_next_action(goal_position)
            
            # Check if STOP
            if oracle_action == 0:  # STOP
                oracle_done = True
            
            # Record demonstration step
            demo_rgbs.append(observations['rgb'].copy())
            demo_actions.append(oracle_action)
            demo_step_ids.append(rollback_state['step_id'] + oracle_step)
            # 🔥 Get position from agent state (ensure 3D numpy array)
            agent_state = env.sim.get_agent_state()
            demo_positions.append(np.array(agent_state.position))
            
            # Execute action
            observations = env.step(oracle_action)
            oracle_step += 1
            
            # Check done
            if observations.get('done', False):
                oracle_done = True
        
        # Check if demonstration is valid
        if len(demo_rgbs) == 0:
            print(f"    Oracle demo empty")
            return None
        
        print(f"  [Oracle Demo] Collected {len(demo_rgbs)} steps")
        
        # Compute exponential decay weights for demonstration
        # Closer to rollback point = higher weight
        # Further away = weight decays towards 1.0
        demo_weights = self._compute_demonstration_weights(len(demo_rgbs))
        
        return {
            'rgbs': demo_rgbs,
            'actions': demo_actions,
            'step_ids': demo_step_ids,
            'positions': demo_positions,  # 🔥 Include positions
            'weights': demo_weights,
            'rollback_step_id': rollback_state['step_id'],
            'rollback_waypoint_idx': rollback_state['nearest_idx'],
        }
    
    def _compute_demonstration_weights(self, num_steps: int) -> List[float]:
        """
        Compute exponential decay weights for oracle demonstration.
        
        Weight formula: weight = base_weight * decay^step
        - Step 0 (rollback point): highest weight
        - Later steps: weight decays exponentially towards 1.0
        
        Args:
            num_steps: Number of demonstration steps
        
        Returns:
            List of weights for each step
        """
        base_weight = self.config.sft_loss_start_weight  # e.g., 2.0
        min_weight = 1.0  # Final weight
        
        # Decay rate: ensures last step reaches min_weight
        if num_steps <= 1:
            return [base_weight]
        
        # decay^(num_steps-1) = min_weight / base_weight
        decay_rate = (min_weight / base_weight) ** (1.0 / (num_steps - 1))
        
        weights = []
        for step in range(num_steps):
            weight = base_weight * (decay_rate ** step)
            weights.append(weight)
        
        return weights
    
    def _init_follower(self, env: habitat.Env) -> ShortestPathFollower:
        """Initialize ShortestPathFollower for oracle demonstration."""
        # ShortestPathFollower requires habitat simulator
        # 🎯 Use oracle_goal_radius (default 1.0m) - closer to target for better learning
        # This ensures oracle demos teach the model to get within 1m of the goal center,
        # while success judgment and goal zone protection still use goal_radius (3.0m)
        follower = ShortestPathFollower(
            env.sim,
            goal_radius=self.config.oracle_goal_radius,  # 🔥 Use configurable oracle goal radius
            return_one_hot=False  # Return action index directly
        )
        return follower
    
    def _preprocess_observation_rgb_only(self, rgb: np.ndarray) -> Dict:
        """Preprocess RGB observation."""
        rgb_pil = Image.fromarray(rgb)
        image_tensor = self.image_processor.preprocess(
            rgb_pil, return_tensors='pt'
        )['pixel_values'][0]
        return {'image': image_tensor}
    
    def _select_action_4way(
        self,
        instruction: str,
        rgb_list: List[torch.Tensor],
        time_ids: List[int],
        step_id: int,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """
        Select action from 4-way action space.
        Returns: action, action_token_id, log_prob, value
        """
        # Build prompt
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize
        input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        input_ids = input_ids.to(self.device)
        
        # Prepare image
        image_tensor = rgb_list[-1]
        
        # Build input dict
        input_dict = {
            'input_ids': input_ids,
            'images': image_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[step_id]],
            'task_type': [0],
            'output_hidden_states': True,
            'return_dict': True,
            'use_cache': False,
        }
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**input_dict)
        
        # Get logits
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]
        
        # Apply temperature
        action_logits = action_logits / temperature
        
        # Sample action
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        if deterministic:
            action = action_probs.argmax(dim=-1).item()
        else:
            action = torch.multinomial(action_probs, num_samples=1).item()
        
        log_prob = action_log_probs[0, action]
        action_token_id = self.action_token_ids[action].item()
        
        # Value head disabled for GRPO: return placeholder
        value = torch.tensor(0.0, device=self.device)
        
        return action, action_token_id, log_prob, value
    
    def _preprocess_qwen(self, sources: List[List[Dict]], add_system: bool = True):
        """
        Preprocess conversation for Qwen model.
        🔥 FIX: Match eval's preprocess_qwen exactly - add conjunction + <image> token
        """
        roles = {"human": "user", "gpt": "assistant"}
        system_message = "You are a helpful assistant."
        
        # Use a deepcopy of tokenizer (matching eval)
        tokenizer = copy.deepcopy(self.tokenizer)
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)
        
        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        
        # Reset Qwen chat templates (matching eval)
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template
        
        conversations = []
        input_ids = []
        
        for source in sources:
            # 🔥 FIX: Use fixed conjunction for deterministic behavior during RL
            # (matching RL trainer exactly)
            prompt = self.conjunction + DEFAULT_IMAGE_TOKEN  # Always use fixed conjunction
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else:
                source[0]["value"] = f"{prompt}."
            
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
            
            input_id = []
            
            # Add system message if needed
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
            
            # Replace special token indices (matching eval)
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
            
            input_ids.append(input_id)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids, conversations
    
    def compute_grpo_advantages_batch(self, all_trajectories: List[Dict]) -> List[Dict]:
        """
        Compute GRPO advantages for all trajectories.
        Group trajectories by instruction and compute group-relative advantages.
        """
        # Group by instruction
        instruction_to_trajs = {}
        for traj in all_trajectories:
            instruction = traj['instruction']
            if instruction not in instruction_to_trajs:
                instruction_to_trajs[instruction] = []
            instruction_to_trajs[instruction].append(traj)
        
        print(f"  Found {len(instruction_to_trajs)} unique instructions")
        
        # Process each group
        processed_trajs = []
        for instruction, group_trajs in instruction_to_trajs.items():
            group_with_adv = self._compute_grpo_advantages_single_group(group_trajs)
            processed_trajs.extend(group_with_adv)
        
        return processed_trajs
    
    def _compute_grpo_advantages_single_group(self, group_trajectories: List[Dict]) -> List[Dict]:
        """
        Compute GRPO advantages for a single group.
        
        Key idea: advantage_i = return_i - baseline(group)
        
        🔥 TDR Enhancement: Adds Time-Decayed Reward based on action matching with GT
        """
        # Compute returns for each trajectory (with optional TDR fusion)
        trajectory_returns = []
        for traj in group_trajectories:
            # Original environment reward
            env_return = sum(traj['rewards'])
            
            # 🔥 TDR: Compute time-decayed reward based on GT action matching
            # Only evaluate policy-generated actions (filter out oracle steps)
            if self.config.use_tdr and 'gt_actions' in traj and 'action_sources' in traj:
                # Extract policy-only subsequence
                policy_actions = [
                    a for a, s in zip(traj['actions'], traj['action_sources']) 
                    if s == 'policy'
                ]
                policy_gt_actions = [
                    g for g, s in zip(traj['gt_actions'], traj['action_sources']) 
                    if s == 'policy' and g >= 0
                ]
                
                # Only compute TDR if there are policy actions with valid GT
                if len(policy_actions) > 0 and len(policy_gt_actions) > 0:
                    tdr_score = self._compute_tdr_score(
                        predicted_actions=policy_actions,
                        gt_actions=policy_gt_actions,
                        gamma=self.config.tdr_gamma,
                        strict_mode=self.config.tdr_strict_mode,
                    )
                    # Reward fusion: weighted sum
                    total_return = env_return + self.config.tdr_weight * tdr_score
                else:
                    # No policy actions with valid GT (e.g., pure oracle trajectory)
                    total_return = env_return
            else:
                total_return = env_return
            
            trajectory_returns.append(total_return)
        
        # Compute group baseline
        if self.config.baseline_type == "mean":
            group_baseline = np.mean(trajectory_returns)
        elif self.config.baseline_type == "median":
            group_baseline = np.median(trajectory_returns)
        else:
            raise ValueError(f"Unknown baseline_type: {self.config.baseline_type}")
        
        # Update EMA baseline (for logging)
        if self.config.use_baseline_ema:
            if self.baseline_ema == 0.0:
                self.baseline_ema = group_baseline
            else:
                self.baseline_ema = (
                    self.config.baseline_ema_decay * self.baseline_ema +
                    (1 - self.config.baseline_ema_decay) * group_baseline
                )
        
        # 🔥 Compute advantages with normalization
        advantages_raw = []
        for i, traj in enumerate(group_trajectories):
            advantage = trajectory_returns[i] - group_baseline
            advantages_raw.append(advantage)
        
        # 🔥 Normalize advantages for stable gradients
        adv_mean = np.mean(advantages_raw)
        adv_std = np.std(advantages_raw)
        if adv_std < 1e-8:
            # Avoid division by zero when all advantages are equal
            advantages_normalized = [adv - adv_mean for adv in advantages_raw]
        else:
            advantages_normalized = [(adv - adv_mean) / adv_std for adv in advantages_raw]
        
        # Distribute advantages to steps
        for i, traj in enumerate(group_trajectories):
            advantage = advantages_normalized[i]
            num_steps = len(traj['rewards'])
            
            # Distribute advantage to steps
            if self.config.advantage_distribution == "uniform":
                # Uniform distribution
                step_advantages = [advantage / num_steps] * num_steps
            elif self.config.advantage_distribution == "temporal":
                # Temporal discount (early actions more important)
                gammas = [self.config.gamma ** j for j in range(num_steps)]
                gamma_sum = sum(gammas)
                step_advantages = [advantage * gamma / gamma_sum for gamma in gammas]
            else:
                raise ValueError(f"Unknown advantage_distribution: {self.config.advantage_distribution}")
            
            # Store
            traj['advantages'] = step_advantages
            traj['returns'] = [trajectory_returns[i]] * num_steps
            traj['group_baseline'] = group_baseline
        
        return group_trajectories
    
    def _grpo_update(self, trajectories: List[Dict]) -> Dict:
        """
        GRPO update: PPO without KL divergence.
        
        Key differences from PPO:
        1. No KL divergence calculation
        2. No KL early stopping
        3. Simplified loss function
        
        🔥 Expert Intervention:
        - Oracle steps: excluded from GRPO, included in SFT
        - Policy steps: included in GRPO, optionally in SFT
        
        🔥 NEW: Rollback demonstrations learning
        - Oracle demonstrations from rollback points included in SFT with exponential decay weights
        """
        # 🔥 NEW: Collect oracle demonstrations from all trajectories
        demo_experiences = []
        for traj in trajectories:
            if 'oracle_demonstrations' in traj:
                for demo in traj['oracle_demonstrations']:
                    instruction = traj['instruction']
                    for rgb, action, weight in zip(demo['rgbs'], demo['actions'], demo['weights']):
                        demo_experiences.append({
                            'rgb': rgb,
                            'action': action,
                            'weight': weight,
                            'instruction': instruction,
                        })
        
        if len(demo_experiences) > 0:
            print(f"  [Rollback Learning] Processing {len(demo_experiences)} oracle demonstration steps")
        
        # Prepare experiences (split by action source)
        all_experiences = []  # For GRPO (policy only)
        sft_experiences = []  # For SFT (oracle + optionally policy)
        
        for traj in trajectories:
            for i in range(len(traj['actions'])):
                action_source = traj['action_sources'][i] if i < len(traj['action_sources']) else 'policy'
                gt_action = traj['gt_actions'][i] if i < len(traj['gt_actions']) else -1
                
                exp = {
                    'state': traj['states'][i],
                    'action': traj['actions'][i],
                    'advantage': traj['advantages'][i],
                    'instruction': traj['instruction'],
                    'old_log_prob': traj['old_log_probs'][i],
                    'action_source': action_source,
                    'gt_action': gt_action,
                }
                
                # GRPO: only policy-generated actions
                if action_source == 'policy':
                    all_experiences.append(exp)
                
                # SFT: oracle actions (high priority) + optionally policy with valid gt
                if action_source == 'oracle':
                    # Always include oracle steps in SFT
                    sft_experiences.append(exp)
                elif self.config.sft_use_policy_steps and self.config.use_hybrid_training and gt_action >= 0:
                    # Include policy steps with valid gt if enabled
                    sft_experiences.append(exp)
        
        # 🔥 Handle cases with only oracle steps (no GRPO, but still do SFT)
        if len(all_experiences) == 0 and len(sft_experiences) == 0:
            return {
                'policy_loss': 0.0, 
                'entropy': 0.0, 
                'sft_loss': 0.0,
                'ratio_mean': 0.0,
                'ratio_std': 0.0,
                'sft_weight': 0.0,
                'sft_samples': 0,
            } 
        
        # 🔥 SFT-only training branch (when only oracle steps exist)
        if len(all_experiences) == 0 and len(sft_experiences) > 0:
            return self._sft_only_update(sft_experiences)
        
        # Convert to tensors (no returns needed for GRPO)
        advantages = torch.tensor(
            [exp['advantage'] for exp in all_experiences],
            dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.tensor(
            [exp['old_log_prob'] for exp in all_experiences],
            dtype=torch.float32, device=self.device
        )
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training stats
        stats = {
            'policy_loss': 0.0,
            'entropy': 0.0,
            'ratio_mean': 0.0,
            'ratio_std': 0.0,
            'demo_loss': 0.0,      # 🔥 NEW: Rollback demonstration loss
            'demo_weight': 0.0,    # 🔥 NEW: Average demonstration weight
            'demo_samples': 0,     # 🔥 NEW: Number of demonstration samples
        }
        num_updates = 0
        
        # 🔥 Set batch size for true batching (allow larger batches if memory permits)
        batch_size = min(12, self.config.mini_batch_size)  # Allow up to 8 for better efficiency
        
        # 🔥 Group experiences by instruction for efficient batching
        from collections import defaultdict
        instruction_groups = defaultdict(list)
        for idx, exp in enumerate(all_experiences):
            instruction_groups[exp['instruction']].append(idx)
        
        print(f"     Using batch_size = {batch_size} (🔥 true batch)")
        
        # 🔥 Initial GPU memory before training
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            print(f"     Initial GPU memory: {initial_memory:.2f}GB")
        
        # PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Create batches grouped by instruction
            all_batches = []
            for instruction, indices in instruction_groups.items():
                random.shuffle(indices)
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    all_batches.append((instruction, batch_indices))
            
            # Shuffle batches across instructions
            random.shuffle(all_batches)
            
            pbar = tqdm.tqdm(
                all_batches,
                desc=f"GRPO Epoch {epoch+1}/{self.config.ppo_epochs}",
                ncols=120,          # 固定宽度
                leave=True,         # 保留最后一次进度（不清除）
                dynamic_ncols=False, # 不动态调整宽度
                file=sys.stdout,    # 强制输出到 stdout
                position=0,         # 固定位置（避免多行）
                miniters=1,         # 每次更新最小迭代数
            )
            
            for instruction, batch_indices in pbar:
                # 🔥 Collect batch data
                batch_states = []
                batch_actions = []
                batch_generated_ids = []
                
                for idx in batch_indices:
                    exp = all_experiences[idx]
                    batch_states.append(exp['state'])
                    batch_actions.append(exp['action'])
                    batch_generated_ids.append(exp['state'].get('generated_ids_so_far', []))
                
                # 🔥 TRUE BATCH: Single forward pass for entire batch!
                batch_new_log_probs, batch_entropies = self._compute_action_log_prob_batch(
                    states=batch_states,
                    actions=batch_actions,
                    instruction=instruction,
                    generated_ids_list=batch_generated_ids,
                )
                
                # Convert batch_indices to tensor for indexing
                batch_indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
                
                # 🔧 Debug: Allow disabling GRPO loss
                if not self.config.disable_grpo_loss:
                    # PPO-Clip policy loss
                    ratio = torch.exp(batch_new_log_probs - old_log_probs[batch_indices_tensor])
                    surr1 = ratio * advantages[batch_indices_tensor]
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range
                    ) * advantages[batch_indices_tensor]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Entropy bonus
                    entropy_loss = batch_entropies.mean()
                    
                    # Stats
                    stats['policy_loss'] += policy_loss.item()
                    stats['entropy'] += entropy_loss.item()
                    stats['ratio_mean'] += ratio.mean().item()
                    if len(ratio) > 1:
                        stats['ratio_std'] += ratio.std().item()
                    else:
                        stats['ratio_std'] += 0.0
                else:
                    # GRPO loss disabled - use dummy values
                    policy_loss = torch.tensor(0.0, device=self.device)
                    entropy_loss = torch.tensor(0.0, device=self.device)
                    ratio = torch.ones(len(batch_indices), device=self.device)
                
                # 🔥 Hybrid Training + Expert Intervention: Compute SFT loss from separate data
                sft_loss = torch.tensor(0.0, device=self.device)
                sft_weight = 0.0
                
                if len(sft_experiences) > 0:
                    # Sample SFT batch (same size as GRPO batch for balance)
                    sft_batch_size = min(len(batch_indices) if len(all_experiences) > 0 else self.config.mini_batch_size, len(sft_experiences))
                    sft_batch_indices = random.sample(range(len(sft_experiences)), sft_batch_size)
                    
                    batch_sft_losses = []
                    batch_sft_weights = []
                    for sft_idx in sft_batch_indices:
                        sft_exp = sft_experiences[sft_idx]
                        generated_ids_so_far = sft_exp['state'].get('generated_ids_so_far', [])
                        gt_action = sft_exp['gt_action']
                        action_source = sft_exp['action_source']
                        
                        # 🔥 For oracle steps: use the action itself as GT (it's the expert action)
                        # For policy steps: use gt_action from TDR computation
                        if action_source == 'oracle':
                            effective_gt = sft_exp['action']  # Oracle action is the GT
                        elif gt_action >= 0:
                            effective_gt = gt_action  # Use TDR's GT action
                        else:
                            continue  # Skip if no valid GT
                        
                        sft_loss_val = self._compute_sft_loss(
                            state=sft_exp['state'],
                            gt_action=effective_gt,
                            instruction=sft_exp['instruction'],
                            generated_ids_so_far=generated_ids_so_far,
                        )
                        batch_sft_losses.append(sft_loss_val)
                        
                        # 🔥 Weight by action source: oracle 2x, policy 1x
                        base_weight = self._get_sft_weight() if self.config.use_hybrid_training else 1.0
                        if action_source == 'oracle':
                            batch_sft_weights.append(base_weight * 2.0)  # Oracle: 2x
                        else:
                            batch_sft_weights.append(base_weight)  # Policy: 1x
                    
                    if len(batch_sft_losses) > 0:
                        # Weighted average of SFT losses
                        sft_weights_tensor = torch.tensor(batch_sft_weights, device=self.device)
                        sft_losses_tensor = torch.stack(batch_sft_losses)
                        sft_loss = (sft_losses_tensor * sft_weights_tensor).sum() / sft_weights_tensor.sum()
                        sft_weight = sft_weights_tensor.mean().item()
                        
                        stats['sft_loss'] = stats.get('sft_loss', 0.0) + sft_loss.item()
                        stats['sft_weight'] = stats.get('sft_weight', 0.0) + sft_weight
                        stats['sft_samples'] = stats.get('sft_samples', 0) + len(batch_sft_losses)
                
                # 🔥 NEW: Rollback demonstrations learning
                demo_loss = torch.tensor(0.0, device=self.device)
                if len(demo_experiences) > 0:
                    # Sample demonstration batch (limit to 2 to save memory)
                    demo_batch_size = min(2, len(demo_experiences))  # 🔥 Reduced from 4 to 2 to prevent OOM
                    demo_batch_indices = random.sample(range(len(demo_experiences)), demo_batch_size)
                    
                    batch_demo_losses = []
                    batch_demo_weights = []
                    for demo_idx in demo_batch_indices:
                        demo_exp = demo_experiences[demo_idx]
                        
                        # Compute SFT loss for this demonstration step
                        demo_loss_val = self._compute_sft_loss_from_rgb(
                            rgb=demo_exp['rgb'],
                            gt_action=demo_exp['action'],
                            instruction=demo_exp['instruction'],
                        )
                        batch_demo_losses.append(demo_loss_val)
                        batch_demo_weights.append(demo_exp['weight'])  # Exponential decay weight
                    
                    if len(batch_demo_losses) > 0:
                        # Weighted average
                        demo_weights_tensor = torch.tensor(batch_demo_weights, device=self.device)
                        demo_losses_tensor = torch.stack(batch_demo_losses)
                        demo_loss = (demo_losses_tensor * demo_weights_tensor).sum() / demo_weights_tensor.sum()
                        
                        stats['demo_loss'] = stats.get('demo_loss', 0.0) + demo_loss.item()
                        stats['demo_weight'] = stats.get('demo_weight', 0.0) + demo_weights_tensor.mean().item()
                        stats['demo_samples'] = stats.get('demo_samples', 0) + len(batch_demo_losses)
                
                # Total loss
                if self.config.disable_grpo_loss:
                    # Only SFT loss (for debugging)
                    loss = sft_weight * sft_loss + demo_loss
                else:
                    # Normal: GRPO + optional SFT + demo learning
                    loss = (
                        policy_loss -
                        self.config.entropy_coef * entropy_loss +
                        sft_weight * sft_loss +
                        demo_loss  # Rollback demonstrations
                    )
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                num_updates += 1
                
                # Update progress bar
                progress_info = {
                    'ploss': f"{policy_loss.item():.3f}",
                    'ratio': f"{ratio.mean().item():.3f}",
                }
                if self.config.use_hybrid_training and sft_loss.item() > 0:
                    progress_info['sft'] = f"{sft_loss.item():.3f}"
                    progress_info['sft_w'] = f"{sft_weight:.2f}"
                if demo_loss.item() > 0:  # 🔥 NEW: Show demo loss if present
                    progress_info['demo'] = f"{demo_loss.item():.3f}"
                pbar.set_postfix(progress_info)
                
                # 🔥 Clear gradients and cached tensors after stats collection
                del loss
                if not self.config.disable_grpo_loss:
                    del policy_loss, entropy_loss, ratio, surr1, surr2
                del batch_new_log_probs, batch_entropies
                if self.config.use_hybrid_training:
                    del sft_loss
                if len(demo_experiences) > 0:  # 🔥 NEW: Clean up demo loss
                    del demo_loss
                # 🔥 Let PyTorch manage memory automatically - removed torch.cuda.empty_cache()
            
            # 🔥 GPU Memory after each epoch
            if torch.cuda.is_available():
                epoch_peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"     Epoch {epoch+1} peak memory: {epoch_peak_memory:.2f}GB")
        
        # Average stats
        for key in stats:
            stats[key] /= max(num_updates, 1)
        
        return stats
    
    def _compute_action_log_prob_and_value(
        self,
        state: Dict,
        action: int,
        instruction: str,
        generated_ids_so_far: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log_prob and entropy for a state-action pair (no value estimation).
        
        🔥 Uses autoregressive context (generated_ids_so_far) for sequence-based training.
        This ensures training-inference consistency when using generate() for rollout.
        
        Args:
            state: State dict with 'rgb', 'step_id'
            action: Action index (0-3)
            instruction: Navigation instruction
            generated_ids_so_far: Previously generated action tokens (autoregressive context)
        """
        rgb = state['rgb']
        step_id = state.get('step_id', 0)
        
        if generated_ids_so_far is None:
            generated_ids_so_far = []
        
        obs_dict = self._preprocess_observation_rgb_only(rgb)
        image_tensor = obs_dict['image']
        
        # Build prompt
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize
        input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        input_ids = input_ids.to(self.device)
        
        # 🔥 Append autoregressive context (previously generated actions)
        if len(generated_ids_so_far) > 0:
            generated_tokens = torch.tensor(
                generated_ids_so_far, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, generated_tokens], dim=1)
        
        # Build input dict
        input_dict = {
            'input_ids': input_ids,
            'images': image_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[step_id]],
            'task_type': [0],
            'output_hidden_states': False,  # No hidden states needed for GRPO
            'return_dict': True,
            'use_cache': False,
        }
        
        # Forward pass (with gradients)
        outputs = self.model(**input_dict)
        
        # Get logits
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]
        
        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Get log_prob for taken action
        log_prob = action_log_probs[0, action]
        
        # Compute entropy
        entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        
        # Value head removed for GRPO; no value computed
        # 🔥 Clear intermediate tensors to save memory
        del outputs, logits, input_dict, image_tensor
        
        return log_prob, entropy
    
    def _compute_action_log_prob_batch(
        self,
        states: List[Dict],
        actions: List[int],
        instruction: str,
        generated_ids_list: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        🔥 Batch version: Compute log_prob and entropy for multiple state-action pairs.
        
        All samples must share the same instruction. Different context lengths are
        handled by padding.
        
        Args:
            states: List of state dicts with 'rgb', 'step_id'
            actions: List of action indices (0-3)
            instruction: Navigation instruction (shared by all samples)
            generated_ids_list: List of autoregressive contexts (may have different lengths)
        
        Returns:
            log_probs: Tensor of shape [batch_size] with log probabilities
            entropies: Tensor of shape [batch_size] with entropies
        """
        batch_size = len(states)
        if batch_size == 0:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device)
        
        # Process all RGB images
        image_tensors = []
        for state in states:
            rgb = state['rgb']
            obs_dict = self._preprocess_observation_rgb_only(rgb)
            image_tensors.append(obs_dict['image'])
        
        # Stack images: [batch_size, C, H, W] -> [batch_size, 1, C, H, W]
        images_batch = torch.stack(image_tensors, dim=0).to(self.device, dtype=torch.bfloat16)
        images_batch = images_batch.unsqueeze(1)
        
        # Build prompt (same for all samples)
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize base prompt (same for all samples)
        base_input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        base_input_ids = base_input_ids.to(self.device)  # [1, seq_len]
        base_seq_len = base_input_ids.shape[1]
        
        # Find max context length for padding
        max_context_len = max(len(ctx) if ctx else 0 for ctx in generated_ids_list)
        
        # Build padded input_ids for each sample
        input_ids_list = []
        attention_masks = []
        
        for i, generated_ids in enumerate(generated_ids_list):
            if generated_ids is None:
                generated_ids = []
            
            # Pad context to max length
            context_len = len(generated_ids)
            padding_len = max_context_len - context_len
            
            if context_len > 0:
                context_tensor = torch.tensor(generated_ids, dtype=torch.long, device=self.device)
            else:
                context_tensor = torch.tensor([], dtype=torch.long, device=self.device)
            
            # Pad on the left (before context)
            if padding_len > 0:
                padding = torch.full((padding_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long, device=self.device)
                context_tensor = torch.cat([padding, context_tensor])
            
            # Combine: [base_prompt, padded_context]
            full_input_ids = torch.cat([base_input_ids[0], context_tensor])
            input_ids_list.append(full_input_ids)
            
            # Attention mask: 0 for padding, 1 for real tokens
            attn_mask = torch.ones(base_seq_len + max_context_len, dtype=torch.long, device=self.device)
            if padding_len > 0:
                attn_mask[base_seq_len:base_seq_len + padding_len] = 0
            attention_masks.append(attn_mask)
        
        # Stack into batch
        input_ids_batch = torch.stack(input_ids_list, dim=0)  # [batch_size, seq_len]
        attention_mask_batch = torch.stack(attention_masks, dim=0)  # [batch_size, seq_len]
        
        # Get step_ids
        step_ids = [state.get('step_id', 0) for state in states]
        
        # Build input dict for batch
        input_dict = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'images': images_batch,
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[sid] for sid in step_ids],
            'task_type': [0] * batch_size,
            'output_hidden_states': False,
            'return_dict': True,
            'use_cache': False,
        }
        
        # Forward pass (with gradients) - single forward for entire batch!
        outputs = self.model(**input_dict)
        
        # Get logits for last position: [batch_size, vocab_size]
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]  # [batch_size, 4]
        
        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)  # [batch_size, 4]
        action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, 4]
        
        # Get log_prob for taken actions
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs = action_log_probs[torch.arange(batch_size, device=self.device), actions_tensor]
        
        # Compute entropy for each sample
        entropies = -(action_probs * action_log_probs).sum(dim=-1)
        
        # Memory cleanup
        del outputs, logits, vocab_mask, action_logits, action_probs, action_log_probs
        del input_dict, images_batch, input_ids_batch, attention_mask_batch, image_tensors
        
        return log_probs, entropies
    
    def _compute_action_log_prob_and_sft_loss(
        self,
        state: Dict,
        action: int,
        gt_action: int,
        instruction: str,
        generated_ids_so_far: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        🔥 Memory-optimized: Compute both GRPO log_prob and SFT loss in single forward pass.
        
        This avoids redundant forward passes in hybrid training mode.
        
        Returns:
            log_prob: Log probability of taken action (for GRPO)
            entropy: Action distribution entropy
            sft_loss: Cross-entropy loss for gt_action (for SFT)
        """
        rgb = state['rgb']
        step_id = state.get('step_id', 0)
        
        if generated_ids_so_far is None:
            generated_ids_so_far = []
        
        obs_dict = self._preprocess_observation_rgb_only(rgb)
        image_tensor = obs_dict['image']
        
        # Build prompt
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize
        input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        input_ids = input_ids.to(self.device)
        
        # Append autoregressive context
        if len(generated_ids_so_far) > 0:
            generated_tokens = torch.tensor(
                generated_ids_so_far, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, generated_tokens], dim=1)
        
        # Build input dict
        input_dict = {
            'input_ids': input_ids,
            'images': image_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[step_id]],
            'task_type': [0],
            'output_hidden_states': False,
            'return_dict': True,
            'use_cache': False,
        }
        
        # 🔥 Single forward pass
        outputs = self.model(**input_dict)
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]
        
        # Compute probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        
        # GRPO: Get log_prob for taken action
        log_prob = action_log_probs[0, action]
        
        # GRPO: Compute entropy
        entropy = -(action_probs * action_log_probs).sum(dim=-1).mean()
        
        # SFT: Compute cross-entropy loss for gt_action
        target = torch.tensor([gt_action], dtype=torch.long, device=self.device)
        sft_loss = F.cross_entropy(action_logits, target)
        
        # Clear intermediate tensors
        del outputs, logits, input_dict, image_tensor
        
        return log_prob, entropy, sft_loss
    
    def _compute_sft_loss(
        self,
        state: Dict,
        gt_action: int,
        instruction: str,
        generated_ids_so_far: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Compute supervised learning loss for a state-gt_action pair.
        
        🔥 Hybrid Training: Use ground truth actions as supervision signal.
        This provides a stable learning signal complementing GRPO's reward-based learning.
        
        Args:
            state: State dict with 'rgb', 'step_id'
            gt_action: Ground truth optimal action index (0-3)
            instruction: Navigation instruction
            generated_ids_so_far: Previously generated action tokens (autoregressive context)
        
        Returns:
            Cross-entropy loss for the ground truth action
        """
        rgb = state['rgb']
        step_id = state.get('step_id', 0)
        
        if generated_ids_so_far is None:
            generated_ids_so_far = []
        
        obs_dict = self._preprocess_observation_rgb_only(rgb)
        image_tensor = obs_dict['image']
        
        # Build prompt (same as log_prob computation)
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize
        input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        input_ids = input_ids.to(self.device)
        
        # 🔥 Append autoregressive context
        if len(generated_ids_so_far) > 0:
            generated_tokens = torch.tensor(
                generated_ids_so_far, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            input_ids = torch.cat([input_ids, generated_tokens], dim=1)
        
        # Build input dict
        input_dict = {
            'input_ids': input_ids,
            'images': image_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[step_id]],
            'task_type': [0],
            'output_hidden_states': False,
            'return_dict': True,
            'use_cache': False,
        }
        
        # Forward pass (with gradients)
        outputs = self.model(**input_dict)
        
        # Get logits for last position
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]
        
        # Compute cross-entropy loss
        target = torch.tensor([gt_action], dtype=torch.long, device=self.device)
        sft_loss = F.cross_entropy(action_logits, target)
        
        # 🔥 Clear intermediate tensors
        del outputs, logits, input_dict, image_tensor
        
        return sft_loss
    
    def _compute_sft_loss_from_rgb(
        self,
        rgb: np.ndarray,
        gt_action: int,
        instruction: str,
    ) -> torch.Tensor:
        """
        Compute SFT loss directly from RGB observation (simplified for demonstrations).
        
        🔥 NEW: Used for rollback demonstrations learning
        No autoregressive context needed as each demonstration step is independent.
        
        Args:
            rgb: RGB observation (numpy array)
            gt_action: Ground truth action index (0-3)
            instruction: Navigation instruction
        
        Returns:
            Cross-entropy loss for the ground truth action
        """
        obs_dict = self._preprocess_observation_rgb_only(rgb)
        image_tensor = obs_dict['image']
        
        # Build prompt
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>", instruction)
        
        # Tokenize
        input_ids, _ = self._preprocess_qwen([sources], add_system=True)
        input_ids = input_ids.to(self.device)
        
        # Build input dict (no autoregressive context for demonstrations)
        input_dict = {
            'input_ids': input_ids,
            'images': image_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.bfloat16),
            'depths': None,
            'poses': None,
            'intrinsics': None,
            'time_ids': [[0]],  # Dummy step_id
            'task_type': [0],
            'output_hidden_states': False,
            'return_dict': True,
            'use_cache': False,
        }
        
        # Forward pass (with gradients)
        outputs = self.model(**input_dict)
        
        # Get logits for last position
        logits = outputs.logits[:, -1, :].float()
        
        # Mask to 4 actions
        vocab_mask = torch.full_like(logits, float('-inf'))
        vocab_mask[:, self.action_token_ids] = logits[:, self.action_token_ids]
        action_logits = vocab_mask[:, self.action_token_ids]
        
        # Compute cross-entropy loss
        target = torch.tensor([gt_action], dtype=torch.long, device=self.device)
        sft_loss = F.cross_entropy(action_logits, target)
        
        # 🔥 Aggressive memory cleanup for demo learning
        del outputs, logits, vocab_mask, action_logits, target
        del input_dict, image_tensor, input_ids
        torch.cuda.empty_cache()
        
        return sft_loss
    
    def _get_sft_weight(self) -> float:
        """
        Get dynamic SFT loss weight based on training progress.
        
        Schedule types:
        - 'linear': Linear decay from start_weight to end_weight
        - 'cosine': Cosine annealing decay (smooth transition)
        - 'exponential': Exponential decay
        
        Returns:
            Current SFT loss weight (float)
        """
        if not self.config.use_hybrid_training:
            return 0.0
        
        # Calculate progress (0.0 to 1.0)
        progress = min(1.0, self.current_update / max(1, self.config.sft_loss_decay_updates))
        
        start_weight = self.config.sft_loss_start_weight
        end_weight = self.config.sft_loss_end_weight
        
        if self.config.sft_loss_decay_type == 'linear':
            # Linear interpolation
            weight = start_weight + (end_weight - start_weight) * progress
        elif self.config.sft_loss_decay_type == 'cosine':
            # Cosine annealing: smooth transition
            weight = end_weight + 0.5 * (start_weight - end_weight) * (1 + np.cos(np.pi * progress))
        elif self.config.sft_loss_decay_type == 'exponential':
            # Exponential decay
            decay_rate = np.log(end_weight / start_weight)
            weight = start_weight * np.exp(decay_rate * progress)
        else:
            # Default to linear
            weight = start_weight + (end_weight - start_weight) * progress
        
        return weight
    
    def _sft_only_update(self, sft_experiences: List[Dict]) -> Dict:
        """
        Training update with only SFT loss (no GRPO).
        Used when all steps are oracle-generated (pure expert demonstration).
        """
        stats = {
            'policy_loss': 0.0,
            'entropy': 0.0,
            'ratio_mean': 0.0,
            'ratio_std': 0.0,
            'sft_loss': 0.0,
            'sft_weight': 0.0,
            'sft_samples': 0,
        }
        num_updates = 0
        
        print(f"  Running SFT-only update ({len(sft_experiences)} oracle samples)...")
        
        for epoch in range(self.config.ppo_epochs):
            indices = list(range(len(sft_experiences)))
            random.shuffle(indices)
            
            for start_idx in range(0, len(indices), self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_sft_losses = []
                batch_sft_weights = []
                
                for idx in batch_indices:
                    exp = sft_experiences[idx]
                    generated_ids_so_far = exp['state'].get('generated_ids_so_far', [])
                    action_source = exp['action_source']
                    
                    # For oracle steps: use the action itself as GT
                    if action_source == 'oracle':
                        effective_gt = exp['action']
                    else:
                        gt_action = exp.get('gt_action', -1)
                        if gt_action < 0:
                            continue
                        effective_gt = gt_action
                    
                    sft_loss_val = self._compute_sft_loss(
                        state=exp['state'],
                        gt_action=effective_gt,
                        instruction=exp['instruction'],
                        generated_ids_so_far=generated_ids_so_far,
                    )
                    batch_sft_losses.append(sft_loss_val)
                    
                    # Weight: oracle 2x, policy 1x
                    base_weight = self._get_sft_weight() if self.config.use_hybrid_training else 1.0
                    if action_source == 'oracle':
                        batch_sft_weights.append(base_weight * 2.0)
                    else:
                        batch_sft_weights.append(base_weight)
                
                if len(batch_sft_losses) == 0:
                    continue
                
                # Compute weighted SFT loss
                sft_weights_tensor = torch.tensor(batch_sft_weights, device=self.device)
                sft_losses_tensor = torch.stack(batch_sft_losses)
                sft_loss = (sft_losses_tensor * sft_weights_tensor).sum() / sft_weights_tensor.sum()
                sft_weight = sft_weights_tensor.mean().item()
                
                # Backward
                self.optimizer.zero_grad()
                sft_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Stats
                stats['sft_loss'] += sft_loss.item()
                stats['sft_weight'] += sft_weight
                stats['sft_samples'] += len(batch_sft_losses)
                num_updates += 1
                
                # Clear memory
                del sft_loss, sft_weights_tensor, sft_losses_tensor
                torch.cuda.empty_cache()
        
        # Average stats
        if num_updates > 0:
            stats['sft_loss'] /= num_updates
            stats['sft_weight'] /= num_updates
        
        return stats
    
    def _compute_tdr_score(
        self,
        predicted_actions: List[int],
        gt_actions: List[int],
        gamma: float = 0.9,
        strict_mode: bool = True,
    ) -> float:
        """
        Compute Time-Decayed Reward (TDR) score.
        
        Formula: score = sum_{t=0}^{T} gamma^t * I(action_t == gt_action_t)
        
        Args:
            predicted_actions: Model's predicted action sequence
            gt_actions: Ground truth optimal action sequence
            gamma: Decay factor (0.9 by default)
            strict_mode: If True, stop accumulating after first mismatch
        
        Returns:
            TDR score (non-negative float)
        """
        if len(predicted_actions) == 0 or len(gt_actions) == 0:
            return 0.0
        
        score = 0.0
        min_len = min(len(predicted_actions), len(gt_actions))
        
        for t in range(min_len):
            # Check if action matches GT
            if predicted_actions[t] == gt_actions[t]:
                # Add decayed reward
                score += (gamma ** t)
            elif strict_mode:
                # Strict mode: stop on first wrong action
                break
            # Else: continue accumulating (non-strict mode)
        
        return score
    
    def _log_update(self, update_id: int, stats: Dict, update_time: float):
        """Log update statistics."""
        print(f"\n📈 Update {update_id+1} Statistics:")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Ratio Mean: {stats['ratio_mean']:.4f} ± {stats['ratio_std']:.4f}")
        
        # 🔥 Hybrid Training + Expert Intervention: Log SFT stats
        if 'sft_loss' in stats:
            print(f"  SFT Loss: {stats['sft_loss']:.4f}")
            print(f"  SFT Weight: {stats['sft_weight']:.3f}")
            if 'sft_samples' in stats:
                print(f"  SFT Samples: {stats['sft_samples']:.0f}")
        
        # 🔥 Expert Intervention: Log recovery stats
        if 'oracle_steps' in stats:
            print(f"  Oracle Steps: {stats['oracle_steps']:.0f}")
            print(f"  Policy Steps: {stats['policy_steps']:.0f}")
            oracle_ratio = stats['oracle_steps'] / max(stats['oracle_steps'] + stats['policy_steps'], 1)
            print(f"  Oracle Ratio: {oracle_ratio:.2%}")
        if 'recovery_triggered' in stats:
            print(f"  Recovery Triggered: {stats['recovery_triggered']:.0f}")
            if stats['recovery_triggered'] > 0:
                success_rate = stats.get('recovery_success', 0) / stats['recovery_triggered']
                print(f"  Recovery Success Rate: {success_rate:.2%}")
        
        print(f"  Baseline EMA: {self.baseline_ema:.4f}")
        print(f"  Update Time: {update_time:.1f}s")
        
        # WandB logging
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                'update': update_id + 1,
                'policy_loss': stats['policy_loss'],
                'entropy': stats['entropy'],
                'ratio_mean': stats['ratio_mean'],
                'ratio_std': stats['ratio_std'],
                'baseline_ema': self.baseline_ema,
                'update_time': update_time,
            }
            
            # Add hybrid training metrics
            if 'sft_loss' in stats:
                log_dict['sft_loss'] = stats['sft_loss']
                log_dict['sft_weight'] = stats['sft_weight']
                if 'sft_samples' in stats:
                    log_dict['sft_samples'] = stats['sft_samples']
            
            # Add expert intervention metrics
            if 'oracle_steps' in stats:
                log_dict['oracle_steps'] = stats['oracle_steps']
                log_dict['policy_steps'] = stats['policy_steps']
                total_steps = stats['oracle_steps'] + stats['policy_steps']
                if total_steps > 0:
                    log_dict['oracle_ratio'] = stats['oracle_steps'] / total_steps
            if 'recovery_triggered' in stats:
                log_dict['recovery_triggered'] = stats['recovery_triggered']
                log_dict['recovery_success'] = stats['recovery_success']
                if stats['recovery_triggered'] > 0:
                    log_dict['recovery_success_rate'] = stats['recovery_success'] / stats['recovery_triggered']
            
            wandb.log(log_dict)
    
    def save_checkpoint(self, update_id):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_path, f"checkpoint_{update_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        

        
        # Save training state
        training_state = {
            'global_step': self.global_step,
            'episode_count': self.episode_count,
            'baseline_ema': self.baseline_ema,
            'current_update': self.current_update,  # 🔥 Save for hybrid training
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
    parser.add_argument("--output_path", type=str, default="./results/grpo_training")
    parser.add_argument("--phase", type=str, default="phase1_stop")
    parser.add_argument("--num_updates", type=int, default=500)
    parser.add_argument("--num_episodes_per_update", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="streamvln_grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # 🔥 Hybrid Training arguments
    parser.add_argument("--use_hybrid_training", action="store_true", 
                        help="Enable hybrid training (GRPO + SFT on gt_actions)")
    parser.add_argument("--sft_use_policy_steps", action="store_true", default=True,
                        help="Include policy steps (with valid gt) in SFT training")
    parser.add_argument("--sft_oracle_only", dest="sft_use_policy_steps", action="store_false",
                        help="Use only oracle steps for SFT (exclude policy steps)")
    parser.add_argument("--sft_loss_start_weight", type=float, default=1.0,
                        help="Initial SFT loss weight (high at beginning)")
    parser.add_argument("--sft_loss_end_weight", type=float, default=0.5,
                        help="Final SFT loss weight (stable, not lower than GRPO)")
    parser.add_argument("--sft_loss_decay_updates", type=int, default=200,
                        help="Number of updates for SFT weight decay")
    parser.add_argument("--sft_loss_decay_type", type=str, default="cosine",
                        choices=['linear', 'cosine', 'exponential'],
                        help="Decay schedule: linear, cosine, or exponential")
    
    # 🔥 Expert Intervention arguments
    parser.add_argument("--enable_recovery", action="store_true", default=True,
                        help="Enable expert intervention (recovery mode)")
    parser.add_argument("--disable_recovery", dest="enable_recovery", action="store_false",
                        help="Disable expert intervention (pure GRPO)")
    parser.add_argument("--offtrack_dist_thresh", type=float, default=3.0,
                        help="Distance threshold to trigger intervention (meters)")
    parser.add_argument("--offtrack_heading_thresh_deg", type=float, default=90.0,
                        help="Heading error threshold to trigger intervention (degrees)")
    parser.add_argument("--offtrack_patience", type=int, default=2,
                        help="Number of consecutive off-track steps before intervention")
    parser.add_argument("--lookahead_k", type=int, default=1,
                        help="Lookahead steps on reference path")
    parser.add_argument("--recovery_dist_thresh", type=float, default=2.0,
                        help="Distance threshold to exit recovery (meters)")
    parser.add_argument("--recovery_heading_thresh_deg", type=float, default=30.0,
                        help="Heading error threshold to exit recovery (degrees)")
    parser.add_argument("--recovery_max_steps", type=int, default=40,
                        help="Maximum recovery steps (safety stop)")
    parser.add_argument("--goal_radius", type=float, default=3.0,
                        help="Goal zone radius for recovery protection (meters)")
    parser.add_argument("--oracle_goal_radius", type=float, default=1.0,
                        help="Oracle demo goal radius - closer to target for better learning (meters)")
    parser.add_argument("--heading_guard_dist", type=float, default=1.0,
                        help="Minimum distance for heading check to apply (meters)")
    parser.add_argument("--goal_grace_steps", type=int, default=5,
                        help="Grace steps in goal zone before forced STOP")
    parser.add_argument("--goal_stop_patience", type=int, default=0,
                        help="Steps before forcing STOP in goal zone (0=disabled)")
    
    # � NEW: Stuck Detection
    parser.add_argument("--stuck_patience", type=int, default=60,
                        help="Number of steps stuck at same waypoint before intervention")
    parser.add_argument("--stuck_enable", action="store_true",
                        help="Enable stuck detection")
    
    # 🔥 TDR (Trajectory Distance Regularization) arguments
    parser.add_argument("--use_tdr", action="store_true", default=True,
                        help="Enable TDR mechanism (default: True)")
    parser.add_argument("--no_tdr", dest="use_tdr", action="store_false",
                        help="Disable TDR mechanism")
    parser.add_argument("--tdr_weight", type=float, default=0.2,
                        help="TDR weight coefficient for reward fusion (default: 0.2)")
    parser.add_argument("--tdr_gamma", type=float, default=0.9,
                        help="TDR decay factor (default: 0.9)")
    parser.add_argument("--tdr_strict_mode", action="store_true", default=True,
                        help="Stop TDR on first wrong action (default: True)")
    parser.add_argument("--tdr_no_strict_mode", dest="tdr_strict_mode", action="store_false",
                        help="Continue TDR accumulation after wrong actions")
    
    # Debug options
    parser.add_argument("--disable_grpo_loss", action="store_true",
                        help="Disable GRPO policy loss (for debugging OOM)")
    parser.add_argument("--sft_only", action="store_true",
                        help="SFT-only mode: only train on oracle data (skip GRPO)")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if args.use_wandb and WANDB_AVAILABLE:
        run_name = args.wandb_run_name or f"grpo_{args.phase}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Create config
    config = GRPOTrainingConfig(
        model_path=args.model_path,
        habitat_config_path=args.habitat_config_path,
        output_path=args.output_path,
        phase=args.phase,
        num_episodes_per_update=args.num_episodes_per_update,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        max_grad_norm=args.max_grad_norm,
        # 🔥 Hybrid Training settings
        use_hybrid_training=args.use_hybrid_training,
        sft_loss_start_weight=args.sft_loss_start_weight,
        sft_loss_end_weight=args.sft_loss_end_weight,
        sft_loss_decay_updates=args.sft_loss_decay_updates,
        sft_loss_decay_type=args.sft_loss_decay_type,
        # 🔥 Expert Intervention settings
        enable_recovery=args.enable_recovery,
        offtrack_dist_thresh=args.offtrack_dist_thresh,
        offtrack_heading_thresh_deg=args.offtrack_heading_thresh_deg,
        offtrack_patience=args.offtrack_patience,
        lookahead_k=args.lookahead_k,
        recovery_dist_thresh=args.recovery_dist_thresh,
        recovery_heading_thresh_deg=args.recovery_heading_thresh_deg,
        recovery_max_steps=args.recovery_max_steps,
        goal_radius=args.goal_radius,
        heading_guard_dist=args.heading_guard_dist,
        goal_grace_steps=args.goal_grace_steps,
        goal_stop_patience=args.goal_stop_patience,
        # 🔥 Stuck Detection
        stuck_patience=args.stuck_patience,
        stuck_enable=args.stuck_enable,
        # 🔥 TDR settings
        use_tdr=args.use_tdr,
        tdr_weight=args.tdr_weight,
        tdr_gamma=args.tdr_gamma,
        tdr_strict_mode=args.tdr_strict_mode,
        # 🔥 SFT-only mode
        sft_only=args.sft_only,
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=4096,
        padding_side="right",
    )
    
    # Load model config first
    model_config = transformers.AutoConfig.from_pretrained(args.model_path)
    
    # Load model
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=model_config,
        low_cpu_mem_usage=False,
    )
    
    # Setup model
    model.model.num_history = 8  # config.num_history
    model.reset(1)
    model.to('cuda')
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled for memory efficiency")
    
    # Apply LoRA if enabled
    if config.lora_enable:
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
    
    # Create trainer (no ref_model!)
    trainer = StreamVLNGRPOTrainer(
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
            trainer.baseline_ema = training_state.get('baseline_ema', 0.0)
            trainer.current_update = training_state.get('current_update', 0)  # 🔥 Load for hybrid training
            trainer.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            start_update = training_state['global_step']
            print(f"  Resumed from global_step={start_update}")
            if config.use_hybrid_training:
                print(f"  Current SFT weight: {trainer._get_sft_weight():.3f}")
        

    
    # Train
    trainer.train(num_updates=args.num_updates, start_update=start_update)
    
    # Close wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
