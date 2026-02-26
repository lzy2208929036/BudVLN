"""
VLN Reward Functions for Reinforcement Learning

This module implements modular reward functions for Vision-and-Language Navigation.
Rewards are designed to be composable and configurable for different training phases:
- Phase 1: Stop optimization
- Phase 2: SPL/efficiency optimization  
- Phase 3: Instruction alignment

Author: StreamVLN Team
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum


class RewardPhase(Enum):
    """Training phases for curriculum learning"""
    PHASE1_STOP = "phase1_stop"
    PHASE2_SPL = "phase2_spl"
    PHASE3_INSTRUCTION = "phase3_instruction"


@dataclass
class RewardConfig:
    """Configuration for VLN reward function weights and parameters.
    
    This config allows easy switching between different reward configurations
    for ablation studies and curriculum learning.
    """
    # ============ Phase 1: Stop Optimization ============
    # Success/Failure rewards
    success_reward: float = 10.0           # Reward for successfully reaching goal and stopping
    failure_reward: float = -5.0           # Penalty for failed episode
    wrong_stop_penalty: float = -5.0       # Penalty for stopping at wrong location
    miss_stop_penalty: float = -2.0        # Penalty for not stopping when should have
    
    # ============ Phase 2: SPL Optimization ============
    # Distance-based rewards
    distance_reward_scale: float = 1.0     # Scale for distance reduction reward
    distance_penalty_scale: float = 1.5    # Scale for distance increase penalty (asymmetric)
    
    # Progress reward (encourage consistent movement toward goal)
    progress_reward_scale: float = 0.3     # Scale for progress bonus when moving toward goal
    
    # Efficiency rewards
    step_penalty: float = -0.01            # Small penalty per step to encourage efficiency
    collision_penalty: float = -0.1        # Penalty for collisions
    
    # Path similarity
    path_similarity_reward_scale: float = 0.5   # Scale for path similarity reward
    use_path_similarity: bool = True       # Whether to use path similarity reward
    
    # ============ Phase 3: Instruction Alignment ============
    # Instruction matching rewards
    instruction_progress_scale: float = 1.0  # Scale for instruction progress reward
    landmark_match_reward: float = 2.0       # Reward for matching landmarks in instruction
    subgoal_completion_reward: float = 3.0   # Reward for completing a subgoal
    
    # ============ Global Parameters ============
    # Goal radius for success determination
    goal_radius: float = 3.0               # Within this distance is considered success
    
    # Discount and normalization
    gamma: float = 0.99                    # Discount factor for returns
    normalize_rewards: bool = False        # Whether to normalize rewards
    reward_clip: Optional[float] = 10.0    # Clip rewards to this range [-clip, clip]
    
    # Phase selection (determines which rewards are active)
    phase: RewardPhase = RewardPhase.PHASE1_STOP
    
    def get_phase_config(self) -> 'RewardConfig':
        """Get a config optimized for the current phase."""
        if self.phase == RewardPhase.PHASE1_STOP:
            # Phase 1: Sparse reward (ETP-R1 style) - 避免密集奖励导致的"早停"问题
            # 核心思想：移除中间步骤的所有奖励/惩罚，只在episode结束时结算
            return RewardConfig(
                success_reward=2.0,           # 🔴 成功基础奖励（稀疏）
                failure_reward=0.0,           # 🔴 失败不惩罚（由距离引导）
                wrong_stop_penalty=0.0,       # 🔴 禁用：由距离自然惩罚
                miss_stop_penalty=0.0,        # 🔴 禁用
                distance_reward_scale=0.0,    # 🔴 禁用中间步距离奖励
                distance_penalty_scale=0.0,   # 🔴 禁用中间步距离惩罚
                progress_reward_scale=0.0,    # 🔴 禁用
                step_penalty=0.0,             # 🔴 禁用步数惩罚（关键！）
                collision_penalty=0.0,        # 🔴 禁用碰撞惩罚
                use_path_similarity=False,
                reward_clip=10.0,
                phase=RewardPhase.PHASE1_STOP
            )
        elif self.phase == RewardPhase.PHASE2_SPL:
            # Phase 2: Same sparse reward as Phase 1 (consistent strategy)
            return RewardConfig(
                success_reward=2.0,
                failure_reward=0.0,
                wrong_stop_penalty=0.0,
                miss_stop_penalty=0.0,
                distance_reward_scale=0.0,
                distance_penalty_scale=0.0,
                progress_reward_scale=0.0,
                step_penalty=0.0,
                collision_penalty=0.0,
                use_path_similarity=False,
                path_similarity_reward_scale=0.0,
                reward_clip=10.0,
                phase=RewardPhase.PHASE2_SPL
            )
        elif self.phase == RewardPhase.PHASE3_INSTRUCTION:
            # Phase 3: Full reward with instruction alignment
            return RewardConfig(
                success_reward=10.0,
                failure_reward=-5.0,
                distance_reward_scale=0.8,
                step_penalty=-0.01,
                use_path_similarity=True,
                path_similarity_reward_scale=0.3,
                instruction_progress_scale=1.0,
                landmark_match_reward=2.0,
                subgoal_completion_reward=3.0,
                phase=RewardPhase.PHASE3_INSTRUCTION
            )
        return self


class VLNRewardFunction:
    """
    Modular reward function for VLN reinforcement learning.
    
    This class computes rewards based on:
    1. Distance progress towards goal
    2. Success/failure outcomes
    3. Path efficiency and similarity
    4. Instruction alignment (Phase 3)
    
    Usage:
        config = RewardConfig(phase=RewardPhase.PHASE1_STOP)
        reward_fn = VLNRewardFunction(config)
        
        reward, reward_dict = reward_fn.compute_step_reward(
            prev_state, curr_state, action, info
        )
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self._episode_rewards = []
        self._prev_distance_to_goal = None
        
    def reset(self):
        """Reset at the beginning of each episode."""
        self._episode_rewards = []
        self._prev_distance_to_goal = None
        
    def compute_step_reward(
        self,
        prev_distance_to_goal: float,
        curr_distance_to_goal: float,
        action: int,
        info: Dict,
        is_done: bool = False,
        reference_path: Optional[List] = None,
        agent_position: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for a single step (稀疏奖励版本 - ETP-R1 style).
        
        核心设计：
        - 中间步骤：返回0奖励（避免密集惩罚导致"早停"）
        - 结束步骤：计算距离引导 + 成功奖励 + SPL奖励
        
        Args:
            prev_distance_to_goal: Distance to goal before action
            curr_distance_to_goal: Distance to goal after action
            action: Action taken (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
            info: Info dict from environment containing metrics
            is_done: Whether episode is done
            reference_path: Optional reference path for SPL computation
            agent_position: Current agent position (unused in sparse reward)
            
        Returns:
            total_reward: Sum of all reward components
            reward_dict: Dictionary with individual reward components
        """
        reward_dict = {}
        
        # ============ 🔥 半稀疏奖励核心逻辑 ============
        if not is_done:
            # 🔥 中间步骤：微小引导信号（避免完全迷失）
            
            # 1. 微小距离引导：告诉模型方向是否正确
            #    累积50步也比不上成功奖励(+2.0)，不会导致绕路刷分
            if curr_distance_to_goal < prev_distance_to_goal:
                reward_dict['distance_guidance'] = 0.05  # 靠近目标：+0.05
            else:
                reward_dict['distance_guidance'] = -0.05  # 远离目标：-0.05
            
            # 2. 碰撞惩罚：即时准确的负反馈
            #    撞墙既没用又浪费步数，必须快速纠正
            if info.get('collided', False) or info.get('collision', False):
                reward_dict['collision'] = -0.1
            else:
                reward_dict['collision'] = 0.0
            
            # 3. 严禁step penalty！
            #    绝对不加-0.01之类的步数惩罚，会导致early stop
            
            reward_dict['success'] = 0.0
            reward_dict['spl'] = 0.0
            total_reward = sum(reward_dict.values())
        else:
            # 🔥 结束步骤：根据结果给奖励
            is_success = info.get('success', False)
            
            # 1. 距离引导惩罚（无论成败都有，保证梯度信号）
            # 距离越近惩罚越小，鼓励即使失败也要靠近目标
            distance_penalty = -(curr_distance_to_goal / 6.0)
            reward_dict['distance_guidance'] = distance_penalty
            
            # 2. 成功基础奖励
            if is_success:
                reward_dict['success'] = self.config.success_reward  # 默认+2.0
            else:
                reward_dict['success'] = 0.0
            
            # 3. SPL效率奖励（只有成功才有）
            if is_success and reference_path is not None:
                # 计算SPL: reference_length / max(reference_length, agent_length)
                reference_length = self._compute_path_length(reference_path)
                agent_length = info.get('agent_path_length', info.get('num_steps', 1) * 0.25)  # 假设每步0.25米
                
                if agent_length > 0:
                    spl = reference_length / max(reference_length, agent_length)
                    reward_dict['spl'] = 1.0 * spl  # SPL权重为1.0
                else:
                    reward_dict['spl'] = 0.0
            else:
                reward_dict['spl'] = 0.0
            
            # 总奖励
            total_reward = sum(reward_dict.values())
            
            # Clip rewards if configured
            if self.config.reward_clip is not None:
                total_reward = np.clip(total_reward, -self.config.reward_clip, self.config.reward_clip)
        
        # Store for episode statistics
        self._episode_rewards.append(total_reward)
        
        return total_reward, reward_dict
    
    def compute_episode_reward(
        self,
        trajectory: List[Dict],
        final_info: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for an entire episode trajectory.
        
        This is useful for sparse reward settings where we only give
        reward at the end of the episode.
        
        Args:
            trajectory: List of step data dicts
            final_info: Final info dict from environment
            
        Returns:
            total_reward: Episode total reward
            reward_summary: Summary statistics of rewards
        """
        episode_reward = 0.0
        reward_components = {
            'distance': 0.0,
            'step': 0.0,
            'collision': 0.0,
            'success': 0.0,
            'stop': 0.0,
            'path_similarity': 0.0,
        }
        
        for i, step_data in enumerate(trajectory):
            is_last = (i == len(trajectory) - 1)
            
            step_reward, step_dict = self.compute_step_reward(
                prev_distance_to_goal=step_data.get('prev_dist', 0),
                curr_distance_to_goal=step_data.get('curr_dist', 0),
                action=step_data.get('action', 0),
                info=step_data.get('info', {}),
                is_done=is_last,
                reference_path=step_data.get('reference_path'),
                agent_position=step_data.get('position'),
            )
            
            episode_reward += step_reward
            for k, v in step_dict.items():
                if k in reward_components:
                    reward_components[k] += v
                    
        # Add final metrics
        reward_summary = {
            'total': episode_reward,
            **reward_components,
            'success': float(final_info.get('success', False)),
            'spl': final_info.get('spl', 0.0),
            'distance_to_goal': final_info.get('distance_to_goal', 0.0),
            'num_steps': len(trajectory),
        }
        
        return episode_reward, reward_summary
    
    def _compute_path_length(self, path: List) -> float:
        """
        Compute total length of a path.
        
        Args:
            path: List of waypoints (each waypoint is [x, y, z] or similar)
            
        Returns:
            Total path length in meters
        """
        if len(path) <= 1:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i + 1])
            # Use 2D distance (x, y) ignoring height
            if len(p1) >= 2 and len(p2) >= 2:
                dist = np.linalg.norm(p2[:2] - p1[:2])
                total_length += dist
        
        return total_length
    
    def _compute_path_similarity(
        self,
        agent_position: np.ndarray,
        reference_path: List[np.ndarray],
    ) -> float:
        """
        Compute how close the agent is to the reference path.
        
        Returns a value in [0, 1] where 1 means on the path.
        """
        if len(reference_path) == 0:
            return 0.0
            
        # Find minimum distance to any point on reference path
        min_dist = float('inf')
        for ref_point in reference_path:
            ref_point = np.array(ref_point)
            if len(ref_point) >= 2 and len(agent_position) >= 2:
                dist = np.linalg.norm(agent_position[:2] - ref_point[:2])
                min_dist = min(min_dist, dist)
                
        # Convert to reward (closer = higher reward)
        # Using exponential decay: reward = exp(-dist / scale)
        scale = 2.0  # Distance scale parameter
        similarity = np.exp(-min_dist / scale)
        
        return similarity
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get statistics for the current episode."""
        if len(self._episode_rewards) == 0:
            return {'mean': 0.0, 'sum': 0.0, 'max': 0.0, 'min': 0.0}
            
        return {
            'mean': np.mean(self._episode_rewards),
            'sum': np.sum(self._episode_rewards),
            'max': np.max(self._episode_rewards),
            'min': np.min(self._episode_rewards),
            'num_steps': len(self._episode_rewards),
        }


class InstructionProgressTracker:
    """
    Track progress through natural language instructions.
    
    This is a placeholder for Phase 3 implementation.
    It would parse instructions into subgoals and track completion.
    """
    
    def __init__(self, instruction: str = ""):
        self.subgoals = []
        self.completed = []
        self.phrases = []  # Alias for subgoals
        
        if instruction:
            self.parse_instruction(instruction)
        
    def parse_instruction(self, instruction: str) -> List[str]:
        """Parse instruction into subgoals."""
        # Simple heuristic: split by common connectors
        connectors = [', then ', '. Then ', ' and then ', '. ', ', ']
        
        subgoals = [instruction]
        for conn in connectors:
            new_subgoals = []
            for sg in subgoals:
                new_subgoals.extend(sg.split(conn))
            subgoals = new_subgoals
            
        # Clean up
        subgoals = [sg.strip() for sg in subgoals if sg.strip()]
        self.subgoals = subgoals
        self.phrases = subgoals  # Alias
        self.completed = [False] * len(subgoals)
        
        return subgoals
    
    def mark_completed(self, index: int):
        """Mark a subgoal as completed."""
        if 0 <= index < len(self.completed):
            self.completed[index] = True
    
    def get_progress(self) -> float:
        """Get completion progress as a ratio."""
        if len(self.completed) == 0:
            return 0.0
        return sum(self.completed) / len(self.completed)
    
    def check_progress(
        self,
        observation: Dict,
        action: int,
        agent_position: np.ndarray,
    ) -> Tuple[float, int]:
        """
        Check if any subgoals were completed.
        
        Returns:
            reward: Reward for progress
            num_completed: Number of newly completed subgoals
        """
        # Placeholder - would need semantic understanding
        # Could use CLIP similarity, object detection, etc.
        return 0.0, 0
    
    def reset(self, instruction: str = ""):
        """Reset for new episode."""
        self.subgoals = []
        self.phrases = []
        self.completed = []
        if instruction:
            self.parse_instruction(instruction)


# ============ Utility Functions ============

def compute_discounted_returns(
    rewards: List[float],
    gamma: float = 0.99,
) -> List[float]:
    """Compute discounted returns for a trajectory."""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Value estimate for next state
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = []
    gae = 0
    
    values = values + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    
    return advantages, returns
