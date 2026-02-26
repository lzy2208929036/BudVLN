#!/bin/bash
# Hybrid Training Script: GRPO + SFT
# This script demonstrates how to use the hybrid training mode
# where GRPO's reward-based learning is combined with supervised learning on ground truth actions.
#
# Usage: bash scripts/train_hybrid.sh  (run from project root)

# Auto cd to project root (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# 🔥 Key Features:
# 1. All sampled trajectories use their gt_actions for SFT loss
# 2. SFT weight dynamically decays from high (1.0) to stable (0.5)
# 3. Loss = GRPO_loss + dynamic_weight * SFT_loss

MODEL_PATH="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
# 使用 R2R + RxR 混合数据集 (通过 scripts/merge_r2r_rxr.py 生成)
HABITAT_CONFIG="config/vln_r2r_rxr.yaml"
OUTPUT_PATH="result/grpo_hybrid_trainingV10_multi_dataset"
PHASE="phase1_stop"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_PATH}

python -u streamvln/streamvln_grpo_train.py \
    --model_path ${MODEL_PATH} \
    --habitat_config_path ${HABITAT_CONFIG} \
    --output_path ${OUTPUT_PATH} \
    --phase ${PHASE} \
    --num_updates 500 \
    --num_episodes_per_update 1 \
    --group_size 2 \
    --learning_rate 5e-7 \
    --ppo_epochs 1 \
    --mini_batch_size 2 \
    --max_grad_norm 0.5 \
    --use_wandb \
    --wandb_project "streamvln_hybrid" \
    --wandb_run_name "grpo_sft_hybrid_$(date +%Y%m%d_%H%M%S)" \
    --use_hybrid_training \
    --sft_loss_start_weight 1.0 \
    --sft_loss_end_weight 0.9 \
    --sft_loss_decay_updates 600 \
    --sft_loss_decay_type cosine \
    --enable_recovery \
    --offtrack_dist_thresh 3.0 \
    --offtrack_heading_thresh_deg 120.0 \
    --offtrack_patience 8 \
    --lookahead_k 1 \
    --recovery_dist_thresh 2.0 \
    --recovery_heading_thresh_deg 45.0 \
    --recovery_max_steps 100 \
    --goal_radius 3 \
    --heading_guard_dist 3.0 \
    --goal_grace_steps 7 \
    --goal_stop_patience 5 \
    2>&1 | tee -a ${OUTPUT_PATH}/training.log

# Explanation of Hybrid Training Parameters:
# 
# --use_hybrid_training: Enable hybrid GRPO+SFT training
#
# 🔥 SFT Data Source Control (NEW):
# --sft_use_policy_steps (default): SFT uses both oracle steps AND policy steps with valid GT
#   - Oracle steps: Expert demonstrations (from recovery intervention)
#   - Policy steps: Model's own actions that have valid GT from TDR
#   - Balanced learning from both exploration and expert guidance
#
# --sft_oracle_only: SFT uses ONLY oracle steps (exclude policy steps)
#   - Pure imitation learning on expert demonstrations
#   - More conservative, less exploration
#   - Useful for ablation studies or when policy quality is poor
#
# Note: Oracle steps always get 2x weight in SFT loss regardless of this setting
#
# --sft_loss_start_weight 1.0: Initial SFT weight (high)
#   - Early training emphasizes learning from expert demonstrations
#   - Provides stable learning signal when model is still weak
#
# --sft_loss_end_weight 0.5: Final SFT weight (stable)
#   - Remains significant throughout training
#   - Prevents catastrophic forgetting of expert knowledge
#
# --sft_loss_decay_updates 200: Number of updates for weight transition
#   - Smooth decay over first 200 updates
#   - After 200 updates, weight stabilizes at end_weight
#
# --sft_loss_decay_type cosine: Decay schedule type
#   - 'linear': Linear interpolation (simple)
#   - 'cosine': Smooth cosine annealing (recommended)
#   - 'exponential': Exponential decay (fast initial drop)
#
# 🔥 Expert Intervention Parameters:
#
# --enable_recovery: Enable expert intervention (default: enabled)
#   - Use --disable_recovery to turn off for ablation studies
#
# --offtrack_dist_thresh 3.0: Distance threshold to trigger intervention (meters)
#   - Larger values = more exploration before intervention
#   - Smaller values = more conservative, earlier intervention
#
# --offtrack_heading_thresh_deg 90.0: Heading error threshold (degrees)
#   - 90° = triggers when facing perpendicular or away from target
#   - Larger values = only intervene on severe misalignment
#
# --offtrack_patience 2: Consecutive off-track steps before intervention
#   - Prevents intervention on temporary deviations
#   - Higher values = more lenient
#
# --lookahead_k 1: Number of waypoints to look ahead on reference path
#   - Helps agent anticipate upcoming turns
#   - Larger values = smoother recovery trajectories
#
# --recovery_dist_thresh 1.0: Distance to exit recovery mode (meters)
#   - How close to path before returning control to model
#
# --recovery_heading_thresh_deg 30.0: Heading alignment to exit recovery (degrees)
#   - Ensures agent is facing correct direction before resuming
#
# --recovery_max_steps 40: Maximum recovery steps (safety stop)
#   - Prevents infinite recovery loops
#   - Episode terminates if this limit is reached
#
# 🔥 Goal Zone Protection Parameters:
#
# --goal_radius 3.0: Goal zone radius (meters)
#   - Recovery completely disabled within this radius
#   - Matches reward function's goal detection threshold
#   - Prevents recovery triggering when agent reaches goal
#
# --heading_guard_dist 1.0: Minimum distance for heading check (meters)
#   - Heading error only triggers recovery when far from path
#   - Allows agent to turn freely when close to path
#   - Default = recovery_dist_thresh for consistency
#
# --goal_stop_patience 5: Steps before forcing STOP in goal zone
#   - If > 0: Forces STOP after N steps in goal zone
#   - If = 0: Disabled, agent decides when to STOP
#   - Forced STOP is marked as oracle demonstration (2x SFT weight)

# Expected Behavior:
# Update 1-50:    SFT weight ≈ 0.8-1.0 (high supervision)
# Update 51-150:  SFT weight ≈ 0.6-0.8 (gradual transition)
# Update 151-200: SFT weight ≈ 0.5-0.6 (approaching stable)
# Update 201+:    SFT weight = 0.5     (stable supervision)
#
# Recovery Intervention:
# - Activates when agent strays >3m from path OR faces >90° away
# - Expert takes control and guides back to path
# - Control returns when <1m from path AND facing <30° off
# - Oracle demonstrations get 2x weight in SFT loss
