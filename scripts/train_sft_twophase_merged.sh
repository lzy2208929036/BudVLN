#!/bin/bash
# Two-Phase Separated Training Script (Merged Dataset Version)
#
# Usage: bash scripts/train_sft_twophase_merged.sh  (run from project root)

# Auto cd to project root (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"
# Phase 1: GRPO Policy Optimization (RL)
# Phase 2: Demo Correction Learning (SL, 3 epochs)
# 
# 🔥 Key Improvements over Hybrid Training:
# 1. 100% data utilization (no wasted oracle demos)
# 2. Separated optimization: RL and SL don't interfere
# 3. Each demo learned 3 times for better correction
#
# 🆕 Using Merged Maximum Dataset (without ScaleVLN):
# - R2R: 10,819 episodes (full version)
# - RxR: 58,752 episodes (train_follower.json.gz, multilingual)
# - EnvDrop: 146,304 episodes (envdrop/envdrop.json.gz, full augmentation)
# - Total: 215,875 episodes

MODEL_PATH="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
HABITAT_CONFIG="config/vln_merged_maximum.yaml"  # 🔥 使用合并数据集
OUTPUT_PATH="result/sft_twophase_merged_maximum_v2"
PHASE="phase1_stop"

# Resume from checkpoint (optional, uncomment and set your checkpoint path)
# RESUME_FROM="result/your_training_output/checkpoint_XXX"
RESUME_FROM=""

# Create output directory
mkdir -p ${OUTPUT_PATH}

# Build resume argument
RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
    RESUME_ARG="--resume_from ${RESUME_FROM}"
fi

echo "=================================================="
echo "🚀 Two-Phase Separated Training (Merged Dataset)"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "Config: ${HABITAT_CONFIG}"
echo "Dataset: Merged Maximum (215,875 episodes, without ScaleVLN)"
echo "Output: ${OUTPUT_PATH}"
if [ -n "$RESUME_FROM" ]; then
    echo "Resume: ${RESUME_FROM}"
fi
echo "=================================================="
echo ""

# 🔥 验证数据集
echo "📊 验证数据集配置..."
python scripts/verify_merged_dataset.py 2>/dev/null | grep -A 5 "Maximum"
echo ""

python -u streamvln/streamvln_sft_train.py \
    --model_path ${MODEL_PATH} \
    --habitat_config_path ${HABITAT_CONFIG} \
    --output_path ${OUTPUT_PATH} \
    --phase ${PHASE} \
    ${RESUME_ARG} \
    --num_updates 1000 \
    --num_episodes_per_update 1 \
    --group_size 2 \
    --learning_rate 5e-6 \
    --ppo_epochs 1 \
    --mini_batch_size 4 \
    --max_grad_norm 0.5 \
    --use_wandb \
    --wandb_project "streamvln_twophase_merged" \
    --wandb_run_name "twophase_merged_$(date +%Y%m%d_%H%M%S)" \
    --enable_recovery \
    --offtrack_dist_thresh 3.0 \
    --offtrack_heading_thresh_deg 120.0 \
    --offtrack_patience 8 \
    --lookahead_k 1 \
    --recovery_dist_thresh 2.0 \
    --recovery_heading_thresh_deg 45.0 \
    --recovery_max_steps 100 \
    --goal_radius 3.0 \
    --heading_guard_dist 3.0 \
    --goal_grace_steps 7 \
    --goal_stop_patience 5 \
    --no_progress_enable \
    --no_progress_patience 60 \
    --no_tdr \
    2>&1 | tee -a ${OUTPUT_PATH}/training.log

echo ""
echo "=================================================="
echo "✅ Training completed!"
echo "Output: ${OUTPUT_PATH}"
echo "=================================================="
