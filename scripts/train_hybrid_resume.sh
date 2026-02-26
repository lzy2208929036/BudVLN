#!/bin/bash
# Resume Hybrid Training Script
# This script resumes training from a saved checkpoint
#
# Usage: bash scripts/train_hybrid_resume.sh  (run from project root)

# Auto cd to project root (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"

# 🔥 Specify the checkpoint directory to resume from
# Change this to your actual checkpoint path
RESUME_CHECKPOINT="result/your_training_output/checkpoint_XXX"

MODEL_PATH="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
HABITAT_CONFIG="config/vln_r2r_rxr.yaml"
OUTPUT_PATH="result/grpo_hybrid_trainingV10_multi_dataset"
PHASE="phase1_stop"

# Check if checkpoint exists
if [ ! -d "${RESUME_CHECKPOINT}" ]; then
    echo "❌ Error: Checkpoint directory not found: ${RESUME_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${RESUME_CHECKPOINT}/training_state.pt" ]; then
    echo "❌ Error: training_state.pt not found in checkpoint"
    exit 1
fi

echo "✅ Resuming training from: ${RESUME_CHECKPOINT}"

# Create output directory if it doesn't exist
if [ ! -d "${OUTPUT_PATH}" ]; then
    echo "📁 Creating output directory: ${OUTPUT_PATH}"
    mkdir -p ${OUTPUT_PATH}
else
    echo "✅ Output directory exists: ${OUTPUT_PATH}"
fi

python -u streamvln/streamvln_grpo_train.py \
    --model_path ${MODEL_PATH} \
    --resume_from ${RESUME_CHECKPOINT} \
    --habitat_config_path ${HABITAT_CONFIG} \
    --output_path ${OUTPUT_PATH} \
    --phase ${PHASE} \
    --num_updates 1000 \
    --num_episodes_per_update 1 \
    --group_size 2 \
    --learning_rate 5e-6 \
    --ppo_epochs 1 \
    --mini_batch_size 4 \
    --max_grad_norm 0.5 \
    --use_wandb \
    --wandb_project "streamvln_hybrid" \
    --wandb_run_name "grpo_sft_hybrid_resumed_$(date +%Y%m%d_%H%M%S)" \
    --use_hybrid_training \
    --sft_loss_start_weight 1.0 \
    --sft_loss_end_weight 0.95 \
    --sft_loss_decay_updates 1000 \
    --sft_loss_decay_type cosine \
    --no_tdr \
    --enable_recovery \
    --offtrack_dist_thresh 3.0 \
    --offtrack_heading_thresh_deg 120.0 \
    --offtrack_patience 8 \
    --lookahead_k 1 \
    --recovery_dist_thresh 2.0 \
    --recovery_heading_thresh_deg 45.0 \
    --recovery_max_steps 100 \
    --goal_radius 3 \
    --oracle_goal_radius 1.0 \
    --heading_guard_dist 3.0 \
    --goal_grace_steps 7 \
    --goal_stop_patience 5 \
    2>&1 | tee -a ${OUTPUT_PATH}/training.log
