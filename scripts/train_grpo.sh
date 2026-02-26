#!/bin/bash
# StreamVLN GRPO Training Script
#
# Usage: bash scripts/train_grpo.sh  (run from project root)

# Auto cd to project root (parent of scripts/)
cd "$(dirname "$0")/.." || exit 1
echo "Working directory: $(pwd)"
# Usage: bash scripts/train_grpo.sh [phase1|phase2|phase3] [num_updates]

set -e

# ============================================================
# Environment Setup
# ============================================================

# Set CUDA device (change as needed)
export CUDA_VISIBLE_DEVICES=0

# Suppress Habitat/Magnum verbose logging
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export GLOG_logtostderr=0

# ============================================================
# Configuration
# ============================================================

# Model path (change to your model)
MODEL_PATH="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"

# Habitat config
HABITAT_CONFIG="config/vln_r2r.yaml"

# Base output directory
OUTPUT_BASE="result/grpo_training"

# Training phase (default: phase1_stop)
PHASE=${1:-"phase1_stop"}

# Number of updates (default from phase config)
NUM_UPDATES_ARG=${2:-""}

# Resume from checkpoint (optional, e.g., "./results/grpo_training/phase1_stop/checkpoint_100")
RESUME_FROM=${3:-""}

echo "============================================================"
echo "StreamVLN GRPO Training"
echo "============================================================"
echo "Phase: ${PHASE}"
echo "Model: ${MODEL_PATH}"
echo "CUDA Device: ${CUDA_VISIBLE_DEVICES}"
if [ -n "${RESUME_FROM}" ]; then
    echo "Resume from: ${RESUME_FROM}"
fi
echo "============================================================"

# ============================================================
# Phase-specific configurations
# ============================================================

case $PHASE in
    "phase1_stop"|"phase1")
        PHASE_NAME="phase1_stop"
        NUM_UPDATES="${NUM_UPDATES_ARG:-500}"
        NUM_EPISODES=2          # Instructions per update (reduced for speed)
        GROUP_SIZE=4            # Samples per instruction (total: 2*2=4 trajectories)
        LR="2e-5"
        PPO_EPOCHS=1            # Can be higher than PPO (no KL constraint)
        MINI_BATCH=1            # 🔥 Set to 1 to minimize memory per forward pass
        GRAD_ACCUM=2            # 🔥 Use gradient accumulation (effective batch size = 1*2=2)
        MAX_GRAD_NORM="0.5"
        OUTPUT_PATH="${OUTPUT_BASE}/phase1_stop_v6"
        ;;
    "phase1_test"|"test")
        # Quick test with minimal samples
        PHASE_NAME="phase1_stop"
        NUM_UPDATES="${NUM_UPDATES_ARG:-50}"
        NUM_EPISODES=2
        GROUP_SIZE=2
        LR="2e-7"
        PPO_EPOCHS=2
        MINI_BATCH=1
        GRAD_ACCUM=2
        MAX_GRAD_NORM="0.5"
        OUTPUT_PATH="${OUTPUT_BASE}/phase1_test11"
        ;;
    "phase2_spl"|"phase2")
        PHASE_NAME="phase2_spl"
        NUM_UPDATES="${NUM_UPDATES_ARG:-1000}"
        NUM_EPISODES=4
        GROUP_SIZE=6            # Larger group for more comparisons
        LR="5e-7"
        PPO_EPOCHS=4
        MINI_BATCH=1
        GRAD_ACCUM=4
        OUTPUT_PATH="${OUTPUT_BASE}/phase2_spl"
        ;;
    "phase3_instruction"|"phase3")
        PHASE_NAME="phase3_instruction"
        NUM_UPDATES="${NUM_UPDATES_ARG:-1500}"
        NUM_EPISODES=4
        GROUP_SIZE=6
        LR="3e-7"
        PPO_EPOCHS=3
        MINI_BATCH=1
        GRAD_ACCUM=4
        OUTPUT_PATH="${OUTPUT_BASE}/phase3_instruction"
        ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: bash scripts/train_grpo.sh [phase1|phase2|phase3|test] [num_updates] [resume_from]"
        exit 1
        ;;
esac

# Create output directory
mkdir -p ${OUTPUT_PATH}

# ============================================================
# Run Training
# ============================================================

echo ""
echo "Starting GRPO training..."
echo "Output: ${OUTPUT_PATH}"
echo "Total samples per update: ${NUM_EPISODES} x ${GROUP_SIZE} = $((NUM_EPISODES * GROUP_SIZE)) trajectories"
echo ""

# Build resume argument
RESUME_ARG=""
if [ -n "${RESUME_FROM}" ]; then
    RESUME_ARG="--resume_from ${RESUME_FROM}"
fi

python streamvln/streamvln_grpo_train.py \
    --model_path ${MODEL_PATH} \
    --habitat_config_path ${HABITAT_CONFIG} \
    --output_path ${OUTPUT_PATH} \
    --phase ${PHASE_NAME} \
    --num_updates ${NUM_UPDATES} \
    --num_episodes_per_update ${NUM_EPISODES} \
    --group_size ${GROUP_SIZE} \
    --learning_rate ${LR} \
    --ppo_epochs ${PPO_EPOCHS} \
    --mini_batch_size ${MINI_BATCH} \
    --gradient_accumulation_steps ${GRAD_ACCUM:-1} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    ${RESUME_ARG} \
    2>&1 | tee -a ${OUTPUT_PATH}/train.log

echo ""
echo "============================================================"
echo "GRPO Training completed!"
echo "Output saved to: ${OUTPUT_PATH}"
echo "============================================================"
