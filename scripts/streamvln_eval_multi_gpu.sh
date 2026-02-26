export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
MASTER_PORT=$((RANDOM % 101 + 20000))

CHECKPOINT="checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"
echo "CHECKPOINT: ${CHECKPOINT}"
LORA="result/your_training_output/checkpoint_XXX"
echo "LORA: ${LORA}"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT streamvln/streamvln_eval.py --model_path $CHECKPOINT --lora_path $LORA 
