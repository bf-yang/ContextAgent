#!/bin/bash
BASE_DIR=$(pwd) # current directory

# echo "Running SFT training..."
# cd LLaMA-Factory
# bash /home/bufang/ContextAgent/LLaMA-Factory/experiments/cab_lora_train.sh
# cd "$BASE_DIR"

echo "Running SFT evaluation..."
bash src/sft/eval_sft.sh
