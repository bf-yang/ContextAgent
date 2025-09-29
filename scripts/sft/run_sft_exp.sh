#!/bin/bash
BASE_DIR=$(pwd) # current directory

echo "Running SFT training..."
cd LLaMA-Factory
bash experiments/cab_lora_train_full_exp.sh
# bash experiments/cab_lora_train.sh
cd "$BASE_DIR"

echo "Running SFT evaluation..."
bash scripts/sft/run_sft_eval_full_exp.sh
# bash scripts/sft/run_sft_eval.sh