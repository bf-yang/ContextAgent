#!/bin/bash
API_PORT=8000

echo "Checking and killing processes on port $API_PORT before starting..."
lsof -i:$API_PORT | awk 'NR>1 {print $2}' | xargs -r kill -9


experiments=(
    # Qwen-7B
    "experiments/configs/cab_ood/inference/qwen_lora_sft.yaml|qwen7b|"
    # "experiments/configs/cab_ood/inference/qwen_lora_sft_wo_p.yaml|qwen7b|--personas wo_p"
    # "experiments/configs/cab_ood/inference/qwen_lora_sft_wo_t.yaml|qwen7b|--think wo_t"
    # "experiments/configs/cab_ood/inference/qwen_lora_sft_wo_t_wo_p.yaml|qwen7b|--think wo_t --personas wo_p"

    # # Llama3-8B
    # "experiments/configs/cab_ood/inference/llama3_lora_sft.yaml|llama8b|"
    # "experiments/configs/cab_ood/inference/llama3_lora_sft_wo_p.yaml|llama8b|--personas wo_p"
    # "experiments/configs/cab_ood/inference/llama3_lora_sft_wo_t.yaml|llama8b|--think wo_t"
    # "experiments/configs/cab_ood/inference/llama3_lora_sft_wo_t_wo_p.yaml|llama8b|--think wo_t --personas wo_p"
)

for exp in "${experiments[@]}"; do
    IFS='|' read -r config_file model_base_value eval_args <<< "$exp"
    
    echo "========================================"
    echo "Initializing experiments: $config_file"
    echo "Model: $model_base_value | Arguments: $eval_args"

    # Start the API service in the background
    (
        cd /home/bufang/ContextAgent/LLaMA-Factory && \
        CUDA_VISIBLE_DEVICES=3,4 llamafactory-cli api "$config_file"
    ) &> "api_${model_base_value}.log" &
    API_PID=$!

    # Wait for the API service to be ready
    echo -n "Waiting for API to start..."
    while ! nc -z localhost $API_PORT; do
        sleep 1
    done
    echo "API started!"

    # Run evaluation
    (
        cd /home/bufang/ContextAgent && \
        echo "Starting evaluation..."
        CUDA_VISIBLE_DEVICES=3,4 python src/sft/inference.py \
            --dataset cab_ood \
            --model_base $model_base_value \
            $eval_args
        
        echo "Calculating scores..."
        python src/calculate_scores.py \
            --dataset cab_ood \
            --methods "sft" \
            --model_base_sft $model_base_value \
            $eval_args
    )

    # Kill the API service
    echo "Checking and killing processes on port $API_PORT after finishing..."
    lsof -i:$API_PORT | awk 'NR>1 {print $2}' | xargs -r kill -9

done