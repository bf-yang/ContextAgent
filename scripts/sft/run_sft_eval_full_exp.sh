#!/bin/bash
API_PORT=8009

# Get the base directory (project root) based on script location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
FACTORY_DIR="$BASE_DIR/LLaMA-Factory"

echo "Checking and killing processes on port $API_PORT before starting..."
fuser -k ${API_PORT}/tcp
sleep 2

experiments=(
    # Qwen-7B
    # "experiments/configs/cab/inference/qwen_lora_sft.yaml|qwen7b|"
    "experiments/configs/cab/inference/qwen_lora_sft_wo_t.yaml|qwen7b|--think wo_t"
    "experiments/configs/cab/inference/qwen_lora_sft_wo_p.yaml|qwen7b|--personas wo_p"
    "experiments/configs/cab/inference/qwen_lora_sft_wo_t_wo_p.yaml|qwen7b|--think wo_t --personas wo_p"

    # Llama3-8B
    "experiments/configs/cab/inference/llama3_lora_sft.yaml|llama8b|"
    "experiments/configs/cab/inference/llama3_lora_sft_wo_t.yaml|llama8b|--think wo_t"
    "experiments/configs/cab/inference/llama3_lora_sft_wo_p.yaml|llama8b|--personas wo_p"
    "experiments/configs/cab/inference/llama3_lora_sft_wo_t_wo_p.yaml|llama8b|--think wo_t --personas wo_p"
)

for exp in "${experiments[@]}"; do
    IFS='|' read -r config_file model_base_value eval_args <<< "$exp"
    
    echo "========================================"
    echo "Initializing experiments: $config_file"
    echo "Model: $model_base_value | Arguments: $eval_args"

    # Start the API service in the background
    (
        cd "$FACTORY_DIR" && \
        API_PORT=$API_PORT CUDA_VISIBLE_DEVICES=0 llamafactory-cli api "$config_file"
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
        cd "$BASE_DIR" && \
        echo "Starting evaluation..."
        CUDA_VISIBLE_DEVICES=1,3 python src/sft/inference.py \
            --mode sandbox \
            --dataset cab \
            --model_base $model_base_value \
            $eval_args
        
        echo "Calculating scores..."
        python src/calculate_scores.py \
            --dataset cab \
            --methods sft \
            --model_base_sft $model_base_value \
            $eval_args
    )

    echo "Checking and killing processes on port $API_PORT after finishing..."
    fuser -k ${API_PORT}/tcp
    echo "Waiting for port to be released..."
    sleep 5
done
