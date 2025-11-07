#!/bin/bash
API_PORT=${API_PORT:-8009}
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
FACTORY_DIR="$BASE_DIR/LLaMA-Factory"

echo "Starting LLM agent on port $API_PORT..."
cd "$FACTORY_DIR" && API_PORT=$API_PORT CUDA_VISIBLE_DEVICES=0 llamafactory-cli api experiments/configs/cab/inference/qwen_lora_sft.yaml
