#!/bin/bash

# GPU Configuration
# Set CUDA_VISIBLE_DEVICES environment variable or use default
# Examples:
#   export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use GPUs 0,1,2,3
#   export CUDA_VISIBLE_DEVICES=0        # Use only GPU 0
#   export CUDA_VISIBLE_DEVICES=""       # Use CPU only
# If not set, will use all available GPUs

models=("gpt-35-turbo")
# models=("gpt-35-turbo" "gpt-4o-2" "gpt-4o-mini-2b" )
combinations=(
    "true wo_t wo_p"  # zs
    "false w_t w_p"   # 
    # "false wo_t wo_p" # Context-only (ICL)
    # "false wo_t w_p"  # ICL-P
    # "false w_t wo_p"  # ICL-CoT
)

for model in "${models[@]}"; do
    for combination in "${combinations[@]}"; do
        read -r zs think persona <<< "$combination"
        
        echo "Running experiment for model: $model, zs: $zs, think: $think, persona: $persona"
        echo "Using GPUs: ${CUDA_VISIBLE_DEVICES:-all available}"
        
        # icl inference via api
        python src/icl/inference_api.py --mode sandbox --dataset cab --model_base $model --zs $zs --think $think --personas $persona
        
        # score calculation
        python src/calculate_scores.py --dataset cab --methods icl --model_base_icl $model --zs $zs --think $think --personas $persona
    done
done