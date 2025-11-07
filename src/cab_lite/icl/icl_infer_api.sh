models=("gpt-35-turbo")
# models=("gpt-35-turbo" "gpt-4o-2" "gpt-4o-mini-2b" )
combinations=(
    "true wo_t wo_p"  # zs
    # "false w_t w_p"   # 
    # "false wo_t wo_p" # Context-only (ICL)
    # "false wo_t w_p"  # ICL-P
    # "false w_t wo_p"  # ICL-CoT
)

for model in "${models[@]}"; do
    for combination in "${combinations[@]}"; do
        read -r zs think persona <<< "$combination"
        
        echo "Running experiment for model: $model, zs: $zs, think: $think, persona: $persona"
        
        # icl inference via api
        CUDA_VISIBLE_DEVICES=CUDA_DEVICES=2,3,4,7 python src/icl/inference_api.py --dataset cab_lite --model_base $model --zs $zs --think $think --personas $persona
        
        # score calculation
        python src/calculate_scores.py --dataset cab_lite --methods icl --model_base_icl $model --zs $zs --think $think --personas $persona
    done
done