models=("qwen2.5:latest" "llama3.1:8b")
# models=("llama3.1:8b" "qwen2.5:latest" "deepseek-r1" "qwen2.5:3b" "qwen2.5:1.5b" "deepseek-r1:1.5b" "qwen2.5:0.5b" "qwen2.5:72b" "llama3.1:70b" "deepseek-r1:32b" )
combinations=(
    "true wo_t wo_p"  # zs
    "false w_t w_p"   # 
    "false wo_t wo_p" # Context-only (ICL)
    "false wo_t w_p"  # ICL-P
    "false w_t wo_p"  # ICL-CoT
)

for model in "${models[@]}"; do
    for combination in "${combinations[@]}"; do
        read -r zs think persona <<< "$combination"
        
        echo "Running experiment for model: $model, zs: $zs, think: $think, persona: $persona"
        
        # icl inference
        CUDA_VISIBLE_DEVICES=0,2 python src/icl/inference.py --mode sandbox --dataset cab --model_base $model --zs $zs --think $think --personas $persona

        # score calculation
        python src/calculate_scores.py --dataset cab --methods icl --model_base_icl $model --zs $zs --think $think --personas $persona
    done
done