## Test different number of examples in prompt (n-shots) for ICL

models=("qwen2.5:latest")
# models=("llama3.1:8b" "qwen2.5:latest" "deepseek-r1" "qwen2.5:3b" "qwen2.5:1.5b" "deepseek-r1:1.5b" "qwen2.5:0.5b" "qwen2.5:72b" "llama3.1:70b" "deepseek-r1:32b" )
combinations=(
    "false w_t w_p"
)
n_fewshots=(0 1 2 3 4 5 6 7 8 9 10 15 20)

for model in "${models[@]}"; do
    for combination in "${combinations[@]}"; do
        read -r zs think persona <<< "$combination"
        
        for n_fewshot in "${n_fewshots[@]}"; do
            echo "Running experiment for model: $model, zs: $zs, think: $think, persona: $persona, n_fewshot: $n_fewshot"
            
            # # icl inference
            # CUDA_VISIBLE_DEVICES=2,3 python src/icl/inference_nshots.py --dataset cab_lite --model_base $model --zs $zs --think $think --personas $persona --n_fewshot $n_fewshot
            
            # score calculation
            python src/calculate_scores_nshots.py --dataset cab_lite --methods icl --model_base_icl $model --zs $zs --think $think --personas $persona --n_fewshot $n_fewshot
        done
    done
done