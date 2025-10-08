VISIBLE_DEVICES=0

# Qwen-7B
CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES llamafactory-cli train experiments/configs/cab/train/qwen_lora_sft.yaml

# Llama-8B
CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICES llamafactory-cli train experiments/configs/cab/train/llama3_lora_sft.yaml
