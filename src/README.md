# ICL Inference
### API
Use API inference for proprietary LLMs (e.g., GPT-4o)
```shell
bash src/icl/icl_infer_api.sh
```


### Local Inference
Test open-source LLMs (e.g., Llama-3.1-8B-Instruct and Qwen2.5-7BInstruct)
```shell
bash src/icl/icl_infer.sh
```

# SFT Inference
Running the following script for full SFT experiments, including model training, inference on benchmark, and generate scores.
```shell
bash src/sft/sft_exp.sh
```
> Note that ```sft_exp.sh``` contains two scripts. The fisrt one ```LLaMA-Factory/experiments/configs/lora_train.sh``` is used for model training. The second one ```src/sft/eval_sft.sh``` is used for evaluation of the fine-tuned LLMs.
