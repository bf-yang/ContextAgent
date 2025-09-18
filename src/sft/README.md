# SFT Setting
This folder provides scripts for evaluating different LLMs under SFT settings.  It supports multiple base models (e.g., Qwen, LLaMA, DeepSeek series) and two execution modes: **`live`** and **`sandbox`**.

### 🔑 API Keys
Several experiments rely on external APIs (e.g., Google Maps, AMap, LocationIQ, SerpAPI). Please configure the required keys via environment variables:

```bash
export GOOGLE_MAP_API_KEY=<YOUR_GOOGLE_MAP_API_KEY>
export AMAP_API_KEY=<YOUR_AMAP_API_KEY>
export LOCATIONIQ_API_KEY=<YOUR_LOCATIONIQ_API_KEY>
export SERPAPI_KEY=<YOUR_SERPAPI_KEY>
export GOOGLE_CALENDAR_ACCOUNT=<GOOGLE_CALENDAR_ACCOUNT>
```
### ️ ▶️ Usage
Run the following script:
```
bash src/sft/sft_exp.sh
```
> Note:
> - Note that ```sft_exp.sh``` contains two scripts. The fisrt one ```LLaMA-Factory/experiments/configs/lora_train.sh``` is used for model training. The second one ```src/sft/eval_sft.sh``` is used for evaluation of the fine-tuned LLMs.
> - Training recipe: Edit LLaMA-Factory/experiments/cab_lora_train.sh to set the base model and LoRA/SFT training configs.
> - Evaluation setup: Edit src/sft/eval_sft.sh to specify the base model and evaluation mode.


| Argument  | Type   | Description                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| `--model` | string | Base model to evaluate (e.g., `qwen2.5:latest`, `llama3.1:8b`, `deepseek-r1`) 
| `--mode`  | string | • **`live`** – the agent actually executes external tools and APIs <br>• **`sandbox`** – the agent uses predefined sandboxed results without making real API calls |





