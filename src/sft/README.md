# SFT Setting
This folder provides scripts for evaluating different LLMs under SFT settings.  It supports multiple base models (e.g., Qwen, LLaMA, DeepSeek series) and two execution modes: **`live`** and **`sandbox`**.

### 🔑 API Keys
Choose ONE to configure environment variables for tool integrations (Google Maps, AMap, LocationIQ, SerpAPI, etc.):

#### Source a shell script (recommended)
```bash
$EDITOR scripts/env/export_env.sh   # fill your keys
source scripts/env/export_env.sh
```

#### Or export inline
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
bash scripts/sft/run_sft_exp.sh
```
> Note:
> - `run_sft_exp.sh` orchestrates training and evaluation. It calls `LLaMA-Factory/experiments/cab_lora_train.sh` for training and `scripts/sft/run_sft_eval.sh` for evaluation.
> - Training recipe: Edit `LLaMA-Factory/experiments/cab_lora_train.sh` to set the base model and LoRA/SFT training configs.
> - Evaluation setup: Edit `scripts/sft/run_sft_eval.sh` to specify the base model and evaluation mode.


| Argument  | Type   | Description                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| `--model` | string | Base model to evaluate (e.g., `qwen2.5:latest`, `llama3.1:8b`, `deepseek-r1`) 
| `--mode`  | string | • **`live`** – the agent actually executes external tools and APIs <br>• **`sandbox`** – the agent uses predefined sandboxed results without making real API calls |





