# ICL Setting
This folder provides scripts for evaluating different LLMs under In-Context Learning (ICL) settings.  It supports multiple base models (e.g., GPT-4o, Qwen, LLaMA, and DeepSeek series) and two execution modes: **`live`** and **`sandbox`**.

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
- **Open-source models.** Test open-source LLMs (e.g., Llama-3.1-8B-Instruct and Qwen2.5-7BInstruct).
```
CUDA_VISIBLE_DEVICES=0,2 python src/icl/inference.py --model <MODEL_NAME> --mode sandbox
```

- **Proprietary LLMs.** Use API inference for proprietary LLMs (e.g., GPT-4o).
```
python src/icl/inference_api.py --model <MODEL_NAME> --mode sandbox
```

| Argument  | Type   | Description                                                                 |
|-----------|--------|-----------------------------------------------------------------------------|
| `--model` | string | Base model to evaluate (e.g., `qwen2.5:latest`, `llama3.1:8b`, `deepseek-r1`) 
| `--mode`  | string | • **`live`** – the agent actually executes external tools and APIs <br>• **`sandbox`** – the agent uses predefined sandboxed results without making real API calls |

### ️ 📊 Scoring
After inference finishes, compute metrics per model:
```
python src/calculate_scores.py --methods icl --model_base_icl <MODEL_NAME>
```

### ️ ⚙️ Different Settings
Use the scripts below to run ICL (in-context learning) performance experiments under different settings:
```
bash src/icl/icl_infer.sh
bash src/icl/icl_infer_api.sh
```
> [!NOTE]
> - icl_infer.sh runs experiments using open-source models.
> - icl_infer_api.sh runs experiments using proprietary models.

Here are some additional settings you can test:
| Setting              | Description                          |
|---------------------|--------------------------------------|
| `"true wo_t wo_p"`  | zero-shot (no tools, no extra prompt) |
| `"false w_t w_p"`   | xxx                                  |
| `"false wo_t wo_p"` | ICL                                  |
| `"false wo_t w_p"`  | ICL-P                                |
| `"false w_t wo_p"`  | ICL-CoT                              |



