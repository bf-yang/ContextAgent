<br>
<p align="center">
<h1 align="center"><strong>ContextAgent: Context-Aware Proactive LLM Agents
with Open-World Sensory Perceptions</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=q_KcYaQAAAAJ&hl=zh-CN' target='_blank'>Bufang Yang<sup>†</sup></a>&emsp;
    <a href='https://initxu.github.io/' target='_blank'>Lilin Xu<sup>†</sup></a>&emsp;
    <a href='https://acomze.github.io/' target='_blank'>Liekang Zeng</a>&emsp;
    <a href='https://www.linkedin.com/in/kaiwei-liu-743567219/?originalSubdomain=hk' target='_blank'>Kaiwei Liu</a>&emsp;
    <a href='http://syjiang.com/' target='_blank'>Siyang Jiang</a>&emsp;
    <a href='https://openreview.net/profile?id=~Wenrui_Lu3' target='_blank'>Wenrui Lu</a>&emsp; <br>
    <a href='https://www.ie.cuhk.edu.hk/faculty/chen-hongkai/' 
    target='_blank'>Hongkai Chen</a>&emsp;
    <a href='http://fredjiang.com/' target='_blank'>Xiaofan Jiang</a>&emsp;
    <a href='https://staff.ie.cuhk.edu.hk/~glxing/' target='_blank'>Guoliang Xing</a>&emsp;
    <a href='https://yanzhenyu.com/' target='_blank'>Zhenyu Yan</a>&emsp;
    <br>
    The Chinese University of Hong Kong&emsp;Columbia University
  </p>
</p>


## 🏠 About
<div style="text-align: center;">
    <img src="assets/teaser_contextagent.png" alt="Dialogue_Teaser" width=100% >
</div>
In this paper, we introduce ContextAgent, the first context-aware proactive LLM agent that harnesses extensive sensory contexts for enhanced proactive services.


<!-- ## Overview -->

## Overview
<div style="text-align: center;">
    <img src="assets/overview_contextagent.png" alt="Dialogue_Teaser" width=100% >
</div>

## Project Structure
```
ContextAgent/
├─ data/
│  ├─ cab/
│  ├─ cab_lite/
│  ├─ cab_ood/
├─ prompt/
├─ results/
├─ src/
│  ├─ cab_lite/
│  ├─ cab_ood/
│  ├─ data_process/
│  ├─ icl/
│  ├─ sft/
│  ├─ utils/
├─ .gitignore
├─ README.md
```

## Installation
1. Clone the repository.
```bash
git clone https://github.com/bf-yang/ContextAgent.git
cd ContextAgent
```
2. Install packages.
```bash
conda env create -f environment.yml
conda activate contextagent
```

## Evaluation
### ICL Evaluation
- **Proprietary LLMs.** Use API inference for proprietary LLMs (e.g., GPT-4o)
```shell
bash src/icl/icl_infer_api.sh
```

- **Local LLM Inference.** Test open-source LLMs (e.g., Llama-3.1-8B-Instruct and Qwen2.5-7BInstruct)
```shell
bash src/icl/icl_infer.sh
```

### SFT Evaluation
Running the following script for full SFT experiments, including model training, inference on benchmark, and generate scores.
```shell
bash src/sft/sft_exp.sh
```
> Note that ```sft_exp.sh``` contains two scripts. The fisrt one ```LLaMA-Factory/experiments/configs/lora_train.sh``` is used for model training. The second one ```src/sft/eval_sft.sh``` is used for evaluation of the fine-tuned LLMs.


## Experiment Results
Please refer to our paper for more results.
### Quantitative Results
<div style="text-align: center;">
    <img src="assets/quantitative_results.png" alt="Dialogue_Teaser" width=100% >
</div>

### Qualitative Results
- Proactive Examples
<div style="text-align: center;">
    <img src="assets/qualitative_results_overall_proactive.png" alt="Dialogue_Teaser" width=100% >
</div>

- Non-proactive Examples
<div style="text-align: center;">
    <img src="assets/qualitative_results_overall_noproactive.png" alt="Dialogue_Teaser" width=100% >
</div>

## 🔗 Citation

If you find our work and this codebase helpful, please consider starring this repo 🌟 and cite:

```bibtex
@article{yang2025contextagent,
  title={ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions},
  author={Yang, Bufang and Xu, Lilin and Zeng, Liekang and Liu, Kaiwei and Jiang, Siyang and Lu, Wenrui and Chen, Hongkai and Jiang, Xiaofan and Xing, Guoliang and Yan, Zhenyu},
  journal={arXiv preprint arXiv:2505.14668},
  year={2025}
}
```