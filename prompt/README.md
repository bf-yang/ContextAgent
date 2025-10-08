📁 Prompt Directory Organization
```
prompt/
├── prompt_sys.txt              # Main method (with thought generation)
├── prompt_summarize.txt        # Summarization functionality
└── baselines/                  # Baseline methods and variants
    ├── icl_fs.txt             # ICL few-shot (with thought generation)
    ├── icl_fs_wo_t.txt        # ICL few-shot (without thought generation)
    ├── icl_zs.txt             # ICL zero-shot (without thought generation)
    └── prompt_sys_wo_t.txt    # SFT baseline (without thought generation)
```