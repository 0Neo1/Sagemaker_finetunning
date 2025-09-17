# SageMaker Fine-Tuning Labs

This repository contains three Jupyter notebooks that walk through dialogue summarization and model fine-tuning workflows using Hugging Face Transformers, PEFT, and TRL. These labs are designed to be run in an environment with sufficient CPU/RAM (the notebooks include a verifier for an `ml.m5.2xlarge`-like spec).

## Contents
- Lab 1: Summarize Dialogue — Prompt engineering with FLAN-T5 (zero/one/few-shot), generation config basics, and evaluation hints.
- Lab 2: Fine-Tune Generative Model — Full fine-tuning and PEFT (LoRA) on DialogSum with ROUGE evaluation.
- Lab 3: Detoxify Summaries — Reinforcement learning with PPO and a hate-speech reward model to reduce toxicity; uses TRL with PEFT.

## Prerequisites
- Python 3.10+
- pip
- Jupyter Lab/Notebook (or run on Amazon SageMaker Studio/Notebook Instance)
- Sufficient compute (example in notebooks checks for ~8 vCPU / 32 GiB RAM)

## Quickstart (local)
1. Create and activate a virtual environment.
2. Upgrade pip and install dependencies used in the labs:

```bash
pip install --upgrade pip
pip install tensorflow==2.18.0 keras==3.9.0
pip install --no-deps torch==2.5.1 torchdata==0.6.0
pip install datasets==2.17.0 transformers==4.38.2 accelerate==0.28.0 evaluate==0.4.0 rouge_score==0.1.2 peft==0.3.0
# For Lab 3 (PPO with TRL):
pip install --no-deps git+https://github.com/lvwerra/trl.git@25fa1bd
```

3. Launch Jupyter and open the notebooks:
```bash
jupyter lab
# or
jupyter notebook
```

## Running the Labs
- Lab 1: `Lab_1_summarize_dialogue (1).ipynb`
  - Explore zero/one/few-shot prompting for dialog summarization with `google/flan-t5-base`.
- Lab 2: `Lab_2_fine_tune_generative_ai_model (1).ipynb`
  - Full fine-tuning, PEFT (LoRA), and ROUGE evaluation on DialogSum.
- Lab 3: `Lab_3_fine_tune_model_to_detoxify_summaries.ipynb`
  - PPO fine-tuning with TRL and a toxicity reward model to reduce toxic generations.

## Notes
- The notebooks pin specific package versions for reproducibility.
- If running on SageMaker, follow the notebook prompts; they verify instance specs and include all setup cells.
- You may need GPU for faster training; CPU-only will be slow.

## License
Educational use. Review individual dataset/model licenses (Hugging Face) before redistribution or production use.
