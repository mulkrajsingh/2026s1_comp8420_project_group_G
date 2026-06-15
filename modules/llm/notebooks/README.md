# Notebooks

Runnable notebooks for the LLM module.

| Notebook | Purpose |
| --- | --- |
| [`model_comparison.ipynb`](model_comparison.ipynb) | Executable view of retained Ollama base-vs-LoRA and citation-guard results |
| [`train_lora_adapter.ipynb`](train_lora_adapter.ipynb) | Full-train-split QLoRA on Qwen3 8B; splits train/val/test in-notebook, eval metrics, resumable Drive checkpoints, adapter zip download |

Use the training JSONL delivered via `setup_assets.py` or the public Google Drive
copy configured in `train_lora_adapter.ipynb`. On Colab, the notebook downloads it
to `/content/train_data/research_lora_train.jsonl`; on local Jupyter, it downloads
to `data/processed/final_dataset/research_lora_train.jsonl`.

Colab training mounts Google Drive and uses the stable run folder
`research_lora_adapter/checkpoints/qwen3_8b_lora_v2/`. A complete checkpoint is
mirrored every 50 optimizer steps; a restarted session resumes from the highest
valid checkpoint after verifying the dataset hash.

The retained June 9, 2026 run metrics are in
`../results/model_comparison/lora_training_metrics_20260609.json`. The adapter
archive is delivered to `../models/releases/` via `setup_assets.py`.
