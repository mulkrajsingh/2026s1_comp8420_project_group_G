# Notebooks

Runnable notebooks for the research LLM workstream.

| Notebook | Purpose |
| --- | --- |
| [`model_comparison.ipynb`](model_comparison.ipynb) | Executable view of retained real Ollama base-vs-LoRA and citation-guard results |
| [`train_lora_adapter.ipynb`](train_lora_adapter.ipynb) | Full-train-split QLoRA on Qwen3 8B; splits train/val/test in-notebook, eval metrics, resumable Drive checkpoints, adapter zip download |

Use the public Google Drive training JSONL configured in `train_lora_adapter.ipynb`. On Colab, the notebook downloads it directly to `/content/train_data/research_lora_train.jsonl`; on local Jupyter, it downloads to `data/processed/final_dataset/research_lora_train.jsonl`. The default run requires the complete 17,298-row JSONL, keeps an 80/10/10 split, and trains one full epoch over the training split.

Colab training mounts Google Drive and uses the stable run folder `research_lora_adapter/checkpoints/qwen3_8b_lora_v2/`. A complete checkpoint is mirrored atomically every 50 optimizer steps, the latest three Drive checkpoints are retained, and a restarted session resumes from the numerically highest valid checkpoint after verifying the dataset hash and training metadata. Successful completion writes a marker that prevents accidental retraining; set `FORCE_RETRAIN = True` in the notebook to deliberately start a new run.

The canonical notebook contains the concise retained output from the completed
9 June 2026 run. The released adapter is
`../models/releases/research_lora_adapter_20260609_114209.zip`; the machine-readable
metrics are in
`../results/model_comparison/lora_training_metrics_20260609.json`. Older copied
training notebooks and the June 3 adapter archive were removed to avoid an
ambiguous deployment source.
