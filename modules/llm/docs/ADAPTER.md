# LoRA Adapter Training and Deployment

## Overview

The team fine-tuned Qwen3-8B with QLoRA on a merged instruction dataset. Training
data and the adapter archive are delivered via `setup_assets.py`; they are not
present in a fresh clone until setup runs.

## Workflow

1. **Build training data:** [`lora_dataset/README.md`](../lora_dataset/README.md) —
   `python -m modules.llm.lora_dataset.create_dataset`
2. **Train on GPU:** [`notebooks/train_lora_adapter.ipynb`](../notebooks/train_lora_adapter.ipynb)
   (Colab or local Jupyter; checkpoints, optional Drive backup, zip download)
3. **Evaluate:** `training/evaluate.py`, `training/evaluate_faithfulness.py`
4. **Deploy locally:** extract the trained PEFT adapter to
   `models/adapters/research_lora_adapter/`, pull `ollama pull qwen3:8b`, then
   run the root-relative build command documented in
   [`RUNTIME.md`](RUNTIME.md). The default path converts only the adapter and
   uses Ollama `FROM qwen3:8b` plus `ADAPTER`.

Default train outputs: `data/processed/final_dataset/research_lora_train.jsonl`
and `research_lora_train.zip` (upload the zip to Colab via Google Drive).

Training data can also be installed directly:

```text
python setup_assets.py   # from repository root
```

## Retained training run

The June 9, 2026 Colab run metrics and provenance are recorded in
`results/model_comparison/lora_training_run_20260609.md`. The adapter archive
is delivered to `models/releases/` via `setup_assets.py`.

For local deployment, extract the adapter to `models/adapters/research_lora_adapter/`,
then run the Ollama build script documented in [`RUNTIME.md`](RUNTIME.md).

## Dataset pipeline summary

From the repository root, create `.env` using `.env.example` as the template,
set `KAGGLE_API_TOKEN`, then run:

```text
pip install kaggle datasets huggingface_hub
python -m modules.llm.lora_dataset.create_dataset
```

Outputs: hybrid open JSONL (~14k rows), Kaggle raw snapshot (~5 GB), random 3k
corpus, 3,000 project RAG rows (500×6 tasks), and a local merged train file
(**16,998** rows with the current default seeds).

Useful flags: `--skip-download`, `--skip-hybrid`, `--rebuild-corpus`, `--no-seeds`.

QLoRA training is optional and targets Colab or a Linux CUDA environment. It is
not required for the cross-platform runtime or checker tests.
