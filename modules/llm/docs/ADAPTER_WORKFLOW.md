# Adapter workflow (redirect)

1. **Build train data:** [`lora_dataset/README.md`](../lora_dataset/README.md) — `python -m lora_dataset.create_dataset`
2. **Train on GPU:** [`notebooks/train_lora_adapter.ipynb`](../notebooks/train_lora_adapter.ipynb) (Colab or local Jupyter; checkpoints, Drive backup, zip download)
3. **Evaluate:** `training/evaluate.py`, `training/evaluate_faithfulness.py`
4. **Deploy locally:** extract the trained PEFT adapter to `models/adapters/research_lora_adapter/`, pull `ollama pull qwen3:8b`, then run `python scripts/build_ollama_research_lora_model.py --llama-cpp /path/to/llama.cpp` from the workstream root. The default path converts only the adapter and uses Ollama `FROM qwen3:8b` plus `ADAPTER`.

Default train outputs: `data/processed/final_dataset/research_lora_train.jsonl` and `research_lora_train.zip` (upload the zip to Colab via Google Drive)

Colab Drive fallback path: `MyDrive/colab_notebooks/nlp/research_lora_adapter/` (adapters zip and `checkpoints/{RUN_ID}/`).
