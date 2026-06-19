# LoRA training dataset builder

Single entry point for the research LoRA collection: open academic datasets + ResearchQA hybrid, Kaggle arXiv corpus, project BM25 RAG instructions, and final merge.

## Prerequisites

From the repository root:

```text
pip install kaggle datasets huggingface_hub
```

### Kaggle credentials (required for download)

1. Create a token at [Kaggle settings](https://www.kaggle.com/settings) (API → Generate New Token).
2. Copy `.env.example` to `.env` at the **repository root**.
3. Add:

```env
KAGGLE_API_TOKEN=your_token_here
```

Legacy `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env` also work.

There is **no** synthetic corpus fallback. If credentials are missing when a download is needed, the CLI prints setup steps and exits with code 1.

## One command

```text
python -m modules.llm.lora_dataset.create_dataset
```

### Flags


| Flag               | Effect                                                      |
| ------------------ | ----------------------------------------------------------- |
| `--skip-hybrid`    | Reuse existing hybrid JSONL                                 |
| `--skip-download`  | Require `data/raw/arxiv-metadata-oai-snapshot.json` on disk |
| `--rebuild-corpus` | Delete and resample 3k PaperRecords (seed 42)               |
| `--no-seeds`       | Merge without 6 `local_fixed_prompt` rows                   |


Example (corpus + RAG + merge only):

```text
python -m modules.llm.lora_dataset.create_dataset --skip-download --skip-hybrid
```

## Pipeline steps


| Step                | Output                                                              | Notes                                                            |
| ------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 1. Open hybrid      | `data/processed/open_academic_researchqa_hybrid_lora_dataset.jsonl` | QASPER, SciTLDR, SciCite, PeerRead, ResearchQA; seed 42, 3k caps |
| 2. Kaggle download  | `data/raw/arxiv-metadata-oai-snapshot.json`                         | Skip if present                                                  |
| 3. Corpus           | `data/processed/project_arxiv_rag_corpus_3k.jsonl`                  | Random 3k reservoir, seed 42                                     |
| 4. RAG instructions | `data/processed/project_arxiv_rag_lora_instructions.jsonl`          | 500 queries × 6 tasks = 3000                                     |
| 5. Merge            | `data/processed/final_dataset/research_lora_train.jsonl` and `.zip` | Hybrid + RAG + 6 seeds (unless `--no-seeds`) |


## Sampling

- **Open sources:** up to 3000 rows per source (SciTLDR may be lower if the Hub split is smaller).
- **Corpus:** reservoir sample of 3000 papers from the Kaggle snapshot (not “first 3000”).
- **Project RAG:** 500 title/abstract-derived queries × 6 task types.
- **Seeds:** six fixed local prompts (`local_fixed_prompt`) anchoring task shapes in merge.

## Verification

```text
python -c "from pathlib import Path; p=Path('modules/llm/data/processed/final_dataset/research_lora_train.jsonl'); print('rows', sum(1 for _ in p.open()))"
```

Expected total: **16998** rows (13992 hybrid + 3000 project_arxiv_rag + 6 seeds). The merge step (step 5) writes a `research_lora_train_manifest.md` next to the JSONL with per-source counts; committed per-source manifests are under `data/processed/*_manifest.md`. Upload `final_dataset/research_lora_train.zip` to Colab via Google Drive.

## Troubleshooting

- **HF download errors:** set `HF_TOKEN` in `.env` if a gated dataset fails; retry with stable network.
- **`kaggle` not found:** `pip install kaggle` or use `python -m kaggle`.
- **Synthetic corpus refused:** delete `project_arxiv_rag_corpus_3k.jsonl` if it contains `sample_`* IDs and rebuild with Kaggle raw present.
- **Missing hybrid with `--skip-hybrid`:** run once without `--skip-hybrid` (downloads from Hugging Face).

## Module layout


| Module                                    | Role                         |
| ----------------------------------------- | ---------------------------- |
| `create_dataset.py`                       | CLI orchestrator             |
| `open_hybrid.py`                          | Open academic + ResearchQA   |
| `kaggle_download.py` / `kaggle_corpus.py` | Raw snapshot + 3k corpus     |
| `arxiv_rag.py`                            | BM25 RAG instruction rows    |
| `merge_train.py`                          | Final train JSONL + manifest |


Train after build:

- Colab or local GPU: `notebooks/train_lora_adapter.ipynb`

QLoRA training is optional and targets Colab or a Linux CUDA environment. The
dataset builder itself is cross-platform and is not needed for checker-facing
runtime or deterministic tests.
