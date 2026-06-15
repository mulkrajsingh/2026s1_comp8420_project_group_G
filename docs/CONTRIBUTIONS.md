# Project contributions

Module folders are named by **capability**, not by contributor. This file records
who built each part of the COMP8420 Use Case 3 system for the report and submission.

Renamed from team workstreams? See [`WORKSTREAM_PATH_MAP.md`](WORKSTREAM_PATH_MAP.md).

| Contributor | Module | Primary responsibilities | Key artifacts |
| --- | --- | --- | --- |
| Yash | [`modules/dataset/`](../modules/dataset/) | 5k `PaperRecord` export, validation, EDA, S2 enrichment | `data/processed/dev_5k_enriched.jsonl`, `results/data_validation/s2_enrichment_summary.md` |
| Bank | [`modules/retrieval/`](../modules/retrieval/) | TF-IDF, BM25, embeddings, hybrid RAG, APA, eval | `outputs/recommendations.json`, `results/retrieval/` |
| Mulkraj | [`modules/llm/`](../modules/llm/) | Local LLM, prompts, paper-aware synthesis, LoRA, model comparison | `models/releases/`, `results/model_comparison/` |
| Nadiyah | [`modules/pdf_nlp/`](../modules/pdf_nlp/) | SciER DistilBERT fine-tuning, KeyBERT, PDF-NLP pipeline | `paper_analysis.py`, `results/pdf_nlp/` |
| Sidharth | [`integration/`](../integration/) | CLI/API/jobs, Vite frontend, structured traces, orchestration | `analyze-paper` integration, `frontend/`, `results/traces/` |

## How to run the full app (integration only)

```bash
pip install -r requirements.txt
scripts/rpa run --query "your topic"
scripts/rpa analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa web
```

Or from repo root: `scripts/rpa run --query "..."`

## Module-only development

Each module can be tested alone; see `modules/<name>/README.md`.
