# Project Contributions

This document records individual contributions to the COMP8420 Use Case 3
group submission. Module folders are named by **capability**, not by contributor.

| Contributor | Module | Primary responsibilities | Key artifacts |
| --- | --- | --- | --- |
| Yash | [`modules/dataset/`](../modules/dataset/) | 5k `PaperRecord` export, validation, EDA, Semantic Scholar enrichment | `modules/dataset/data/processed/dev_5k_enriched.jsonl`, `modules/dataset/results/data_validation/s2_enrichment_summary.md` |
| Bank | [`modules/retrieval/`](../modules/retrieval/) | TF-IDF, BM25, embeddings, hybrid RAG, APA citations, evaluation | `modules/retrieval/results/retrieval/` |
| Mulkraj | [`modules/llm/`](../modules/llm/) | Local LLM, prompts, paper-aware synthesis, LoRA, model comparison | `modules/llm/models/releases/`, `modules/llm/results/model_comparison/` |
| Nadiyah | [`modules/pdf_nlp/`](../modules/pdf_nlp/) | SciER DistilBERT fine-tuning, KeyBERT, PDF-NLP pipeline | `modules/pdf_nlp/paper_analysis.py`, `modules/pdf_nlp/results/pdf_nlp/` |
| Sidharth | [`integration/`](../integration/) | CLI/API, Vite frontend, structured traces, orchestration | `integration/app/pipeline.py`, `integration/frontend/`, `integration/results/traces/` |

## Running the full application

```bash
pip install -r requirements.txt
python setup_assets.py
ollama pull qwen3:8b
scripts/rpa run --query "your topic"
scripts/rpa analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa web
```

Download test PDFs first — see [`tests/README.md`](../tests/README.md).

## Module-only development

Each module can be tested independently. See `modules/<name>/README.md`.
