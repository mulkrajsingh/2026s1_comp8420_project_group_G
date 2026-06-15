# Reproducibility and Evidence Map

This document maps each project module to what was built, where the evidence
lives in this repository, and how to reproduce the work.

For rubric-aligned claim status, see
[`outputs/report_presentation_evidence_index.md`](../outputs/report_presentation_evidence_index.md).

## Quick setup

```bash
pip install -r requirements.txt
python setup_assets.py              # runtime models + training data (~1.4 GB)
python setup_assets.py --optional   # + raw arXiv snapshot + E2E logs (~4.9 GB)
ollama pull qwen3:8b
```

Fill Google Drive file IDs in [`setup_assets.py`](../setup_assets.py) before
running. Check status with `python setup_assets.py --check`.

---

## Dataset module

**What we built:** arXiv corpus sampling, Semantic Scholar enrichment, EDA,
balanced 5k production corpus, domain classifier evaluation.

| Evidence | Path |
|---|---|
| Base corpus (5,000 papers) | `modules/dataset/data/processed/dev_5k.jsonl` |
| Enriched corpus | `modules/dataset/data/processed/dev_5k_enriched.jsonl` |
| Production retrieval corpus | `modules/dataset/data/processed/dev_5k_balanced.jsonl` |
| Validation reports | `modules/dataset/results/data_validation/` |
| EDA figures and policy | `modules/dataset/results/eda/` |
| Classifier metrics | `modules/dataset/results/classification/` |
| Preprocessing notebook | `modules/dataset/01_data_preprocessing_2.ipynb` |
| EDA notebook | `modules/dataset/03_eda_1.ipynb` |

**Reproduce:**

```bash
# Corpus already committed; optional full rebuild from raw snapshot:
python setup_assets.py --optional   # downloads modules/dataset/data/raw/

cd modules/dataset
python scripts/build_balanced_corpus.py --help
python scripts/evaluate_domain_classifier.py --help
```

---

## PDF-NLP module

**What we built:** rule-based PDF parser, deterministic enrichment (POS, NER,
keyphrases, TextRank summary, structural checks), five-paper evaluation.

| Evidence | Path |
|---|---|
| Parser + enrichment source | `modules/pdf_nlp/pdf_parser.py`, `paper_analysis.py` |
| Model manifest (checksums) | `modules/pdf_nlp/models/manifest.json` |
| Five-paper eval report | `modules/pdf_nlp/results/pdf_nlp/` |
| Historical comparison | `modules/pdf_nlp/results/historical_comparison/` |
| Unit tests | `modules/pdf_nlp/tests/` |

**Reproduce:**

```bash
python setup_assets.py   # downloads modules/pdf_nlp/models/runtime/

cd modules/pdf_nlp
python -m app.cli model-assets
python -m app.cli analyze-paper ../../tests/papers/drq_v2/2107.09645v1.pdf
pytest tests/
```

---

## Retrieval module

**What we built:** TF-IDF, BM25, dense embeddings, hybrid RRF, RAG evidence
pack export, dual retrieval evaluation on 5k corpus.

| Evidence | Path |
|---|---|
| Retrieval implementations | `modules/retrieval/app/retrieval/` |
| RAG pack builder | `modules/retrieval/app/rag_pack.py` |
| Eval query sets | `modules/retrieval/data/processed/eval_queries_*.json` |
| Benchmark results + charts | `modules/retrieval/results/retrieval/` |
| Eval notebook | `modules/retrieval/notebooks/03_rag_recommendation_evaluation.ipynb` |
| Unit tests | `modules/retrieval/tests/` |

**Reproduce:**

```bash
python setup_assets.py   # downloads retrieval_index/

cd modules/retrieval
python -m app.cli evaluate-retrieval \
  --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
  --out results/retrieval/ \
  --query-set all
pytest tests/
```

---

## LLM module

**What we built:** Ollama runtime, prompt library, query understanding, paper-aware
synthesis, LoRA dataset pipeline, QLoRA training, base-vs-LoRA comparison.

| Evidence | Path |
|---|---|
| Runtime + synthesis | `modules/llm/app/runtime.py`, `synthesis.py` |
| Prompt library | `modules/llm/app/prompt_library.py` |
| Fixed eval prompts | `modules/llm/data/eval/fixed_prompts*.jsonl` |
| Training manifests | `modules/llm/data/processed/*_manifest.md` |
| Training data (GDrive) | `modules/llm/data/processed/` after `setup_assets.py` |
| Adapter release | `modules/llm/models/releases/research_lora_adapter_20260609_114209.zip` |
| Training notebook | `modules/llm/notebooks/train_lora_adapter.ipynb` |
| Comparison notebook | `modules/llm/notebooks/model_comparison.ipynb` |
| LoRA training metrics | `modules/llm/results/model_comparison/` |
| Prompt comparison | `modules/llm/results/prompt_comparison/` |
| Dataset EDA | `modules/llm/results/lora_dataset_eda/` |

**Reproduce:**

```bash
python setup_assets.py   # downloads training JSONL + RAG packs

cd modules/llm
python -m lora_dataset.create_dataset --help
python scripts/build_ollama_research_lora_model.py
python -m app.cli compare-prompts --help
pytest tests/
```

---

## Integration module

**What we built:** unified CLI/API/web app, live provider wiring, session
observability, async jobs, end-to-end PDF and topic flows.

| Evidence | Path |
|---|---|
| Pipeline + contracts | `integration/app/pipeline.py`, `contracts.py` |
| Live providers | `integration/app/providers/live_providers.py` |
| Session schema | `docs/OBSERVABILITY.md` |
| Curated Ollama traces | `integration/results/traces/` |
| Demo screenshots | `integration/results/demo/` |
| Acceptance sessions | `integration/data/sessions/20260614-*/` |
| E2E demo notebook | `integration/05_end_to_end_demo.ipynb` |
| Integration tests | `integration/tests/` |

**Reproduce:**

```bash
scripts/rpa web
scripts/rpa analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa run --query "transformer attention mechanisms"
python -m app.cli session-inspect   # from integration/
```

---

## System tests

| Evidence | Path |
|---|---|
| Test PDF fixtures | `tests/papers/` (5 real PDFs) |
| Reference artifacts | `tests/papers/artifacts/` |
| E2E harness | `tests/harness/` |
| E2E test suite | `tests/e2e/` |
| Test documentation | `tests/TEST_CASES.md` |
| Raw E2E logs (optional, GDrive) | `tests/logs/` after `--optional` |

```bash
pytest modules/dataset/tests modules/pdf_nlp/tests \
       modules/retrieval/tests modules/llm/tests \
       integration/tests tests/
```

---

## Google Drive bundles

| Bundle | Size | Required | Installs to |
|---|---|---|---|
| `pdf_nlp_models.zip` | ~1.2 GB | Yes | `modules/pdf_nlp/models/runtime/` |
| `retrieval_index.zip` | ~16 MB | Yes | `modules/retrieval/data/processed/retrieval_index/` |
| `qwen3_gguf.zip` | ~29 MB | Yes | `modules/llm/models/ollama/qwen3-research-lora/` |
| `lora_training_data.zip` | ~31 MB | Yes | `modules/llm/data/processed/` |
| `arxiv_raw_snapshot.zip` | ~4.9 GB | Optional | `modules/dataset/data/raw/` |
| `e2e_test_logs.zip` | ~29 MB | Optional | `tests/logs/` |

Eval **results** (metrics, charts, reports) are committed in the repo.
Training **inputs** and runtime **models** are downloaded via `setup_assets.py`.
