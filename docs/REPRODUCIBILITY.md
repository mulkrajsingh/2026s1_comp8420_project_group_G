# Reproducibility and Evidence Map

This document maps each project module to what was built, where the evidence
lives in this repository, and how to reproduce the work. Evaluation
limitations are noted where they apply.

## Quick setup

```bash
pip install -r requirements.txt
python setup_assets.py              # runtime models + training data (~1.4 GB)
python setup_assets.py --optional   # + raw arXiv snapshot + E2E logs (~4.9 GB)
ollama pull qwen3:8b
```

Fill Google Drive file IDs in [`setup_assets.py`](../setup_assets.py) before
running. Check status with `python setup_assets.py --check`.

## Asset availability

| Category | Examples | Status |
| --- | --- | --- |
| Committed | JSONL corpora, eval CSV/MD, `tests/papers/artifacts/`, `integration/results/traces/` | In repository |
| Downloaded | PDF-NLP models, retrieval index, LoRA training data | `setup_assets.py` |
| Generated on demand | Retrieval chart PNGs, adapter zip after training | Run scripts/notebooks |
| Local only | Test PDFs, raw arXiv snapshot, session logs | See sections below |

---

## Technique evidence summary

### Basic techniques

| Technique | Primary evidence |
| --- | --- |
| Corpus preprocessing and validation | `modules/dataset/data/processed/dev_5k.jsonl`, `modules/dataset/results/data_validation/paperrecord_validation.json` |
| Semantic Scholar enrichment | `modules/dataset/data/processed/dev_5k_enriched.jsonl`, `modules/dataset/results/data_validation/s2_enrichment_summary.{json,md}` |
| Dataset EDA | `modules/dataset/results/eda/`, notebooks `01_data_preprocessing_2.ipynb`, `03_eda_1.ipynb` |
| TF-IDF / BM25 | `modules/retrieval/app/retrieval/tfidf_bm25.py`, `modules/retrieval/results/retrieval/` |
| Rule-based PDF parsing | `modules/pdf_nlp/pdf_parser.py`, `modules/pdf_nlp/tests/` |
| POS / NER / keyphrases | `modules/pdf_nlp/paper_analysis.py`, `modules/pdf_nlp/results/pdf_nlp/` |
| Traditional classifier | `modules/dataset/results/classification/` |
| Extractive summarization | TextRank in `modules/pdf_nlp/paper_analysis.py`; five-paper eval in `modules/pdf_nlp/results/pdf_nlp/` |

### Advanced techniques

| Technique | Primary evidence | Qualification |
| --- | --- | --- |
| RAG and APA recommendations | `modules/retrieval/app/rag_pack.py`, `modules/retrieval/app/citation.py`, `integration/results/demo/` | Curated demo artifacts |
| 5,000-paper retrieval evaluation | `modules/retrieval/results/retrieval/retrieval_comparison_*.csv` | Five gold queries only |
| Hybrid retrieval (RRF) | `modules/retrieval/app/retrieval/hybrid_ranker.py`, `modules/retrieval/results/retrieval/retrieval_eval_summary.md` | TF-IDF leads on aggregate metrics |
| Prompt engineering / ReAct | `modules/llm/app/prompt_library.py`, `modules/llm/results/prompt_comparison/` | Regenerate with Ollama for measured rows |
| Local Qwen3 runtime | `modules/llm/app/runtime.py`, `integration/app/providers/live_providers.py` | Requires running Ollama |
| LoRA fine-tuning | `modules/llm/notebooks/train_lora_adapter.ipynb`, `modules/llm/results/model_comparison/` | Training data via `setup_assets.py` |
| Query understanding | `modules/llm/app/query_understanding.py`, integration chat routing tests | Local MiniLM + TinyBERT models |

---

## Test PDF fixtures

PDF files are **not committed** to the repository (size and licensing). Download
them from arXiv into the paths expected by `tests/harness/paths.py`:

| Paper | Local path | Source |
| --- | --- | --- |
| DrQ-v2 | `tests/papers/drq_v2/2107.09645v1.pdf` | https://arxiv.org/pdf/2107.09645v1 |
| SIGA | `tests/papers/siga/SIGA_Self_Evolving_Coding_Agent_Adapters_for_Scientific_Simulation.pdf` | https://arxiv.org/pdf/2606.09774v1 |
| Transformer | `tests/papers/transformer/1706.03762v7.pdf` | https://arxiv.org/pdf/1706.03762v7 |
| BERT | `tests/papers/bert/1810.04805v2.pdf` | https://arxiv.org/pdf/1810.04805v2 |
| RAG | `tests/papers/rag/2005.11401v4.pdf` | https://arxiv.org/pdf/2005.11401v4 |

Committed reference outputs under `tests/papers/artifacts/` allow offline
inspection of parse, retrieval, and synthesis results without downloading PDFs.

Example download (DrQ-v2):

```bash
mkdir -p tests/papers/drq_v2
curl -L -o tests/papers/drq_v2/2107.09645v1.pdf \
  https://arxiv.org/pdf/2107.09645v1
```

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
| EDA policy and limitations | `modules/dataset/results/eda/` |
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
| Parser + enrichment source | `modules/pdf_nlp/pdf_parser.py`, `modules/pdf_nlp/paper_analysis.py` |
| Model manifest (checksums) | `modules/pdf_nlp/models/manifest.json` |
| Five-paper eval report | `modules/pdf_nlp/results/pdf_nlp/` |
| Historical comparison | `modules/pdf_nlp/results/historical_comparison/` |
| Unit tests | `modules/pdf_nlp/tests/` |

**Reproduce:**

```bash
python setup_assets.py   # downloads modules/pdf_nlp/models/runtime/

cd modules/pdf_nlp
python -m app.cli model-assets
# After downloading the test PDF (see above):
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
| Benchmark results (CSV/MD) | `modules/retrieval/results/retrieval/` |
| Eval notebook | `modules/retrieval/notebooks/03_rag_recommendation_evaluation.ipynb` |
| Unit tests | `modules/retrieval/tests/` |

Chart PNGs under `modules/retrieval/results/retrieval/` are not committed.
Regenerate with:

```bash
cd modules/retrieval
python scripts/regenerate_eval_charts.py
```

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
| Runtime + synthesis | `modules/llm/app/runtime.py`, `modules/llm/app/synthesis.py` |
| Prompt library | `modules/llm/app/prompt_library.py` |
| Fixed eval prompts | `modules/llm/data/eval/fixed_prompts*.jsonl` |
| Training manifests | `modules/llm/data/processed/*_manifest.md` |
| Training data | `modules/llm/data/processed/` after `setup_assets.py` |
| Adapter release | `modules/llm/models/releases/` after `setup_assets.py` |
| Training notebook | `modules/llm/notebooks/train_lora_adapter.ipynb` |
| Comparison notebook | `modules/llm/notebooks/model_comparison.ipynb` |
| LoRA training metrics | `modules/llm/results/model_comparison/` |
| Prompt comparison | `modules/llm/results/prompt_comparison/` |

Runtime synthesis outputs are written to `integration/outputs/` during
end-to-end runs, not under `modules/llm/outputs/`.

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
| Pipeline + contracts | `integration/app/pipeline.py`, `integration/app/contracts.py` |
| Live providers | `integration/app/providers/live_providers.py` |
| Session schema | `docs/OBSERVABILITY.md` |
| Curated Ollama traces | `integration/results/traces/` |
| Demo artifacts | `integration/results/demo/` |
| Acceptance sessions | `integration/data/sessions/20260614-*/` |
| E2E demo notebook | `integration/05_end_to_end_demo.ipynb` |
| Integration tests | `integration/tests/` |

**Reproduce:**

```bash
scripts/rpa web
# After downloading the test PDF:
scripts/rpa analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa run --query "transformer attention mechanisms"
cd integration && python -m app.cli session-inspect
```

---

## System tests

| Evidence | Path |
|---|---|
| Test PDF fixtures | Download to `tests/papers/` (see above) |
| Reference artifacts | `tests/papers/artifacts/` |
| E2E harness | `tests/harness/` |
| E2E test suite | `tests/e2e/` |
| Test documentation | `tests/TEST_CASES.md`, `tests/README.md` |
| Raw E2E logs (optional) | `tests/logs/` after `setup_assets.py --optional` |

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

Evaluation **results** (metrics tables, reports) are committed in the repository.
Training **inputs** and runtime **models** are downloaded via `setup_assets.py`.
