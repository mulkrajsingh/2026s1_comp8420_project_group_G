# COMP8420 Major Project — Research Paper Assistant (Use Case 3)

A local-first research paper assistant that parses PDFs, enriches them with
NLP analysis, retrieves related papers from a curated corpus, and synthesises
answers with Ollama.

**Evidence index:** [`outputs/report_presentation_evidence_index.md`](outputs/report_presentation_evidence_index.md)

**Reproducibility map:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — module-by-module
evidence paths and reproduction commands.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/download) installed and running
- ~15 GB free disk space (models + base LLM; more if downloading optional raw corpus)

## Setup

```bash
# 1. Install Python packages
pip install -r requirements.txt

# 2. Download required assets from Google Drive (~1.4 GB total)
python setup_assets.py

# 3. Optional: raw arXiv snapshot + E2E test logs (~4.9 GB extra)
python setup_assets.py --optional

# 4. Pull the base LLM (~5 GB, one-time)
ollama pull qwen3:8b

# 5. Build the fine-tuned LoRA model for Ollama (optional — improves research answers)
python modules/llm/scripts/build_ollama_research_lora_model.py

# 6. Validate PDF-NLP runtime models
cd modules/pdf_nlp && python -m app.cli model-assets && cd ../..
```

Fill in the Google Drive file IDs in `setup_assets.py` before running step 2.
See `python setup_assets.py --check` to verify what is installed.

### Google Drive bundles

| Bundle | Size | Required |
|---|---|---|
| `pdf_nlp_models.zip` | ~1.2 GB | Yes |
| `retrieval_index.zip` | ~16 MB | Yes |
| `qwen3_gguf.zip` | ~29 MB | Yes |
| `lora_training_data.zip` | ~31 MB | Yes |
| `arxiv_raw_snapshot.zip` | ~4.9 GB | Optional (`--optional`) |
| `e2e_test_logs.zip` | ~29 MB | Optional (`--optional`) |

Eval results, notebooks, and test fixtures are already in the repo. Training
inputs and runtime models are downloaded via `setup_assets.py`.

## Run

```bash
# Web application (recommended)
scripts/rpa web
# → open http://127.0.0.1:8000

# CLI — analyse a PDF
scripts/rpa analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf

# CLI — topic search
scripts/rpa run --query "transformer attention mechanisms"
```

See [`integration/README.md`](integration/README.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

The production PDF path uses the PDF-NLP parser, retrieval for related papers
(unless `--no-related-papers` is supplied), and Ollama paper-aware synthesis.

The web launcher installs/builds the Vite frontend when needed and serves the UI
and `/api/*` from the same FastAPI process.

The active trained adapter is
`modules/llm/models/releases/research_lora_adapter_20260609_114209.zip`.
The canonical enriched corpus is
`modules/dataset/data/processed/dev_5k_enriched.jsonl`.

## Run tests

```bash
pytest modules/dataset/tests modules/pdf_nlp/tests \
       modules/retrieval/tests modules/llm/tests \
       integration/tests tests/
```

## Rebuild the frontend (if needed)

```bash
cd integration/frontend
pnpm install && pnpm run build
```

## Modules

| Module | Path |
| --- | --- |
| Dataset | [`modules/dataset/`](modules/dataset/) |
| PDF-NLP | [`modules/pdf_nlp/`](modules/pdf_nlp/) |
| Retrieval | [`modules/retrieval/`](modules/retrieval/) |
| LLM | [`modules/llm/`](modules/llm/) |
| Integration | [`integration/`](integration/) |

[`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) · [`docs/WORKSTREAM_PATH_MAP.md`](docs/WORKSTREAM_PATH_MAP.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
