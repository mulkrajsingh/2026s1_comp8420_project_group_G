# COMP8420 Major Project — Research Paper Assistant (Use Case 3)

This repository implements a local research paper assistant for COMP8420. It
parses PDF uploads, enriches them with NLP features, retrieves related papers
from a curated arXiv corpus, and answers questions with a local Ollama model.

Evidence paths and reproduction commands are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The full documentation
index is in [`docs/README.md`](docs/README.md).

## How to run

You need Python 3.11 or newer, Node.js 18 or newer, pnpm, Ollama installed and
running, and about 15 GB of free disk space for models and the base LLM. The
optional raw arXiv snapshot requires additional space.

Install dependencies:

```bash
pip install -r requirements.txt
```

Download runtime assets from Google Drive. Fill in the file IDs at the top of
`setup_assets.py` before the first run. About 1.4 GB is required for the core
bundles.

```bash
python setup_assets.py
python setup_assets.py --check
```

Optional bundles include the raw arXiv snapshot and end-to-end test logs:

```bash
python setup_assets.py --optional
```

Pull the base language model once (about 5 GB):

```bash
ollama pull qwen3:8b
```

The fine-tuned LoRA adapter can be built from training data delivered by
`setup_assets.py`:

```bash
python modules/llm/scripts/build_ollama_research_lora_model.py
```

Check that PDF-NLP models are present:

```bash
cd modules/pdf_nlp && python -m app.cli model-assets && cd ../..
```

Install Node.js and pnpm before building the web UI. Vite 5 needs Node 18 or
newer:

```bash
corepack enable
corepack prepare pnpm@latest --activate
cd integration/frontend
pnpm install && pnpm run build
```

Start the web app from the repository root and open http://127.0.0.1:8000 when
the server is ready. The launcher builds the frontend automatically when
`dist/` is missing or stale.

```bash
scripts/rpa web
```

CLI examples use the same entry point. Download test PDFs first (see
[`tests/README.md`](tests/README.md)) or use any local research PDF:

```bash
scripts/rpa analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa run --query "transformer attention mechanisms"
scripts/rpa search-topic "retrieval augmented generation"
```

Run the test suite from the repository root:

```bash
pytest modules/dataset/tests modules/pdf_nlp/tests \
       modules/retrieval/tests modules/llm/tests \
       integration/tests tests/
```

The balanced corpus used for production retrieval is
`modules/dataset/data/processed/dev_5k_balanced.jsonl` (the enriched variant
`dev_5k_enriched.jsonl` adds Semantic Scholar fields used during dataset
analysis). The LoRA adapter archive
is delivered via `setup_assets.py` to
`modules/llm/models/releases/` (not present in a fresh clone until setup runs).

Google Drive bundles include `pdf_nlp_models.zip` (~1.2 GB, required),
`retrieval_index.zip` (~16 MB, required), `qwen3_gguf.zip` (~29 MB, required),
`lora_training_data.zip` (~31 MB, required), `arxiv_raw_snapshot.zip`
(~4.9 GB, optional), and `e2e_test_logs.zip` (~29 MB, optional). Evaluation
tables, notebooks, and small fixtures are committed. Large runtime models and
training inputs are downloaded via `setup_assets.py`.

Web routing, session logging, and API details are in
[`integration/README.md`](integration/README.md). System layout is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What each part does

### Dataset (`modules/dataset/`)

The dataset module builds the arXiv paper records that retrieval and training
consume. Notebook `modules/dataset/01_data_preprocessing_2.ipynb` filters the
Kaggle arXiv metadata dump into `PaperRecord` JSONL files. Notebook
`modules/dataset/03_eda_1.ipynb` explores category balance, abstract length, and
vocabulary. Script `modules/dataset/scripts/build_balanced_corpus.py` creates a
balanced subset for production retrieval. Script
`modules/dataset/scripts/evaluate_domain_classifier.py` trains a TF-IDF plus
logistic regression baseline. The balanced file
`modules/dataset/data/processed/dev_5k_balanced.jsonl` is the default corpus
path for production retrieval; the enriched variant
`dev_5k_enriched.jsonl` adds Semantic Scholar fields used during analysis.

### PDF-NLP (`modules/pdf_nlp/`)

The PDF-NLP module turns an uploaded file into a structured `ParsedPaper`.
`modules/pdf_nlp/pdf_parser.py` extracts sections and metadata with
deterministic rules. `modules/pdf_nlp/paper_analysis.py` adds spaCy POS tags,
SciER entities, KeyBERT keyphrases, and an extractive summary. The CLI under
`modules/pdf_nlp/app/cli.py` supports parse, enrich, evaluate, and model
asset checks. Runtime models are installed through `setup_assets.py`.

### Retrieval (`modules/retrieval/`)

The retrieval module ranks papers from the enriched corpus and builds evidence
packs for the LLM. `modules/retrieval/app/retrieval/tfidf_bm25.py` implements
sparse baselines. `embeddings.py` and `section_aware.py` add dense and
section-aware options. `hybrid_ranker.py` combines sparse and dense scores with
reciprocal rank fusion. `rag_pack.py` formats top passages for synthesis.
Notebook `modules/retrieval/notebooks/03_rag_recommendation_evaluation.ipynb`
holds the full retrieval evaluation workflow.

### LLM (`modules/llm/`)

The LLM module classifies queries, calls Ollama, and optionally uses a
fine-tuned adapter. `modules/llm/app/query_understanding.py` routes chat, paper
recommendations, and research questions. `runtime.py` and `ollama_transport.py`
send generation requests to Ollama. `synthesis.py` and `react_loop.py` ground
answers in retrieved evidence. `verification.py` and `faithfulness.py` check
citations against source IDs. LoRA training is documented in
[`modules/llm/docs/ADAPTER.md`](modules/llm/docs/ADAPTER.md).

### Integration (`integration/`)

The integration layer wires the modules into one CLI, HTTP API, and web UI.
`integration/app/pipeline.py` routes PDF, topic, chat, and recommendation
flows. `service.py` and `providers/live_providers.py` call PDF-NLP, retrieval,
and LLM subprocesses. `session_log.py` writes structured JSONL session files.
Notebook `integration/05_end_to_end_demo.ipynb` loads committed production
artifacts without rerunning models.

[`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · [`docs/README.md`](docs/README.md)
