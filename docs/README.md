# Documentation

This repository implements **COMP8420 Use Case 3**: a local research paper
assistant that parses PDF uploads, enriches them with NLP features, retrieves
related papers from a curated arXiv corpus, and answers questions with a local
Ollama model.

## Reading order

1. **[Architecture](ARCHITECTURE.md)** — system layout, data flow, and shared contracts
2. **[Reproducibility](REPRODUCIBILITY.md)** — evidence map and commands to reproduce each component
3. **[Contributions](CONTRIBUTIONS.md)** — individual responsibilities within the group submission
4. **[Observability](OBSERVABILITY.md)** — structured session logging for evaluation and demonstration

## Running the application

Operational instructions (CLI, API, web UI) are in
[`integration/README.md`](../integration/README.md). From the repository root:

```bash
pip install -r requirements.txt
python setup_assets.py
ollama pull qwen3:8b
scripts/rpa web
```

## Module documentation

Each capability module has its own README for standalone development:

| Module | Path | Responsibility |
| --- | --- | --- |
| Dataset | [`modules/dataset/README.md`](../modules/dataset/README.md) | Corpus export, enrichment, EDA |
| PDF-NLP | [`modules/pdf_nlp/README.md`](../modules/pdf_nlp/README.md) | PDF parsing and deterministic enrichment |
| Retrieval | [`modules/retrieval/README.md`](../modules/retrieval/README.md) | Hybrid ranking, RAG evidence packs |
| LLM | [`modules/llm/README.md`](../modules/llm/README.md) | Local generation, prompts, LoRA adapter |
| Integration | [`integration/README.md`](../integration/README.md) | Orchestration, API, and web frontend |
