# Architecture — Research Paper Assistant

Use Case 3: local research-paper analysis and recommendation.

## Layout

```
integration/          # Single entry — CLI, API, Vite frontend, service, results
modules/
  dataset/            # PaperRecord corpus, EDA
  retrieval/          # RAG, hybrid rank, metrics
  llm/                # Prompts, Ollama, LoRA synthesis
  pdf_nlp/            # Live parse, POS, NER, KeyBERT, TextRank, structural checks
docs/CONTRIBUTIONS.md
team_plans/           # Historical stage prompts (paths may say old folder names)
```

## Using the app (integration only)

```bash
cd integration
python -m app.cli run --query "topic text"
```

`run` orchestrates:

1. Canonical `modules/dataset/data/processed/dev_5k_balanced.jsonl`
2. Subprocess → `modules/retrieval` (`recommend-topic`)
3. Subprocess → `modules/llm` (`synthesize` with Ollama)
4. Writes runtime artifacts under `integration/outputs/` and
   `integration/data/sessions/`

`analyze-pdf` orchestrates:

1. Subprocess → `modules/pdf_nlp` (`analyze-paper`) producing an enriched `ParsedPaper`
2. Optional subprocess → `modules/retrieval` using the parsed title/abstract query
3. Subprocess → `modules/llm` with the paper and deterministic `analysis` evidence
4. Peer-review synthesis and a single `AnalysisResult.paper_analysis`

Use `--no-related-papers` for the paper-only path. Production commands reject
executed mock providers; `run-demo` intentionally remains the deterministic mock path.

The canonical browser UI is the Vite project under `integration/frontend/`.
Development uses its `/api` proxy; a production build is served by FastAPI from
the same origin. The UI uses one timestamped session per analysis and one
synchronous request per action, restores its tab-scoped transcript, and shows compact
status only. Optional queued endpoints remain available for automation, while all
routes write structured JSONL sessions outside the conversation.

Repo-root wrapper: `scripts/rpa <command>`

The combined browser application starts with `scripts/rpa web`. It builds stale
or missing Vite assets before serving the UI and API from
`http://127.0.0.1:8000`.

## Developing a module

Each module has its own `app/` package:

```bash
cd modules/retrieval
python -m app.cli recommend-topic --papers ../dataset/data/processed/dev_5k.jsonl ...
```

Integration does **not** duplicate module implementations. `app/service.py`
constructs concrete providers for one request, then `app/pipeline.py` passes
shared contracts between module CLIs.

Runtime outputs, session logs, raw datasets, and caches are ignored. Curated,
sanitized report/video traces live under `integration/results/`.

Structured event schema and session navigation:
[`OBSERVABILITY.md`](OBSERVABILITY.md).

## Shared contracts

JSON schemas: [`team_plans/06_integration_contract.md`](../team_plans/06_integration_contract.md)

## Claim status

Label outputs **implemented**, **mixed**, or **pending** per [`guardrail.md`](../guardrail.md).
