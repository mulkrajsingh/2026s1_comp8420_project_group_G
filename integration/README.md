# Integration — Research Paper Assistant

Entry point for running the full COMP8420 Use Case 3 system.

Contributions: [`docs/CONTRIBUTIONS.md`](../docs/CONTRIBUTIONS.md)  
Architecture: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)

## Quick start

From the repository root:

```text
pip install -r requirements.txt
python setup_assets.py
ollama pull qwen3:8b

python rpa.py run --query "retrieval augmented generation for scientific literature"
python rpa.py analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf
python rpa.py analyze-pdf tests/papers/drq_v2/2107.09645v1.pdf --no-related-papers
python rpa.py search-topic "your topic keywords here"
python rpa.py session-inspect --component pdf_nlp
python rpa.py integration-status
```

The five checker fixtures are committed under `tests/papers/`.

## CLI commands

| Command | Description |
| --- | --- |
| `run --query "..."` | Topic search with retrieval and LLM synthesis |
| `analyze-pdf <path>` | Parse PDF, optional related-paper retrieval, synthesis |
| `analyze-pdf <path> --no-related-papers` | Paper-only summary without retrieval |
| `search-topic "..."` | Retrieval and recommendations only |
| `session-inspect` | Inspect structured session logs |
| `integration-status` | Report provider wiring and module health |
| `web` | Build frontend (if needed) and serve UI + API |

All commands are available via `python rpa.py <command>` from the repository
root. Module developers can also run `python -m app.cli <command>` from
`integration/`.

## Web app

```text
python rpa.py web
# open http://127.0.0.1:8000
```

Web startup requires local Ollama and load-warms `qwen3:8b` plus the pinned
query-understanding models before accepting requests. `/api/models` exposes the
installed project base and adapter tags.

Requests are serialized through one model lane: when the selected tag changes,
the previous project model is unloaded before the new one is warmed. Override
the startup target with `COMP8420_OLLAMA_MODEL` and `COMP8420_OLLAMA_HOST`;
`/api/health` reports the residency state.

Use `python rpa.py web --rebuild` to force a new production build. For frontend
development, keep the backend running with `python rpa.py web --reload`, then
run `pnpm --dir integration/frontend dev`; Vite proxies `/api` requests to
`http://127.0.0.1:8000`.

## Routing behaviour

- Paper-only summaries use the LLM module's `summarize` contract without retrieval.
- PDF analysis parses the upload first, then uses its title (abstract fallback) as
  the related-paper retrieval query.
- The live parser adapter calls PDF-NLP `analyze-paper`, so `ParsedPaper` reaches
  the pipeline with deterministic POS, entities, keyphrases, extractive summary,
  structural checks, timings, and provenance.
- Retrieved evidence and related-paper analysis use `synthesize`.
- Conversational text detected by query understanding bypasses document retrieval
  but still uses the selected local LLM; research text uses retrieval plus LLM synthesis.
- Text style defaults to `auto`: MiniLM cosine similarity classifies the query
  before providers are configured, and fields below `0.70` are reranked by local
  TinyBERT.
- The web UI creates one timestamped session per analysis. Every text/PDF turn
  appends user-visible communication, routing, component activity, output metadata,
  and failures to the same JSONL file.
- Refresh restores the active tab-scoped transcript; **New analysis** completes
  that session and creates another.
- Optional async PDF/topic jobs remain serialized through one local worker. See
  [`docs/OBSERVABILITY.md`](../docs/OBSERVABILITY.md).

Full system tests: `python tests/run_system_tests.py --skip-ollama` from the
repository root. Use `--require-ollama` for live-model E2E coverage.

## Local latency expectations (`qwen3:8b`)

On a typical local setup (~20 tokens/s generation):

| Scenario | Approximate wall time |
| --- | ---: |
| Direct chat | ~15 s |
| Topic RAG recommendation | ~60–90 s |
| PDF summary (paper-only, no peer review) | ~3–5 min |
| PDF with peer review enabled | add ~60–120 s |

Only `qwen3:8b` and `qwen3-research-lora:latest` are supported project models.

## Curated evidence

Demonstration traces and redacted session logs are stored under:

- `integration/results/traces/`
- `integration/results/demo/`
- `integration/data/sessions/` (committed acceptance sessions; see `data/sessions/README.md`)

## Module-only development

Each capability lives under `modules/` with its own CLI. See `modules/<name>/README.md`.

## More detail

- Provider wiring: [`INTEGRATION.md`](INTEGRATION.md)
- Frontend: [`frontend/README.md`](frontend/README.md)
- Reproducibility: [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md)
