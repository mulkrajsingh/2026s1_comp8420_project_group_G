# Integration — Research Paper Assistant (entry point)

**Start here** to run the full COMP8420 Use Case 3 system.

Contributions: [`docs/CONTRIBUTIONS.md`](../docs/CONTRIBUTIONS.md)  
Architecture: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md)

## Quick start

From the repository root:

```bash
pip install -r requirements.txt

scripts/rpa run --query "retrieval augmented generation for scientific literature"
scripts/rpa analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf --no-related-papers
scripts/rpa search-topic "your topic keywords here"
scripts/rpa session-inspect --component pdf_nlp
scripts/rpa integration-status
```

From repo root (wrapper):

```bash
scripts/rpa run --query "your topic"
```

## Web app

The canonical backend and frontend both live under `integration/`. From the
repository root, one command builds stale or missing frontend assets and serves
the Vite UI and FastAPI API from the same origin:

```bash
scripts/rpa web
# open http://127.0.0.1:8000
```

Web startup requires local Ollama and load-warms `qwen3:8b` plus the pinned
query-understanding models before accepting requests. `/api/models` exposes the
installed project base and adapter tags.
Requests are serialized through one model lane: when the selected tag changes,
the previous project model is unloaded before the new one is warmed, preventing
both 8B variants from remaining in memory. The active model is unloaded on a
clean shutdown. Override the startup target with `COMP8420_OLLAMA_MODEL` and
`COMP8420_OLLAMA_HOST`; `/api/health` reports the residency state.

Use `scripts/rpa web --rebuild` to force a new production build. For frontend
development, keep the backend running with `scripts/rpa web --reload`, then run
`pnpm dev` in `integration/frontend/`; Vite proxies `/api` requests to
`http://127.0.0.1:8000`.

## Current routing

- Paper-only summaries use the LLM module's `summarize` contract without retrieval.
- PDF analysis parses the upload first, then uses its title (abstract fallback) as
  the related-paper retrieval query.
- The live parser adapter calls PDF-NLP `analyze-paper`, so `ParsedPaper` reaches
  the pipeline with deterministic POS, entities, keyphrases, extractive summary,
  structural checks, timings, and provenance.
- PDF parsing, retrieval, and synthesis use live module adapters in production
  commands. Full system tests: `../tests/run_system_tests.sh`.
- Retrieved evidence and related-paper analysis use `synthesize`.
- Conversational text detected by query understanding bypasses document retrieval
  but still uses the selected local LLM; research text uses retrieval plus LLM synthesis.
- Text style defaults to `auto`: MiniLM cosine similarity classifies the query
  before providers are configured, and fields below `0.70` are reranked by local
  TinyBERT. The resulting `QueryAnalysis` is passed to the LLM subprocess rather
  than recomputed there. A named style remains an explicit override.
- The web UI creates one timestamped session per analysis. Every text/PDF turn
  appends user-visible communication, routing, component activity, output metadata,
  and failures to the same JSONL file.
- Explicit response styles override the automatically inferred style.
- The web experiment panel exposes the implemented comparison paths: base versus
  team-LoRA/custom Ollama tags; TF-IDF/BM25 versus hybrid retrieval; explicit
  SPECTER2, SPECTER, or MiniLM dense embeddings; and top-5/top-10 RAG depth.
  Session configuration events record these settings for later comparison.
- Refresh restores the active tab-scoped transcript; **New analysis** completes
  that session and creates another. Prior turns are logged but are not LLM memory.
- Optional async PDF/topic jobs remain serialized through one local worker and
  expose cursor-based structured events. See
  [`../docs/OBSERVABILITY.md`](../docs/OBSERVABILITY.md).

## Verified status

- PDF-NLP tests: 23 passed.
- Dataset tests: 2 passed.
- Retrieval tests: 7 passed.
- LLM tests: 66 passed.
- Integration tests: 54 passed.
- Five real PDFs parse and enrich successfully in the production evaluation.
- DrQ-v2 paper-only and related-paper paths completed with `qwen3:8b` and no
  executed mock providers.
- The Vite production build passes. The final in-app browser run reached the
  local server with HTTP 200, but further automation was blocked by the browser
  URL security policy. Session restoration, PDF staging, rendering contracts,
  errors, and New Analysis remain covered by integration/API tests and require
  one manual browser pass.
- Curated integration evidence is stored under `results/traces/` and
  `results/demo/`; committed acceptance sessions live under
  `data/sessions/` (see `data/sessions/README.md`).

## Local latency expectations (`qwen3:8b`)

On a typical local CPU/GPU setup (~20 tokens/s generation):

| Scenario | Approximate wall time |
| --- | ---: |
| Direct chat | ~15 s |
| Topic RAG recommendation | ~60–90 s |
| PDF summary (paper-only, no peer review) | ~3–5 min |
| PDF with peer review enabled | add ~60–120 s |

Use the **Include peer review** checkbox only when you need that section.
The production default keeps peer review off for faster PDF analysis.
Only `qwen3:8b` and `qwen3-research-lora:latest` are supported project models.

## Module-only development

Each capability lives under `modules/` with its own CLI. See `modules/<name>/README.md`.

## More detail

- CLI reference: [`README_cli.md`](README_cli.md)
- Provider wiring: [`INTEGRATION.md`](INTEGRATION.md)
- Frontend: [`frontend/README.md`](frontend/README.md)
