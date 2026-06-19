# Canonical Research Assistant UI

A conversational web UI for the Research Paper Analysis system.
Built with **Vite** (vanilla JS, lightweight) and **pnpm**. Talks to the local
FastAPI backend.

## Run (single URL)

From the repository root:

```text
python rpa.py web
```

This installs frontend dependencies when needed, builds stale or missing assets,
and serves the UI and API at http://127.0.0.1:8000.

## Run (development)

1. Start the backend from the repository root:
   ```text
   python rpa.py web --reload
   ```
2. Start Vite from the repository root:
   ```text
   pnpm --dir integration/frontend dev
   ```
3. Open http://localhost:5173

The dev server proxies `/api/*` to `http://127.0.0.1:8000`, so no CORS setup is needed.

## Build for submission

```text
python rpa.py web --rebuild
```

## What it does
- Intent-aware chat and PDF upload through the canonical production API.
- One timestamped session per analysis with synchronous requests per user action.
- Refresh restores the tab-scoped transcript; **New analysis** completes the
  current session and creates another.
- Greetings bypass corpus preparation and retrieval, then use direct local Ollama
  generation through the query-understanding route.
- Full `AnalysisResult` rendering: metadata, summary, findings, gaps,
  recommendations, citations, evidence, peer review, and source labels.
- Separate deterministic PDF-NLP sections for keywords/entities, POS overview,
  extractive summary, and structural checks.
- Compact accessible request status near the composer; detailed execution events
  remain in `integration/data/sessions/<session-id>/session.jsonl` outside the
  conversation.
- The log contains full user-visible messages and responses, but prior turns are
  not sent back to the LLM as conversational memory.
- Automatic response style by default, inferred locally with MiniLM cosine
  similarity and a TinyBERT fallback below `0.70` confidence. Named styles remain
  explicit overrides. Topic and PDF modes require a local Ollama model.
- A backend-populated model dropdown for the installed base and team-LoRA
  Ollama tags, plus hybrid RRF versus TF-IDF retrieval rankers, SPECTER2/SPECTER/MiniLM dense
  embeddings for hybrid mode, and top-5 versus top-10 RAG candidate depth. Every selection is
  recorded in the session JSONL. Changing generation models unloads the previous
  project model before the selected model is warmed. An explicitly selected
  embedding model must be available locally and is never silently replaced.
- Sanitized GitHub-flavored Markdown rendering for assistant answers, summaries,
  recommendations, and peer-review feedback.
- Escaped structured API fields, loading/error states, responsive layout, and
  dark mode.

The production build was verified against the local FastAPI app with a real
Ollama topic run. PDF behavior, structured session events, and upload deletion
are covered by API/integration tests using repository PDFs. Optional queued job
endpoints remain available for compatibility and observability tooling.

## If pnpm pauses on install

pnpm may say `Ignored build scripts: esbuild`. Run this once and select esbuild:
```text
pnpm approve-builds
```
(It's also pre-approved in `pnpm-workspace.yaml`, so on most setups `pnpm dev` just works.)
