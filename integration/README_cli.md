# Research Paper Analysis — Integration Workstream (Sidharth)

CLI · API · Frontend · Demo · Integration for the COMP8420 Use Case 3 system.
Production topic and PDF commands call the real capability modules through
subprocess adapters. `run-demo` remains the deterministic mock/cached path.

## Quick start
From the repository root:

```bash
pip install -r requirements.txt
scripts/rpa run --query "..." --corpus-limit 1000 --top-n 5
scripts/rpa analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf
scripts/rpa analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf --no-related-papers
scripts/rpa session-inspect --component pdf_nlp
scripts/rpa run-demo                 # mock/cached smoke demo
scripts/rpa build-artifacts          # regenerate all stage documents
scripts/rpa web                      # UI + API at http://127.0.0.1:8000
```

## Commands
- Production integration: `analyze-pdf`, `search-topic`, `recommend`,
  `peer-review`, `chat`, `run`, and `web`.
- Deterministic demo: `run-demo` and `run-demo --auto`.
- Member command routing remains available for module-specific commands. PDF
  `basic-nlp`, `peer-review-checks`, `analyze-paper`, and
  `evaluate-real-papers` are implemented module CLIs.
- Reports: `integration-status`, `build-artifacts`
- `run-demo --auto` auto-activates any teammate file that already exists.

`chat` now performs local query classification before provider registration.
Automatic analysis uses MiniLM embedding cosine similarity first, then a batched
TinyBERT cross-encoder fallback for fields below `0.70`. Basic greetings and
thanks bypass retrieval but still use direct local LLM generation; research
questions continue through the recommender and synthesizer. A named `--style`
remains an explicit override. The computed analysis is reused by the child LLM
command, so chat does not load and run the classifiers twice.
The LLM module's `summarize` command is a separate paper-only path and is not the same
as topic `synthesize`.

`analyze-pdf` always invokes Nadiyah's live parser and enrichment first. With related papers
enabled, it passes the parsed title (abstract fallback) to Bank's recommender.
With `--no-related-papers`, it skips retrieval and sends the parsed paper directly
to Mulkraj's `summarize` path. Proxy output proves orchestration only.

FastAPI also exposes queued topic/PDF jobs and cursor-based polling for optional
automation and observability tooling. One worker serializes PDF-NLP and Ollama
loads. The frontend creates one timestamped session per analysis, appends each
synchronous chat/PDF turn to its JSONL trace, and restores the user-visible
transcript after refresh. **New analysis** completes that session. Detailed
processing events remain outside the rendered conversation.

`web` checks whether `frontend/dist/` is missing or older than the Vite sources.
It installs frontend packages when necessary, builds the UI, and starts Uvicorn.
FastAPI then load-warms `qwen3:8b` through Ollama before serving requests, keeps
it resident for the app lifetime, and unloads it on clean shutdown.
Use `--rebuild`, `--reload`, `--host`, and `--port` to override its defaults.

## Where things are
- `app/` — code (see `docs/WORKSTREAM_GUIDE.md` for the file-by-file map)
- `frontend/` — canonical Vite UI; built assets are served by FastAPI
- `results/demo/` and `results/traces/` — curated evidence
- `outputs/`, `data/sessions/` — generated runtime artifacts (ignored)
- `docs/WORKSTREAM_GUIDE.md` — full team reference (mental model, stages, diagram)
- `docs/REPORT_CONTRIBUTION.md` — paste-ready report text for Sidharth's sections
- `INTEGRATION.md` — data flow + how to plug in real modules

## The nine stages
01 CLI scaffold · 02 command wiring · 03 privacy/no-retention · 04 FastAPI ·
05 frontend · 06 cached demo + trace · 07 real-module integration ·
08 video script · 09 packaging checklist. All CLI-testable; each writes its
artifact under `outputs/`.

Start with `docs/WORKSTREAM_GUIDE.md`.
