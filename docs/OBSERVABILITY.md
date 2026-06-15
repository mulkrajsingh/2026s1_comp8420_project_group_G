# Structured Logging and Demonstration Workflow

This document describes the session logging schema used by the integration
layer for evaluation, debugging, and demonstration. Logs are designed to
support reproducibility claims without storing full paper text or prompts.

## Session Layout

Every one-off production command writes a run directory. The web application
uses the same layout for an entire chat analysis, appending every text/PDF turn
until the user selects **New analysis**:

```text
integration/data/sessions/<session-id>/
  session.jsonl
  manifest.json
  summary.md
```

Runtime sessions are ignored. Curated, redacted evidence belongs under
`integration/results/traces/`.

## Event Schema

Every JSONL row includes:

`schema_version`, `timestamp`, `run_id`, `turn_id`, `event_id`, `component`, `phase`,
`status`, `source`, `message`, `duration_ms`, `metrics`, `artifacts`, `error`,
and `payload`.

Components are `pdf_nlp`, `retrieval`, `llm`, and `integration`. Default
production sessions record:

- **Transcript:** `user`, `pdf_attachment`, `assistant`
- **Routing/config:** `user_input`, `route_selected`, `query_analysis` (chat only)
- **Turn progress:** `turn_progress` for major stages (`parse`, `retrieve`,
  `synthesize`, `peer_review`, `done`)
- **Component summaries:** pdf_nlp phase completions, `retrieval`, `synthesis`
- **Outputs:** `outputs_recorded` with relative artifact names and counts
- **Lifecycle:** `meta` (`session_start`, `session_complete`), `request_failed`

Statuses use `queued`, `running`/`started`, `completed`/`succeeded`, and
`failed`.

### Verbose mode

Set `COMP8420_VERBOSE_SESSION_LOG=1` to also record subprocess wrappers,
per-artifact rows, module CLI `user_input`/`user_output`, pdf_nlp `*_started`
phase rows, and unmapped pipeline steps.

### Privacy

Logs never intentionally include extracted paper bodies, full paper prompts, or
temporary upload paths. Temporary paths are replaced with
`[redacted-upload]`. The `assistant` transcript stores user-visible analysis
panels (summary, capped entities, extractive summary text, POS counts) but
omits token-level POS dumps and sentence-level extractive arrays.

## Navigation

```bash
cd integration
python -m app.cli session-inspect
python -m app.cli session-inspect <run-id> --component pdf_nlp
python -m app.cli session-inspect <run-id> --status failed
```

`manifest.json` gives turn/message/event counts and aggregate recorded duration.
`summary.md` groups a filtered report-ready timeline by turn (it omits module
CLI bookkeeping and duplicate diagnostic rows). Use the JSONL file for
filtering or video timelines.

## API and Frontend

The canonical frontend creates one timestamped conversation with
`POST /api/sessions`. Each synchronous text or PDF request includes that
`session_id`:

- `POST /api/chat`
- `POST /api/analyze-pdf`
- `GET /api/sessions/{session_id}` restores the sanitized transcript
- `POST /api/sessions/{session_id}/complete` finalizes **New analysis**

The active ID is tab-scoped in `sessionStorage`, so refresh restores the same
conversation while another tab starts separately. The model remains stateless:
prior turns are recorded but are not added to later prompts.
Optional automation can use the single-worker job endpoints:

- `POST /api/jobs/analyze-pdf`
- `POST /api/jobs/search-topic`
- `GET /api/jobs/{job_id}?after=<cursor>`

Every turn records the full user-visible message or PDF filename, full
user-visible assistant response, input type, query analysis where applicable,
selected route, major turn stages, component completion timings, output
artifact names, and failures. LLM `synthesis` events record whether thinking
was enabled and the selected context/output-token budgets. Logs still omit
hidden reasoning, full prompts, paper bodies, and temporary upload paths.

## Report/Video Workflow

1. Use a real repository PDF or a real topic query.
2. Run with `qwen3:8b` for model-quality evidence.
3. Confirm the session has no failures and expected source labels.
4. Copy only the redacted JSONL/summary and selected output into `integration/results/traces/`.
5. Show the compact UI status, then inspect the matching session timeline and cite
   the same run ID in the report.

Current curated examples include paper-only and related-paper DrQ-v2 Ollama
traces, a real async PDF upload trace, and a topic Ollama UI trace.
