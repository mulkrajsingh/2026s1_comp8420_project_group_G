# Report Contribution — Sidharth (System Integration, CLI/API/Frontend, Demo)

Paste-ready draft text for the shared report. Trim to fit the 5,000-word limit and
align wording with the team. Covers the parts of the rubric this workstream earns:
System Architecture & Workflow (Methodology), System Integration & Performance
(Implementation, 5 marks), and the privacy/ethics technique.

---

## Methodology — System architecture and workflow

The system is organised as a local, modular pipeline behind a single command-line
entry point, with an optional local API and minimal web UI. Two inputs are
supported: an uploaded paper (PDF) and a free-text topic/question. Both flow
through the same pipeline, which orchestrates four interchangeable modules —
corpus access, PDF parsing, retrieval/recommendation, and LLM synthesis — and
returns a single `AnalysisResult` containing metadata, summary, key findings,
research gaps, recommended papers, APA citations, evidence snippets, and
peer-review feedback.

The defining design choice is **dependency inversion via a provider interface**.
The pipeline depends only on four abstract interfaces (`PaperSource`, `PdfParser`,
`Recommender`, `Synthesizer`), never on a concrete module. Each team member
implements one interface; a registry selects the active implementation at run
time. During development every interface is satisfied by a deterministic mock,
so the full system runs end-to-end before any model exists; at integration each
mock is replaced by the real module one at a time, in dependency order
(corpus → parser → retriever → synthesiser), with no change to the pipeline or
CLI. We chose this over a monolithic script because it lets four people work in
parallel, keeps the demo runnable at all times, and makes failures isolatable to
a single module. The cost is a thin layer of indirection, which we judged worth
it for a five-person integration.

In the implemented PDF path, the integration layer invokes the PDF module's
`parse-pdf` CLI and receives the canonical nested `ParsedPaper`. Related-paper
mode derives its retrieval query from the parsed title, with the abstract as a
fallback, then invokes the retrieval module. Paper-only mode skips retrieval and
passes the same parsed object to the LLM module's `summarize` command. This
ensures PDF parsing and retrieval remain owned by their respective modules
rather than being reimplemented in the LLM layer.

*(Insert the architecture diagram from `docs/WORKSTREAM_GUIDE.md` section 3.)*

## Implementation — System integration and performance

All capabilities are exposed through one CLI (`python -m app.cli`): `analyze-pdf`,
`search-topic`, `recommend`, `peer-review`, `chat`, and a deterministic `run-demo`.
The same functions are wrapped by a local FastAPI service (no duplicated logic)
and consumed by a responsive Vite web UI offering the two input modes, backend
and style controls, full `AnalysisResult` rendering, source labels, and error
states. For demo reliability the `run-demo` command is
deterministic and can run from a cached result, removing any dependence on model
latency or network during the recorded video; a step trace is written for the
report. Integration status (which modules are live versus mock) is reported
automatically.

The production PDF flow was verified on `2107.09645v1.pdf`: parsing produced the
correct title and four authors, populated all canonical section fields, and
separated 52 bibliography entries. Automated verification comprised 23
PDF-NLP tests, 24 integration tests, and 32 LLM tests. The related-paper trace
showed a file-backed corpus and live parser, recommender, and synthesiser; the
paper-only trace executed no retrieval subprocess.

## Privacy and no-retention (ethics technique)

Uploaded papers are processed locally only. Each upload is handled through a
temporary working copy that is deleted immediately after processing unless the
user explicitly saves it; generated artifacts are written only to a controlled
`outputs/` directory and never mixed with user documents; and every
model-generated section is explicitly disclosed in the output. This satisfies the
use case's privacy expectations and is verified by a runtime check.

## How the basic and advanced techniques connect

The integration layer is the connective tissue between the team's techniques: the
basic techniques (preprocessing, TF-IDF/BM25, classification, NER/POS) produce the
corpus and parsed inputs; the advanced techniques (SPECTER2 retrieval, hybrid
ranking, RAG, LLM synthesis, LoRA, LLM-as-judge) produce recommendations and
grounded summaries; and this layer assembles them into one user-facing result
with source-tracked evidence and AI disclosure.

## Limitations and future work (this layer)

The current handoff uses subprocesses and JSON files, which is reproducible.
The web UI makes one synchronous request per chat message or PDF upload, while
optional queued endpoints remain available for serialized automation. Structured
events are retained in session JSONL traces instead of being rendered in the chat.
Proxy generation validates orchestration but is not model-quality evidence; real
`qwen3:8b` topic and PDF traces are available.
PDF-NLP evaluation is limited to five papers with provisional annotations, and
the deterministic NER baseline currently outperforms SciER and hybrid. An
authenticated multi-user API is out of scope for this local v1.
