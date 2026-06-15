# System Test Catalogue

Last updated: 2026-06-14

This is the acceptance checklist for the COMP8420 Research Paper Assistant. It
combines the assignment rubric, integration contract, automated tests, real-PDF
fixtures, and production session evidence.

## Evidence Rules

- `ollama` with `qwen3:8b` is required for model-quality claims.
- Production claims use `file-backed` and `live` session rows, not mock rows.
- A passing run must not log paper bodies, full prompts, or temporary upload
  paths.
- Retrieval and PDF-NLP quality results retain their documented small-evaluation
  limitations.

## Acceptance Matrix

| ID | Area | Input and backend | Expected result | Evidence and command |
| --- | --- | --- | --- | --- |
| DATA-01 | Preprocessing | Canonical arXiv sample | Deterministic balanced `PaperRecord` corpus with valid required fields | `modules/dataset/tests/test_build_balanced_corpus.py` |
| DATA-02 | Classification | Deterministic internal split | Metrics, comparison CSV, and confusion matrices are reproducible | `modules/dataset/tests/test_domain_classifier.py` |
| PDF-01 | PDF parsing | Five committed PDFs | Metadata, canonical sections, references, and stable IDs satisfy `ParsedPaper` | `tests/e2e/test_pdf_nlp.py` |
| PDF-02 | Reference extraction | Numbered and author-year fixtures | Entries are separated, deduplicated, normalized, and metadata-preserving | `modules/pdf_nlp/tests/test_pdf_parser.py` |
| PDF-03 | NLP enrichment | DrQ-v2 with validated local assets | POS, entities, keyphrases, TextRank summary, checks, timings, and provenance exist | `python -m app.cli analyze-paper ...` in `modules/pdf_nlp/` |
| PDF-04 | Model assets | Missing or unsafe asset archive | Missing checksums fail clearly; path traversal is rejected | `modules/pdf_nlp/tests/test_paper_analysis.py` |
| RET-01 | Lexical retrieval | Explicit RAG query | Relevant RAG papers rank ahead of generic matches | `modules/retrieval/tests/test_query_expansion.py` |
| RET-02 | Hybrid RRF | Synthetic lexical/dense rankings | Papers strong in both rankings receive the best fused rank | `modules/retrieval/tests/test_hybrid_rrf.py` |
| RET-03 | Embedding choice | Explicit model name | Requested model is used or fails clearly; no silent substitution | `modules/retrieval/tests/test_embedding_selection.py` |
| RET-04 | Recommendation output | Topic query over canonical corpus | Ranked papers include score, APA citation, reason, and evidence IDs | `tests/e2e/test_retrieval.py` |
| LLM-01 | Direct chat | Greeting, Ollama | Short answer; route is `direct_llm_chat`; retrieval is not executed | `scripts/rpa chat "hi"` and session JSONL |
| LLM-02 | Topic synthesis | Research question, Ollama | RAG answer is grounded in supplied evidence and records source IDs | `tests/e2e/test_llm.py` plus live session |
| LLM-03 | Paper summary | Parsed DrQ-v2, Ollama | Paper-only summary uses supplied sections and no retrieval replacement | `tests/e2e/test_llm.py` |
| LLM-04 | Peer review | Parsed SIGA or DrQ-v2, Ollama | Structured strengths, weaknesses, missing evidence, and improvements | `tests/e2e/test_llm.py` |
| LLM-05 | Citation guard | Irrelevant candidates | Candidate is not promoted as a direct citation | `modules/llm/tests/test_ollama_runtime.py` |
| LLM-06 | Empty response | Thinking consumes output budget | Runtime retries once without thinking and records fallback metadata | `test_empty_thinking_response_retries_without_thinking` |
| LLM-07 | Base vs LoRA | Fixed prompts, Ollama | Same prompt strategy and input boundaries; negative result remains reportable | `modules/llm/tests/test_model_comparison_fairness.py` |
| INT-01 | Topic pipeline | Topic query, Ollama | Yash corpus -> Bank retrieval -> Mulkraj synthesis -> `AnalysisResult` | `tests/e2e/test_integration_cli.py` |
| INT-02 | PDF with related papers | DrQ-v2, Ollama | Parse first; title, then abstract fallback, becomes Bank query | `test_run_analyze_pdf_registers_retrieval_with_parsed_title` |
| INT-03 | PDF paper-only | DrQ-v2, Ollama | Parser and synthesizer run; retrieval is not registered or executed | `test_run_analyze_pdf_no_related_papers_skips_retrieval` |
| INT-04 | PDF question | Parsed DrQ-v2, Ollama | Answer is grounded in the uploaded paper and skips external retrieval | `test_chat_with_parsed_paper` |
| INT-05 | API contracts | Valid and invalid request bodies | Existing endpoint paths and response JSON shapes remain compatible | Integration API tests and `/docs` |
| INT-06 | Async jobs | PDF upload and topic job | Single worker emits cursor-readable events and terminal state | `test_job_runner_records_success_and_deletes_upload` |
| UI-01 | Text flow | Browser, Ollama | Session creation, direct chat, RAG chat, recommendation cards, and status render | Manual browser run |
| UI-02 | PDF attachment | Browser, DrQ-v2 | PDF is staged as removable chip and sent only on Send/Enter | Manual browser run |
| UI-03 | Session lifecycle | Browser refresh/New analysis | Refresh restores tab session; New analysis completes it and starts another | Integration session tests plus browser |
| PRIV-01 | Temporary upload | API PDF upload | Temporary file is deleted on success and failure | Integration job/session tests |
| PRIV-02 | Log redaction | PDF upload and analysis | Logs retain filename but exclude temp paths and reconstructable paper text | `integration/tests/test_jobs_and_sessions.py` |
| FAIL-01 | Missing PDF | Invalid path | Clear not-found error; failed event is recorded | Parser and root-wrapper tests |
| FAIL-02 | Invalid/empty PDF | Corrupt or zero-byte input | Clear validation error; no stale output is accepted | Parser tests |
| FAIL-03 | Missing dependency/model | Missing spaCy, KeyBERT, or asset | Command fails explicitly without downloading or substituting a model | PDF-NLP asset tests and historical session regression |
| FAIL-04 | Missing corpus/evidence | Missing production JSONL or RAG pack | Command fails explicitly; no mock/sample fallback | Provider and LLM CLI tests |
| FAIL-05 | Model/runtime failure | Ollama unavailable, timeout, or empty output | Error is recorded; response is never presented as successful model evidence | Runtime and session tests |
| PERF-01 | PDF parser | Real PDF | Investigate total parser subprocess over 45 seconds | Session `duration_ms` |
| PERF-02 | Retrieval | Canonical corpus | Investigate retrieval subprocess over 180 seconds | Session `duration_ms` |
| PERF-03 | Direct chat | Ollama | Investigate request over 180 seconds | Session `duration_ms` |
| PERF-04 | Technical generation | Ollama | Investigate generation over 660 seconds | Generation metadata and session JSONL |

## Required Commands

```bash
# Complete deterministic/module regression
python -m unittest discover -s modules/dataset/tests -p 'test_*.py' -v
(cd modules/pdf_nlp && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd modules/retrieval && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd modules/llm && python -m unittest discover -s tests -p 'test_*.py' -v)
(cd integration && PYTHONPATH="..:." python -m unittest discover -s tests -p 'test_*.py' -v)
(cd integration/frontend && npm run build)

# Real-paper system suite
REQUIRE_OLLAMA=1 ./tests/run_system_tests.sh
```

## Balanced Live Acceptance Run

Use `qwen3:8b` and retain the matching sanitized session IDs:

1. Direct greeting without retrieval.
2. Topic RAG answer.
3. Five-paper recommendation with APA citations.
4. DrQ-v2 paper-only analysis.
5. DrQ-v2 analysis with related papers.
6. Peer review.
7. Specific DrQ-v2 question about compared baselines.

Pass when every expected component completes, no mock provider executes, output
contracts validate, evidence IDs resolve, temporary uploads are absent, and no
session row has `status=failed` or a non-empty `error`.

Accepted sessions:

| Case | Session |
| --- | --- |
| Direct greeting | `20260614-175236-509719` |
| Topic RAG answer | `20260614-175249-269180` |
| Five-paper recommendation | `20260614-180410-800955` |
| Paper-only analysis | `20260614-180804-765352` |
| Analysis with related papers | `20260614-182556-476951` |
| Peer review | `20260614-182925-287725` |
| PDF-grounded question | `20260614-183533-906294` |

All seven traces have zero failed events, zero mock-provider markers, zero
absolute temporary paths, and no retained PDF uploads.

## Browser Acceptance Status

The automated browser request reached the local application and received HTTP
200, but the browser runtime then blocked the local URL under its security
policy. Do not treat this as a UI pass or application failure. The integration
suite covers session restoration, staged uploads, response/error payloads,
upload deletion, and New Analysis lifecycle; perform one manual visual pass for
`UI-01` through `UI-03`.

## Session-Derived Regression Cases

The 64 production session files inspected on 2026-06-14 contained 17 failed
events. They reduce to four regressions:

| Regression | Historical symptom | Required protection |
| --- | --- | --- |
| Missing PDF-NLP dependencies | `spacy` missing despite model assets existing | Explicit dependency error and documented environment check |
| Empty Ollama output | Thinking consumed the output budget | One direct-answer retry and recorded fallback metadata |
| Interrupted long generation | LLM subprocess exited `-15` | Bounded prompt/context/output budgets and visible failure |
| Incorrect PDF path | `integration/tests/papers/...` did not exist | Resolve CLI paths from the caller and fail clearly |

The active session `20260614-162015-998740` had two successful turns and no
failures. Its second turn recorded about 161 seconds retrieval and 102 seconds
synthesis, both below the investigation thresholds above.

## Automated Test Inventory

The inventory below is authoritative for the 2026-06-14 baseline. Rename or
remove a test only when this section and the progress log are updated.

### Dataset (2)

- `test_build_is_deterministic_and_balanced`
- `test_writes_reproducible_metric_artifacts`

### PDF-NLP (23)

- `test_optional_sections_are_preserved`
- `test_structural_checks_are_deterministic`
- `test_enrichment_populates_contract_fields`
- `test_manifest_validation_reports_missing_assets`
- `test_archive_installer_rejects_path_traversal`
- `test_parse_sample_pdf_matches_contract`
- `test_missing_sections_are_empty_strings`
- `test_stable_paper_id_from_filename`
- `test_missing_pdf_raises_clear_error`
- `test_empty_pdf_raises_clear_error`
- `test_invalid_pdf_raises_clear_error`
- `test_cli_round_trip_json`
- `test_bracket_numbered_reference_extraction`
- `test_dot_numbered_reference_extraction`
- `test_one_line_author_year_reference_extraction`
- `test_multiline_author_year_reference_extraction`
- `test_year_suffixes_create_separate_references`
- `test_names_accents_initials_hyphens_apostrophes_and_et_al`
- `test_reference_metadata_is_preserved`
- `test_standalone_page_numbers_are_removed`
- `test_duplicate_references_are_removed`
- `test_missing_references_section_returns_empty_list`
- `test_sample_pdf_references_are_individual_entries`

### Retrieval (7)

- `test_explicit_embedding_model_is_not_silently_replaced`
- `test_saved_index_requires_matching_corpus_and_model`
- `test_rrf_prefers_paper_ranked_high_by_both_engines`
- `test_explicit_rag_query_prioritizes_rag_papers` (hybrid RRF)
- `test_expand_query_adds_rag_aliases`
- `test_rag_aliases_are_added_only_for_explicit_rag_phrase`
- `test_explicit_rag_query_prioritizes_rag_papers` (query expansion)

### LLM (66)

- Checkpointing: latest/valid/atomic restore/metadata/retention/completion/replacement.
- Faithfulness: arXiv and synthetic source-ID extraction and acceptance.
- Library algorithms: BM25 ranking/empty/top-k/schema and percentile edge cases.
- Model comparison: paper-only fairness, DOI safety, input boundaries, equal strategy.
- Ollama runtime: citation guard, bounded reasoning, think cleanup, counters,
  budgets for summary/question/review/recommendation, and empty-response retry.
- Recommendation summaries: source parsing, fallback, and Bank-owned metadata.
- Paper CLI: summary, review, direct chat, precomputed analysis, schema and
  missing-input failures, backend errors, and existing RAG synthesis.
- Query understanding: round trip, confidence/fallback paths, overrides,
  routing, observability scores, prompt adaptation, and topic extraction.
- Synthesis parsing: JSON, noisy/nested JSON, Markdown fallback, failure,
  summary cleanup, and review extraction.
- Training data: date normalization, source IDs, DOI handling, task coverage,
  and train/evaluation separation.

Exact names remain in `modules/llm/tests/test_*.py`; the suite count is an
acceptance assertion.

### Integration (54)

- Sessions/jobs (16): schema, redaction, deletion, lifecycle, transcript,
  concurrency, failure logging, direct chat, and PDF chat.
- Ollama residency (6): warm, startup failure, model switch/reuse, unsupported
  model, and web lifespan.
- Recommendation routing (4): route, extracted topic, full corpus/top five, API shape.
- Production providers (17): corpus limits, source metadata, live CLI calls,
  missing inputs, parse-first behavior, no mocks, and API orchestration.
- Query routing (4): PDF chat, capability greeting, no-retrieval chat, reuse.
- Root wrapper (2): caller-relative PDF and parsed-paper JSON paths.
- Web launcher (5): build freshness, missing tooling, failures, install/build,
  and uvicorn options.

Exact names remain in `integration/tests/test_*.py`; the suite count is an
acceptance assertion.

### System E2E (12)

- `test_llm_summary_and_rag_synthesis`
- `test_integration_topic_pdf_review_and_paper_chat`

- `test_parse_real_papers`
- `test_full_nlp_analysis_on_real_pdf`
- `test_recommend_topic_with_apa_citations`
- `test_paper_only_summarize`
- `test_rag_synthesize`
- `test_peer_review_style_summarize`
- `test_search_topic_from_prompts`
- `test_analyze_pdf_full_pipeline`
- `test_peer_review_pdf`
- `test_chat_with_parsed_paper`
