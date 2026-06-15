# Curated production session traces

These session folders are committed as rubric evidence for live Ollama
(`qwen3:8b`) integration runs. All other runtime sessions live under
`old_sessions/` (gitignored).

| Session ID | Scenario | Model | Duration | Components |
| --- | --- | --- | ---: | --- |
| `20260614-040608-044342` | General conversational chat (`direct_llm_chat`) | `qwen3:8b` | ~19 s | integration, llm |
| `20260614-042831-590300` | PDF analysis with summary and citation support | `qwen3:8b` | ~288 s | integration, pdf_nlp, retrieval, llm |
| `20260614-044236-784050` | Topic paper recommendation via RAG | `qwen3:8b` | ~116 s | integration, retrieval, llm |
| `20260614-050431-210855` | Specific PDF question (DrQ-v2 SAC/CURL/DrQ) | `qwen3:8b` | ~98 s | integration, llm |
| `20260615-200528-379540` | Two-turn chat: direct then retrieval-augmented answer | `qwen3:8b` | ~41 s | integration, retrieval, llm |
| `20260615-221959-652335` | Negative regression: paper-only analysis that failed (retained on purpose) | `qwen3:8b` | n/a | integration, pdf_nlp |
| `20260615-222652-753877` | PDF paper-only analysis and summary | `qwen3:8b` | ~167 s | integration, pdf_nlp, llm |
| `20260615-223048-581121` | Multi-turn conversational chat session | `qwen3:8b` | n/a | integration, retrieval, llm |

Each folder contains `session.jsonl` (redacted structured events), `manifest.json`,
and `summary.md`. Inspect with:

```bash
scripts/rpa session-inspect 20260614-040608-044342
```
