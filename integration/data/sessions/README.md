# Curated production session traces

These four session folders are committed as rubric evidence for live Ollama
(`qwen3:8b`) integration runs. All other runtime sessions live under
`old_sessions/` (gitignored).

| Session ID | Scenario | Model | Duration | Components |
| --- | --- | --- | ---: | --- |
| `20260614-040608-044342` | General conversational chat (`direct_llm_chat`) | `qwen3:8b` | ~19 s | integration, llm |
| `20260614-042831-590300` | PDF analysis with summary and citation support | `qwen3:8b` | ~288 s | integration, pdf_nlp, retrieval, llm |
| `20260614-044236-784050` | Topic paper recommendation via RAG | `qwen3:8b` | ~116 s | integration, retrieval, llm |
| `20260614-050431-210855` | Specific PDF question (DrQ-v2 SAC/CURL/DrQ) | `qwen3:8b` | ~98 s | integration, llm |

Each folder contains `session.jsonl` (redacted structured events), `manifest.json`,
and `summary.md`. Inspect with:

```bash
scripts/rpa session-inspect 20260614-040608-044342
```
