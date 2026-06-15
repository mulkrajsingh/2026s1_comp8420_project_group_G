# Workstream path map (pre-rename → current)

After the modular restructure, capability folders live under `integration/` and `modules/`. Old team-named paths are **not** part of the runtime layout; use this table if you have bookmarks, slides, or logs that still reference them.

| Old path | Current path | Owner | Notes |
| --- | --- | --- | --- |
| `sid_workstream-3/` | [`integration/`](../integration/) | Sidharth | Single app entry: `python -m app.cli`, `run`, session JSONL |
| `yash/Assignment_3/`, `modules/Assignment_3/` | [`modules/dataset/`](../modules/dataset/) | Yash | Canonical corpus and enrichment evidence |
| `bank_rag_workstream/` | [`modules/retrieval/`](../modules/retrieval/) | Bank | `recommend-topic`, `build-retrieval-index`, `evaluate-retrieval` |
| `mulkraj_llm_workstream/` | [`modules/llm/`](../modules/llm/) | Mulkraj | `synthesize`, `compare-prompts`, `compare-models` |
| `nadiyah_stages/` | [`modules/pdf_nlp/`](../modules/pdf_nlp/) | Nadiyah | Live `parse-pdf` CLI and canonical `ParsedPaper`; stage plans remain canonical under `team_plans/nadiyah_stages/` |

## Local-only data

| Path | Purpose |
| --- | --- |
| `modules/dataset/data/raw/` | Ignored raw arXiv snapshot used to rebuild the corpus |
| `modules/dataset/data/cache/s2/` | Ignored Semantic Scholar cache used to rebuild enrichment |

Root team-named folders and upload archives are obsolete and must not be
reintroduced.

## Quick start (current)

```bash
cd integration && python -m app.cli run --query "your topic"
# or from repo root:
scripts/rpa run --query "your topic"
```

See also: [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), root [`readme.md`](../readme.md).
