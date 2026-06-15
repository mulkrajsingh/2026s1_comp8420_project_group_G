# Subset Policy (Stage 01)

## Target categories
Focused NLP / AI / ML scope: `cs.AI`, `cs.CL`, `cs.LG`, `stat.ML`. A paper is included if *any* of its arXiv
categories is in the target set; its primary label is the first of its own
categories in that set.

## Reproducibility
- File-order selection -> same input + same limit = identical subset.
- `source` fixed to `kaggle_arxiv`; display text whitespace-normalised only.

## This run
| Metric | Value |
| --- | --- |
| Input file | `data/raw/arxiv-metadata-oai-snapshot.json` |
| Output file | `data/processed/dev_5k.jsonl` |
| Limit | 5,000 |
| Records scanned | 344,821 |
| Records written | 5,000 |
| Records skipped (invalid) | 0 |

## Primary-category distribution
| Category | Papers |
| --- | --- |
| `cs.AI` | 2,212 |
| `cs.CL` | 489 |
| `cs.LG` | 1,334 |
| `stat.ML` | 965 |
