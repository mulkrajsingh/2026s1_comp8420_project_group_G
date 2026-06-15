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
| Input file | `data/raw/sample_arxiv.json` |
| Output file | `data/processed/dev_sample.jsonl` |
| Limit | 5,000 |
| Records scanned | 5 |
| Records written | 4 |
| Records skipped (invalid) | 0 |

## Primary-category distribution
| Category | Papers |
| --- | --- |
| `cs.AI` | 1 |
| `cs.CL` | 1 |
| `cs.LG` | 1 |
| `stat.ML` | 1 |
