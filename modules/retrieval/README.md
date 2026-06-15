# Retrieval module

**Module:** Retrieval — [`docs/CONTRIBUTIONS.md`](../../docs/CONTRIBUTIONS.md)  
**Full app:** [`integration/`](../../integration/)

Scope: RAG, recommendation, citation analysis, evidence/source attribution, evaluation metrics.

## Standalone use

```bash
pip install -r requirements.txt   # from the repository root
cd modules/retrieval

# Smoke test — build retrieval index
python -m app.cli build-retrieval-index \
    --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
    --out data/processed/retrieval_index/

# Recommend papers for a topic query
python -m app.cli recommend-topic \
    --query "retrieval augmented generation for scientific papers" \
    --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
    --out outputs/recommendations.json

# Run evaluation comparison table
python -m app.cli evaluate-retrieval \
    --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
    --out results/retrieval/
```

## Module Overview

| Module | Description |
|--------|-------------|
| `app/retrieval/tfidf_bm25.py` | TF-IDF and BM25 lexical baseline retrievers |
| `app/retrieval/embeddings.py` | SPECTER2 / sentence-transformer dense retrieval |
| `app/retrieval/section_aware.py` | Section-aware abstract representation (Xu et al. 2025) |
| `app/retrieval/hybrid_ranker.py` | Hybrid ensemble ranking (6 signals combined) |
| `app/citation.py` | APA 7th edition citation formatting |
| `app/rag_pack.py` | RagEvidencePack builder (consumed by the LLM module) |
| `app/evaluation.py` | P@K, R@K, F1@K, MRR, MAP, nDCG evaluation metrics |
| `app/cli.py` | CLI entry point |
| `app/fixtures.py` | Mock PaperRecord data and evaluation queries |

## Outputs

| Path | Contents |
|------|----------|
| `outputs/recommendations.json` | Generated ranked Recommendation list (ignored) |
| `outputs/rag_evidence_pack.json` | Generated RagEvidencePack (ignored) |
| `results/retrieval/retrieval_comparison_keyword.csv` | Keyword-style query benchmark |
| `results/retrieval/retrieval_comparison_user.csv` | User-like natural-language benchmark |
| `results/retrieval/retrieval_eval_summary.md` | Dual-benchmark narrative |
| `results/retrieval/*.png` | Visualisation charts (PCA, heatmap, bar charts) |
| `data/processed/retrieval_index/` | Saved index artifacts |

## Handoff to LLM module

**Production ranker:** integration and CLI default to `hybrid_rrf` (BM25 + SPECTER2
reciprocal rank fusion). `HybridRanker` ensemble and section-aware stacks are
evaluation baselines only — see `app/retrieval/hybrid_ranker.py` and
`results/retrieval/retrieval_eval_summary.md`.

```bash
python -m app.cli recommend-topic \
    --query "YOUR QUERY HERE" \
    --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
    --out outputs/recommendations.json
# Then run LLM module's:
# python -m app.cli synthesize --evidence outputs/rag_evidence_pack.json ...
```

## Canonical Corpus

Retrieval scripts read Yash's enriched dataset directly rather than keeping a
module-local copy:
```bash
python -m app.cli recommend-topic \
    --query "transformer attention for text classification" \
    --papers ../dataset/data/processed/dev_5k_balanced.jsonl \
    --out outputs/recommendations.json
```
