# Retrieval Evaluation Summary

Two benchmarks are reported on the same 5,000-paper corpus:

1. **Keyword benchmark** (`retrieval_comparison_keyword.csv`): five legacy
   keyword-style queries used for baseline tuning.
2. **User-like benchmark** (`retrieval_comparison_user.csv`): ten
   natural-language queries with manually reviewed gold labels.

Do not generalise either table beyond its query set.

## Keyword benchmark (P@5)

| Retriever | P@5 | nDCG@5 | MAP |
| --- | ---: | ---: | ---: |
| tfidf | 0.080 | 0.175 | 0.060 |
| embedding_specter2_base | 0.040 | 0.126 | 0.020 |
| section_aware | 0.040 | 0.200 | 0.040 |
| hybrid_ensemble | 0.040 | 0.126 | 0.020 |
| hybrid_rrf | 0.000 | 0.000 | 0.000 |
| bm25 | 0.000 | 0.000 | 0.000 |

## User-like benchmark (P@5)

| Retriever | P@5 | nDCG@5 | MAP |
| --- | ---: | ---: | ---: |
| hybrid_rrf | 0.380 | 0.464 | 0.426 |
| hybrid_ensemble | 0.280 | 0.415 | 0.258 |
| tfidf | 0.180 | 0.379 | 0.141 |
| section_aware | 0.140 | 0.269 | 0.150 |
| bm25 | 0.120 | 0.234 | 0.078 |
| embedding_specter2_base | 0.100 | 0.247 | 0.108 |

## Production choice

Production retrieval uses BM25 + SPECTER2 reciprocal rank fusion to match
assignment hybrid-search requirements and real user phrasing. TF-IDF remains
a lexical baseline on the keyword benchmark; the user-like benchmark is the
better proxy for frontend topic queries.
