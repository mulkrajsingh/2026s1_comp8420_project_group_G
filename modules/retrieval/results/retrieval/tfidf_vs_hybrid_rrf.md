# TF-IDF vs Hybrid RRF Comparison

Head-to-head comparison on the same 5,000-paper corpus. Full multi-retriever
tables remain in `retrieval_comparison_keyword.csv` and
`retrieval_comparison_user.csv`.

## Keyword benchmark (5 legacy keyword-style queries)

| Retriever | P@5 | nDCG@5 | MAP |
| --- | ---: | ---: | ---: |
| tfidf | 0.080 | 0.175 | 0.060 |
| hybrid_rrf | 0.000 | 0.000 | 0.000 |

TF-IDF leads on this short keyword set.

## User-like benchmark (10 natural-language queries)

| Retriever | P@5 | nDCG@5 | MAP |
| --- | ---: | ---: | ---: |
| hybrid_rrf | 0.380 | 0.464 | 0.426 |
| tfidf | 0.180 | 0.379 | 0.141 |

Hybrid RRF (BM25 + SPECTER2 reciprocal rank fusion) leads on user-like
frontend topic phrasing.

## Production default

The integration UI defaults to `hybrid_rrf` to satisfy hybrid-search
requirements and better match natural-language topic queries. TF-IDF remains
available as a lexical baseline for controlled comparison.

Do not generalise either benchmark beyond its query set.
