# Project arXiv RAG Corpus Manifest

Output: `data/processed/project_arxiv_rag_corpus_3k.jsonl`
PaperRecord rows: 3000
Target limit: 3000
Seed: 42
Categories: cs.AI, cs.CL, cs.LG, stat.ML
Source: cached Kaggle Cornell arXiv sample
Sampling: existing deterministic reservoir sample
Date normalization: ISO YYYY-MM-DD reconstructed from cached creation day and arXiv id.
