"""Diagnostic: why does TF-IDF beat BM25? Tests tokenization variants.

Run: /opt/miniconda3/bin/python modules/retrieval/scripts/diagnose_bm25.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.evaluation import evaluate_retriever  # noqa: E402
from app.fixtures import EVAL_QUERIES, load_papers  # noqa: E402

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CORPUS = REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"
V2_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"

WORD_RE = re.compile(r"\b\w\w+\b")


def tok_split(text):  # current production tokenizer
    return text.lower().split()


def tok_regex(text):  # punctuation-stripped, like sklearn default
    return WORD_RE.findall(text.lower())


def tok_regex_nostop(text):
    return [t for t in WORD_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def tok_regex_nostop_bigram(text):
    uni = [t for t in WORD_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]
    bi = [f"{a}_{b}" for a, b in zip(uni, uni[1:])]
    return uni + bi


class BM25Variant:
    def __init__(self, papers, tokenizer, name, k1=1.5, b=0.75, cls=BM25Okapi):
        self._papers = list(papers)
        self._tok = tokenizer
        self._name = name
        corpus = [tokenizer(f"{p['title']} {p['abstract']}") for p in self._papers]
        self._bm25 = cls(corpus, k1=k1, b=b)

    def retrieve(self, query, top_k=10):
        scores = self._bm25.get_scores(self._tok(query))
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self._papers[i], float(scores[i])) for i in idx if scores[i] > 0]

    @property
    def name(self):
        return self._name


def punctuation_stats(papers):
    """How many query terms get lost to punctuation under .split()?"""
    sample = " ".join(f"{p['title']} {p['abstract']}" for p in papers[:500]).lower()
    split_tokens = set(sample.split())
    regex_tokens = set(WORD_RE.findall(sample))
    only_regex = regex_tokens - split_tokens
    print(f"  unique tokens via .split()      : {len(split_tokens)}")
    print(f"  unique tokens via regex \\w\\w+   : {len(regex_tokens)}")
    print(f"  clean tokens missed by .split() : {len(only_regex)} "
          f"(e.g. {sorted(list(only_regex))[:8]})")


def main():
    papers = load_papers(CORPUS)
    print(f"Loaded {len(papers)} papers\n")

    print("Punctuation impact on tokenization (first 500 papers):")
    punctuation_stats(papers)
    print()

    queries_v1 = [{"query": q["query"], "relevant_ids": q["relevant_ids"]} for q in EVAL_QUERIES]
    v2 = json.loads(V2_PATH.read_text())
    queries_v2 = [{"query": q["query"], "relevant_ids": q["relevant_ids_v2"]} for q in v2]

    variants = [
        BM25Variant(papers, tok_split, "bm25_split(CURRENT)"),
        BM25Variant(papers, tok_regex, "bm25_regex(no-punct)"),
        BM25Variant(papers, tok_regex_nostop, "bm25_regex+nostop"),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25_nostop+bigram"),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25_bi_k1.2b.5", k1=1.2, b=0.5),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25_bi_k1.5b.4", k1=1.5, b=0.4),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25_bi_k0.9b.4", k1=0.9, b=0.4),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25Plus_bi_b.4", k1=1.5, b=0.4, cls=BM25Plus),
        BM25Variant(papers, tok_regex_nostop_bigram, "bm25L_bi_b.4", k1=1.5, b=0.4, cls=BM25L),
    ]

    for label, qs in [("v1 gold (5/query)", queries_v1), ("v2 gold (pooled)", queries_v2)]:
        print(f"=== {label} ===")
        for v in variants:
            df = evaluate_retriever(v, qs)
            print(f"  {v.name:24s} P@5={df['P@5'].mean():.3f} "
                  f"R@5={df['R@5'].mean():.3f} nDCG@5={df['nDCG@5'].mean():.3f} "
                  f"MAP={df['MAP'].mean():.3f} MRR={df['MRR'].mean():.3f}")
        print()

    print("=== BM25L grid (v2 gold, nostop+bigram) ===")
    for k1 in (1.0, 1.2, 1.5, 2.0):
        for b in (0.25, 0.3, 0.4, 0.5):
            v = BM25Variant(papers, tok_regex_nostop_bigram, f"L_k{k1}_b{b}", k1=k1, b=b, cls=BM25L)
            df = evaluate_retriever(v, queries_v2)
            print(f"  k1={k1} b={b}: P@5={df['P@5'].mean():.3f} "
                  f"nDCG@5={df['nDCG@5'].mean():.3f} MAP={df['MAP'].mean():.3f} MRR={df['MRR'].mean():.3f}")
    print()

    # Per-query: TF-IDF (current) vs best BM25 variant
    from app.retrieval.tfidf_bm25 import TfidfRetriever
    tfidf = TfidfRetriever().fit(papers)
    best_bm25 = BM25Variant(papers, tok_regex_nostop_bigram, "bm25_best", k1=1.5, b=0.4, cls=BM25L)
    print("=== Per-query P@5 (v2 gold): TF-IDF vs best BM25 ===")
    dft = evaluate_retriever(tfidf, queries_v2)
    dfb = evaluate_retriever(best_bm25, queries_v2)
    for i, q in enumerate(queries_v2):
        print(f"  P@5 tfidf={dft['P@5'][i]:.2f}  bm25={dfb['P@5'][i]:.2f}  | {q['query']}")


if __name__ == "__main__":
    main()
