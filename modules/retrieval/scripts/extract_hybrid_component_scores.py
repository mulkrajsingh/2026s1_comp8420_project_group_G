"""Cache normalised hybrid ranker component scores per query and paper.

Fits TF-IDF, BM25, SPECTER2, and section-aware retrievers once, then stores
min-max normalised score vectors so weight tuning scripts can rerun quickly.

Output:
    modules/retrieval/data/processed/hybrid_component_scores.npz
    modules/retrieval/data/processed/hybrid_component_meta.json

Usage:
    python modules/retrieval/scripts/extract_hybrid_component_scores.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
CORPUS = REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.fixtures import EVAL_QUERIES, load_papers  # noqa: E402
from app.retrieval.embeddings import EmbeddingRetriever  # noqa: E402
from app.retrieval.hybrid_ranker import HybridRanker, _normalise  # noqa: E402
from app.retrieval.section_aware import SectionAwareRetriever  # noqa: E402
from app.retrieval.tfidf_bm25 import BM25Retriever, TfidfRetriever  # noqa: E402

OUT_NPZ = RETRIEVAL_DIR / "data/processed/hybrid_component_scores.npz"
OUT_META = RETRIEVAL_DIR / "data/processed/hybrid_component_meta.json"

NORMALISED_KEYS = ("tfidf", "bm25", "embedding", "section_aware")
RAW_KEYS = ("recency", "category")


def main() -> None:
    """Extract and save hybrid component score arrays."""
    papers = load_papers(CORPUS)
    print(f"Loaded {len(papers)} papers")

    tfidf = TfidfRetriever().fit(papers)
    bm25 = BM25Retriever().fit(papers)
    print("Fitted TF-IDF + BM25")

    emb = EmbeddingRetriever()
    emb._papers = list(papers)
    emb.load_embeddings(RETRIEVAL_DIR / "data/processed/retrieval_index/embeddings")
    print(f"Loaded precomputed embeddings ({emb.model_name})")

    print("Fitting section-aware retriever...")
    section = SectionAwareRetriever(emb)
    section.fit(papers, show_progress=True)
    print("Section-aware retriever ready")

    ranker = HybridRanker(
        tfidf_retriever=tfidf,
        bm25_retriever=bm25,
        embedding_retriever=emb,
        section_aware_retriever=section,
    )
    ranker._papers = papers

    paper_ids = [p["paper_id"] for p in papers]
    arrays: dict[str, np.ndarray] = {}
    queries_used = []

    for qi, q in enumerate(EVAL_QUERIES):
        query_text = q["query"]
        queries_used.append(query_text)
        raw = ranker._score_map(query_text, top_k_each=len(papers))

        for key in NORMALISED_KEYS:
            vals = [raw[pid][key] for pid in paper_ids]
            normed = _normalise(vals)
            arrays[f"{key}_{qi}"] = np.array(normed, dtype=np.float32)

        for key in RAW_KEYS:
            vals = [raw[pid][key] for pid in paper_ids]
            arrays[f"{key}_{qi}"] = np.array(vals, dtype=np.float32)

        print(f"  [{qi}] {query_text!r}: scores extracted")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    OUT_META.write_text(json.dumps({"paper_ids": paper_ids, "queries": queries_used}, indent=2))
    print(f"\nSaved {OUT_NPZ}")
    print(f"Saved {OUT_META}")


if __name__ == "__main__":
    main()
