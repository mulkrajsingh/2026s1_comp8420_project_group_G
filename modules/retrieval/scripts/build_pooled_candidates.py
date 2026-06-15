"""Build pooled candidate sets for gold-standard expansion.

Runs five retrievers per evaluation query, unions top results with hand-labelled
seeds, and writes deduplicated candidate pools for citation-graph judging.

Output: modules/retrieval/data/processed/pooled_candidates.json

Usage:
    python modules/retrieval/scripts/build_pooled_candidates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
CORPUS = REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.fixtures import EVAL_QUERIES, load_papers  # noqa: E402
from app.retrieval.embeddings import EmbeddingRetriever  # noqa: E402
from app.retrieval.hybrid_ranker import HybridRanker  # noqa: E402
from app.retrieval.section_aware import SectionAwareRetriever  # noqa: E402
from app.retrieval.tfidf_bm25 import BM25Retriever, TfidfRetriever  # noqa: E402

POOL_TOP_K = 15
OUT_PATH = RETRIEVAL_DIR / "data/processed/pooled_candidates.json"


def main() -> None:
    """Build pooled candidate JSON from multi-retriever top-k unions."""
    papers = load_papers(CORPUS)
    print(f"Loaded {len(papers)} papers")

    tfidf = TfidfRetriever().fit(papers)
    bm25 = BM25Retriever().fit(papers)
    print("Fitted TF-IDF + BM25")

    emb = EmbeddingRetriever()
    emb._papers = list(papers)
    emb.load_embeddings(RETRIEVAL_DIR / "data/processed/retrieval_index/embeddings")
    print(f"Loaded precomputed embeddings ({emb.model_name})")

    print("Fitting section-aware retriever (encodes 4 text variants x 5000 papers)...")
    section = SectionAwareRetriever(emb)
    section.fit(papers, show_progress=True)
    print("Section-aware retriever ready")

    hybrid = HybridRanker(
        tfidf_retriever=tfidf,
        bm25_retriever=bm25,
        embedding_retriever=emb,
        section_aware_retriever=section,
    )
    hybrid._papers = papers

    retrievers = {
        "tfidf": tfidf,
        "bm25": bm25,
        "embedding_specter2_base": emb,
        "section_aware": section,
        "hybrid_ensemble": hybrid,
    }

    pooled = {}
    for q in EVAL_QUERIES:
        query_text = q["query"]
        seed_ids = list(q["relevant_ids"])
        sources: dict[str, list[str]] = {pid: ["seed"] for pid in seed_ids}

        for name, retriever in retrievers.items():
            results = retriever.retrieve(query_text, top_k=POOL_TOP_K)
            for paper, _score in results:
                pid = paper["paper_id"]
                sources.setdefault(pid, []).append(name)

        candidates = sorted(sources.keys())
        pooled[query_text] = {
            "seed_ids": seed_ids,
            "candidates": candidates,
            "sources": sources,
        }
        print(f"{query_text!r}: {len(candidates)} unique candidates (seeds={len(seed_ids)})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(pooled, f, indent=2)
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
