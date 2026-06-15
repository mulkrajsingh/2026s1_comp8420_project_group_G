"""Iterative tuning for SPECTER, section-aware, and hybrid retrievers.

Runs query-format variants, section weight grids, hybrid ensembles, and RRF
fusion against the frozen v2 gold standard. Does not rebuild gold labels.

Usage:
    /opt/miniconda3/bin/python modules/retrieval/scripts/tune_retrievers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.evaluation import evaluate_retriever  # noqa: E402
from app.fixtures import load_papers  # noqa: E402
from app.retrieval.embeddings import EmbeddingRetriever  # noqa: E402
from app.retrieval.hybrid_ranker import HybridRanker, HybridRRFRanker  # noqa: E402
from app.retrieval.section_aware import SectionAwareRetriever  # noqa: E402
from app.retrieval.tfidf_bm25 import BM25Retriever, TfidfRetriever  # noqa: E402

CORPUS = REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"
EMB_DIR = RETRIEVAL_DIR / "data/processed/retrieval_index/embeddings"
V2_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"


def load_queries_v2():
    data = json.loads(V2_PATH.read_text())
    return [{"query": q["query"], "relevant_ids": q["relevant_ids_v2"]} for q in data]


def mean_metrics(df, keys=("P@5", "nDCG@5", "MAP", "MRR")):
    return {k: round(float(df[k].mean()), 4) for k in keys}


def fmt_metrics(m: dict) -> str:
    return " ".join(f"{k}={v:.3f}" for k, v in m.items())


class SpecterQueryVariant(EmbeddingRetriever):
    """SPECTER2 with configurable query encoding."""

    def __init__(self, base: EmbeddingRetriever, query_formatter):
        self._base = base
        self._papers = base._papers
        self._embeddings = base._embeddings
        self.model_name = base.model_name
        self._model = base._model
        self._query_formatter = query_formatter
        self._variant_name = query_formatter.__name__

    def retrieve(self, query: str, top_k: int = 10):
        from sklearn.metrics.pairwise import cosine_similarity

        q_text = self._query_formatter(query)
        q_emb = self._model.encode(
            [q_text], convert_to_numpy=True, normalize_embeddings=True
        )
        scores = cosine_similarity(q_emb, self._embeddings)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._papers[i], float(scores[i])) for i in top_idx]

    @property
    def name(self):
        return f"embedding_{self._variant_name}"


def q_plain(q):
    return q


def q_title_style(q):
    """Treat keyword query as a paper title (SPECTER training format hint)."""
    return q.strip()


def q_title_abstract_dup(q):
    """Duplicate query as title+abstract — closer to paper encoding."""
    q = q.strip()
    return f"{q} {q}"


def q_pseudo_abstract(q):
    """Prefix with abstract cue used in scientific writing."""
    return f"This paper presents work on {q.strip()}."


class SectionWeightVariant(SectionAwareRetriever):
    def __init__(self, base_section: SectionAwareRetriever, weights, label):
        self._emb = base_section._emb
        self._weights = weights
        self._section_embs = base_section._section_embs
        self._papers = base_section._papers
        self._label = label

    def fit(self, papers, show_progress=True):
        return self

    @property
    def name(self):
        return f"section_{self._label}"


class HybridWeightVariant(HybridRanker):
    def __init__(self, *args, label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._label = label

    @property
    def name(self):
        return f"hybrid_{self._label}"


class HybridRRFWeightVariant(HybridRRFRanker):
    def __init__(self, *args, label: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._label = label

    @property
    def name(self):
        return f"hybrid_rrf_{self._label}"


def build_base_retrievers(papers):
    tfidf = TfidfRetriever().fit(papers)
    bm25 = BM25Retriever().fit(papers)
    emb = EmbeddingRetriever()
    emb._papers = list(papers)
    emb.load_embeddings(EMB_DIR)
    section = SectionAwareRetriever(emb)
    section.fit(papers, show_progress=False)
    return tfidf, bm25, emb, section


def main() -> None:
    """Grid-search retriever variants against the frozen v2 gold standard."""
    papers = load_papers(CORPUS)
    queries = load_queries_v2()
    print(f"Loaded {len(papers)} papers, {len(queries)} v2 queries\n")

    tfidf, bm25, emb, section = build_base_retrievers(papers)
    baseline_tfidf = mean_metrics(evaluate_retriever(tfidf, queries))
    print(f"TF-IDF baseline: {fmt_metrics(baseline_tfidf)}\n")

    # --- SPECTER query format variants ---
    print("=== SPECTER query format variants ===")
    best_specter = None
    for formatter in (q_plain, q_title_abstract_dup, q_pseudo_abstract):
        variant = SpecterQueryVariant(emb, formatter)
        m = mean_metrics(evaluate_retriever(variant, queries))
        print(f"  {variant.name:30s} {fmt_metrics(m)}")
        score = m["P@5"] + m["nDCG@5"]
        if best_specter is None or score > best_specter[1]:
            best_specter = (variant, score, m)

    # --- Section-aware weight grids ---
    print("\n=== Section-aware weight grids ===")
    weight_grids = [
        ("default", {"whole": 0.4, "method": 0.3, "results": 0.2, "background": 0.1}),
        ("whole_heavy", {"whole": 0.6, "method": 0.2, "results": 0.1, "background": 0.1}),
        ("whole_max", {"whole": 0.7, "method": 0.15, "results": 0.1, "background": 0.05}),
        ("method_heavy", {"whole": 0.3, "method": 0.4, "results": 0.2, "background": 0.1}),
        ("keyword", {"whole": 0.5, "method": 0.25, "results": 0.15, "background": 0.1}),
    ]
    best_section = None
    for label, weights in weight_grids:
        sv = SectionWeightVariant(section, weights, label)
        m = mean_metrics(evaluate_retriever(sv, queries))
        print(f"  section_{label:15s} {fmt_metrics(m)}")
        score = m["P@5"] + m["nDCG@5"]
        if best_section is None or score > best_section[2]:
            best_section = (label, weights, score, m)

    # --- Hybrid weight grids (fixed BM25) ---
    print("\n=== Hybrid weight grids ===")
    weight_sets = [
        ("default", {"tfidf": 0.30, "bm25": 0.05, "embedding": 0.25, "section_aware": 0.30, "recency": 0.05, "category": 0.05}),
        ("lexical_heavy", {"tfidf": 0.25, "bm25": 0.20, "embedding": 0.20, "section_aware": 0.25, "recency": 0.05, "category": 0.05}),
        ("bm25_boost", {"tfidf": 0.20, "bm25": 0.25, "embedding": 0.20, "section_aware": 0.25, "recency": 0.05, "category": 0.05}),
        ("semantic_heavy", {"tfidf": 0.15, "bm25": 0.15, "embedding": 0.30, "section_aware": 0.30, "recency": 0.05, "category": 0.05}),
        ("balanced_lex", {"tfidf": 0.25, "bm25": 0.15, "embedding": 0.25, "section_aware": 0.25, "recency": 0.05, "category": 0.05}),
        ("tfidf_bm25_only", {"tfidf": 0.45, "bm25": 0.45, "embedding": 0.0, "section_aware": 0.0, "recency": 0.05, "category": 0.05}),
    ]
    best_hybrid = None
    for label, weights in weight_sets:
        h = HybridWeightVariant(
            tfidf_retriever=tfidf,
            bm25_retriever=bm25,
            embedding_retriever=emb,
            section_aware_retriever=section,
            weights=weights,
            label=label,
        )
        h._papers = papers
        m = mean_metrics(evaluate_retriever(h, queries))
        print(f"  hybrid_{label:18s} {fmt_metrics(m)}")
        score = m["P@5"] + m["nDCG@5"]
        if best_hybrid is None or score > best_hybrid[2]:
            best_hybrid = (label, weights, score, m)

    # --- Production RRF: BM25 + SPECTER2 only ---
    print("\n=== Hybrid RRF (BM25 + SPECTER2) weight grids ===")
    best_rrf = None
    for label, weights in [
        ("bm25_40_emb_60", {"bm25": 0.4, "embedding": 0.6}),
        ("balanced", {"bm25": 0.5, "embedding": 0.5}),
        ("bm25_60_emb_40", {"bm25": 0.6, "embedding": 0.4}),
    ]:
        h = HybridRRFWeightVariant(
            bm25_retriever=bm25,
            embedding_retriever=emb,
            weights=weights,
            label=label,
        )
        h._papers = papers
        m = mean_metrics(evaluate_retriever(h, queries))
        print(f"  hybrid_rrf_{label:18s} {fmt_metrics(m)}")
        score = m["P@5"] + m["nDCG@5"]
        if best_rrf is None or score > best_rrf[2]:
            best_rrf = (label, weights, score, m)

    # --- Legacy 4-engine RRF (evaluation reference) ---
    print("\n=== Legacy 4-engine RRF fusion ===")
    for label, weights in [
        ("rrf_default", {"tfidf": 0.30, "bm25": 0.05, "embedding": 0.25, "section_aware": 0.30}),
        ("rrf_lexical", {"tfidf": 0.30, "bm25": 0.25, "embedding": 0.20, "section_aware": 0.25}),
        ("rrf_balanced", {"tfidf": 0.25, "bm25": 0.25, "embedding": 0.25, "section_aware": 0.25}),
    ]:
        h = HybridWeightVariant(
            tfidf_retriever=tfidf,
            bm25_retriever=bm25,
            embedding_retriever=emb,
            section_aware_retriever=section,
            weights=weights,
            label=label,
        )
        h._papers = papers
        m = mean_metrics(evaluate_retriever(h, queries))
        print(f"  hybrid_{label:18s} {fmt_metrics(m)}")
        score = m["P@5"] + m["nDCG@5"]
        if best_hybrid is None or score > best_hybrid[2]:
            best_hybrid = (label, weights, score, m)

    print("\n=== BEST CONFIGURATIONS ===")
    print(f"Best SPECTER:  {best_specter[2] if best_specter else 'n/a'}")
    print(f"Best section:  {best_section[3] if best_section else 'n/a'}  weights={best_section[1] if best_section else 'n/a'}")
    print(f"Best RRF:      {best_rrf[3] if best_rrf else 'n/a'}  weights={best_rrf[1] if best_rrf else 'n/a'}")
    print(f"Best hybrid:   {best_hybrid[3] if best_hybrid else 'n/a'}  weights={best_hybrid[1] if best_hybrid else 'n/a'}")


if __name__ == "__main__":
    main()
