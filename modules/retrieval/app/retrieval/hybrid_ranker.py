"""Hybrid retrieval rankers for BM25, dense, and multi-signal fusion.

HybridRRFRanker is the production path. It fuses BM25 and SPECTER2 with
weighted reciprocal rank fusion. HybridRanker keeps a legacy six-signal ensemble
for evaluation comparisons.
"""

from __future__ import annotations

import re

import numpy as np

from ..citation import format_apa

_DEFAULT_WEIGHTS = {
    "tfidf": 0.25,
    "bm25": 0.20,
    "embedding": 0.20,
    "section_aware": 0.25,
    "recency": 0.05,
    "category": 0.05,
}

_RRF_DEFAULT_WEIGHTS = {
    "bm25": 0.6,
    "embedding": 0.4,
}

_RELATION_MAP = {
    # Heuristic: assign relation type from dominant score source
    "similar": lambda scores: scores["embedding"] >= 0.7,
    "foundational": lambda scores: scores["recency"] < 0.2,
    "recent": lambda scores: scores["recency"] >= 0.8,
    "method_related": lambda scores: scores["section_aware"] > scores["embedding"],
    "same_topic": lambda scores: scores["category"] >= 0.5,
}

_RAG_PHRASE = re.compile(
    r"\bretrieval[\s-]+augmented[\s-]+generation\b",
    re.IGNORECASE,
)
_RAG_ACRONYM = re.compile(r"\bRAG\b")


def expand_query(query: str) -> str:
    """Add canonical aliases for explicit technical concepts."""
    expanded = query.strip()
    if _RAG_PHRASE.search(expanded):
        expanded += " RAG retrieval-augmented generation"
    if "scientific literature" in expanded.lower():
        expanded += " scientific papers scholarly documents"
    return expanded


def _matches_explicit_query_concept(query: str, paper: dict) -> bool:
    if not _RAG_PHRASE.search(query):
        return True
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    return bool(_RAG_PHRASE.search(text) or _RAG_ACRONYM.search(text))


def _recency_score(paper: dict, reference_year: int = 2024) -> float:
    """Return a 0-1 score based on publication year (older = 0, newest = 1)."""
    date_str = paper.get("published_date", "") or ""
    try:
        year = int(date_str[:4])
    except (ValueError, TypeError):
        return 0.5
    # Papers from last 5 years score 0.6-1.0; older papers score lower
    age = max(0, reference_year - year)
    return max(0.0, 1.0 - age * 0.1)


def _category_score(paper: dict, query_categories: list[str]) -> float:
    """Fraction of query categories that overlap with paper categories."""
    if not query_categories:
        return 0.0
    paper_cats = set(paper.get("categories", []))
    query_cats = set(query_categories)
    if not paper_cats or not query_cats:
        return 0.0
    return len(paper_cats & query_cats) / len(query_cats)


def _infer_relation(component_scores: dict[str, float]) -> str:
    if component_scores.get("recency", 0) >= 0.8:
        return "recent"
    if component_scores.get("recency", 1) < 0.2:
        return "foundational"
    if component_scores.get("category", 0) >= 0.5:
        return "same_topic"
    if component_scores.get("section_aware", 0) > component_scores.get("embedding", 0):
        return "method_related"
    return "similar"


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores to [0, 1]."""
    if not scores:
        return []
    arr = np.array(scores, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return [0.5] * len(scores)
    return list((arr - mn) / (mx - mn))


class HybridRanker:
    """Ensemble ranker that combines all retrieval signals into one ranked list."""

    def __init__(
        self,
        tfidf_retriever=None,
        bm25_retriever=None,
        embedding_retriever=None,
        section_aware_retriever=None,
        weights: dict[str, float] | None = None,
        query_categories: list[str] | None = None,
    ):
        self._tfidf = tfidf_retriever
        self._bm25 = bm25_retriever
        self._emb = embedding_retriever
        self._section = section_aware_retriever
        self._weights = weights or _DEFAULT_WEIGHTS
        self._query_categories = query_categories or []
        self._papers: list[dict] = []

    def fit(self, papers: list[dict]) -> "HybridRanker":
        self._papers = list(papers)
        if self._tfidf:
            self._tfidf.fit(papers)
        if self._bm25:
            self._bm25.fit(papers)
        if self._emb:
            self._emb.fit(papers)
        if self._section:
            self._section.fit(papers)
        return self

    def _score_map(self, query: str, top_k_each: int = 20) -> dict[str, dict[str, float]]:
        """Collect raw scores from each retriever, keyed by paper_id."""
        score_maps: dict[str, dict[str, float]] = {
            p["paper_id"]: {
                "tfidf": 0.0,
                "bm25": 0.0,
                "embedding": 0.0,
                "section_aware": 0.0,
                "recency": _recency_score(p),
                "category": _category_score(p, self._query_categories),
            }
            for p in self._papers
        }

        def _fill(retriever, key: str) -> None:
            if retriever is None:
                return
            for paper, score in retriever.retrieve(query, top_k=top_k_each):
                pid = paper["paper_id"]
                if pid in score_maps:
                    score_maps[pid][key] = score

        _fill(self._tfidf, "tfidf")
        _fill(self._bm25, "bm25")
        _fill(self._emb, "embedding")
        _fill(self._section, "section_aware")
        return score_maps

    def rank(self, query: str, top_k: int = 10) -> list[tuple[dict, float, dict]]:
        """Return (paper, final_score, component_scores) sorted by final_score."""
        scoring_query = expand_query(query)
        raw = self._score_map(scoring_query, top_k_each=len(self._papers))

        # Normalise each signal across all papers
        for key in ("tfidf", "bm25", "embedding", "section_aware"):
            vals = [raw[pid][key] for pid in raw]
            normed = _normalise(vals)
            for pid, nv in zip(raw.keys(), normed):
                raw[pid][key] = nv

        results = []
        paper_by_id = {p["paper_id"]: p for p in self._papers}
        for pid, comp in raw.items():
            final = sum(self._weights.get(k, 0) * v for k, v in comp.items())
            results.append((paper_by_id[pid], final, comp))

        results.sort(key=lambda x: x[1], reverse=True)
        matching = [
            row
            for row in results
            if _matches_explicit_query_concept(query, row[0])
        ]
        if matching:
            matching_ids = {row[0]["paper_id"] for row in matching}
            results = matching + [
                row for row in results if row[0]["paper_id"] not in matching_ids
            ]
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """Unified interface matching TfidfRetriever / BM25Retriever / EmbeddingRetriever."""
        ranked = self.rank(query, top_k=top_k)
        return [(paper, score) for paper, score, _ in ranked]

    def recommend(self, query: str, top_k: int = 10) -> list[dict]:
        """Return a list of Recommendation objects (shared contract)."""
        ranked = self.rank(query, top_k=top_k)
        recommendations = []
        for paper, score, comp in ranked:
            relation = _infer_relation(comp)
            rec = {
                "paper": paper,
                "score": round(score, 4),
                "reason": _build_reason(query, paper, comp),
                "evidence": [paper["paper_id"]],
                "apa_citation": format_apa(paper),
                "relation": relation,
            }
            recommendations.append(rec)
        return recommendations

    @property
    def name(self) -> str:
        return "hybrid_ensemble"


def _build_reason(query: str, paper: dict, comp: dict[str, float]) -> str:
    dominant = max(comp, key=lambda k: comp[k] if k not in ("recency", "category") else 0)
    explanations = {
        "embedding": "high semantic similarity between query and abstract",
        "section_aware": "strong method/results section alignment with query topic",
        "tfidf": "high keyword overlap with query terms",
        "bm25": "strong term-frequency match with query",
    }
    base = explanations.get(dominant, "combined retrieval signals")
    return f"Recommended due to {base}. Title: \"{paper['title']}\"."


def _build_rrf_reason(paper: dict, comp: dict[str, float | int]) -> str:
    bm25_rank = int(comp.get("bm25_rank", 0))
    emb_rank = int(comp.get("embedding_rank", 0))
    if bm25_rank and emb_rank:
        if bm25_rank <= emb_rank:
            base = "strong term-frequency match with query and semantic alignment"
        else:
            base = "high semantic similarity between query and abstract"
    elif bm25_rank:
        base = "strong term-frequency match with query"
    elif emb_rank:
        base = "high semantic similarity between query and abstract"
    else:
        base = "combined BM25 and semantic retrieval signals"
    return f"Recommended due to {base}. Title: \"{paper['title']}\"."


def _infer_rrf_relation(comp: dict[str, float | int]) -> str:
    bm25_rank = int(comp.get("bm25_rank", 0))
    emb_rank = int(comp.get("embedding_rank", 0))
    if emb_rank and (not bm25_rank or emb_rank < bm25_rank):
        return "similar"
    return "same_topic"


class HybridRRFRanker:
    """Production ranker: BM25 + SPECTER2 fused with weighted reciprocal rank fusion."""

    RRF_K = 60

    def __init__(
        self,
        bm25_retriever=None,
        embedding_retriever=None,
        weights: dict[str, float] | None = None,
    ):
        self._bm25 = bm25_retriever
        self._emb = embedding_retriever
        self._weights = weights or _RRF_DEFAULT_WEIGHTS
        self._papers: list[dict] = []

    def fit(self, papers: list[dict]) -> "HybridRRFRanker":
        self._papers = list(papers)
        if self._bm25:
            self._bm25.fit(papers)
        if self._emb:
            self._emb.fit(papers)
        return self

    def _rank_lists(self, query: str) -> dict[str, list[str]]:
        rank_lists: dict[str, list[str]] = {}
        for retriever, key in (
            (self._bm25, "bm25"),
            (self._emb, "embedding"),
        ):
            if retriever is None:
                continue
            results = retriever.retrieve(query, top_k=len(self._papers))
            rank_lists[key] = [p["paper_id"] for p, _ in results]
        return rank_lists

    def rank(self, query: str, top_k: int = 10) -> list[tuple[dict, float, dict]]:
        """Return (paper, rrf_score, component_ranks) sorted by rrf_score."""
        scoring_query = expand_query(query)
        rank_lists = self._rank_lists(scoring_query)

        rrf_scores: dict[str, float] = {}
        rank_by_engine: dict[str, dict[str, int]] = {key: {} for key in rank_lists}
        for key, pids in rank_lists.items():
            weight = self._weights.get(key, 0.5)
            for rank, pid in enumerate(pids, start=1):
                rrf_scores[pid] = rrf_scores.get(pid, 0.0) + weight / (self.RRF_K + rank)
                rank_by_engine[key][pid] = rank

        paper_by_id = {p["paper_id"]: p for p in self._papers}
        results = []
        for pid, score in rrf_scores.items():
            if pid not in paper_by_id:
                continue
            comp = {
                "rrf": score,
                "bm25_rank": rank_by_engine.get("bm25", {}).get(pid, 0),
                "embedding_rank": rank_by_engine.get("embedding", {}).get(pid, 0),
            }
            results.append((paper_by_id[pid], score, comp))

        results.sort(key=lambda x: x[1], reverse=True)
        matching = [
            row for row in results if _matches_explicit_query_concept(query, row[0])
        ]
        if matching:
            matching_ids = {row[0]["paper_id"] for row in matching}
            results = matching + [
                row for row in results if row[0]["paper_id"] not in matching_ids
            ]
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        ranked = self.rank(query, top_k=top_k)
        return [(paper, score) for paper, score, _ in ranked]

    def recommend(self, query: str, top_k: int = 10) -> list[dict]:
        ranked = self.rank(query, top_k=top_k)
        recommendations = []
        for paper, score, comp in ranked:
            recommendations.append(
                {
                    "paper": paper,
                    "score": round(score, 4),
                    "reason": _build_rrf_reason(paper, comp),
                    "evidence": [paper["paper_id"]],
                    "apa_citation": format_apa(paper),
                    "relation": _infer_rrf_relation(comp),
                }
            )
        return recommendations

    @property
    def name(self) -> str:
        return "hybrid_rrf"
