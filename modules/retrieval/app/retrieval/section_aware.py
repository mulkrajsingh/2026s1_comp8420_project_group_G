"""Section-aware abstract representation retrieval.

Inspired by Xu et al. (2025) "Research Paper Recommender System by Considering
Users' Information Seeking Behaviors." We classify abstract sentences into
background / method / results roles and build separate embeddings for each role.

Combined similarity score (configurable weights):
  whole_abstract * 0.4 + method * 0.3 + results * 0.2 + background * 0.1
"""

from __future__ import annotations

import re

import numpy as np

# Rule-based cue lists for sentence role classification (Week 1 rule-based technique)
_METHOD_CUES = [
    "we propose", "we present", "our method", "we introduce", "we develop",
    "our approach", "this paper presents", "we describe", "in this paper, we",
    "we design", "our model", "we formulate",
]
_RESULTS_CUES = [
    "we show", "outperforms", "achieves", "improves", "our experiments",
    "results show", "we demonstrate", "state-of-the-art", "our model achieves",
    "we report", "we obtain", "significantly better", "competitive with",
    "we evaluate",
]
_BACKGROUND_CUES = [
    "recent work", "has been widely", "existing methods", "previous work",
    "traditionally", "it is well known", "in recent years", "have shown",
    "is an important", "remains challenging", "despite", "however,",
    "current approaches",
]


def classify_sentence(sentence: str) -> str:
    """Return 'background' | 'method' | 'results' | 'other' for one sentence."""
    s = sentence.lower()
    method_hits = sum(1 for cue in _METHOD_CUES if cue in s)
    results_hits = sum(1 for cue in _RESULTS_CUES if cue in s)
    background_hits = sum(1 for cue in _BACKGROUND_CUES if cue in s)

    if method_hits >= results_hits and method_hits >= background_hits and method_hits > 0:
        return "method"
    if results_hits >= method_hits and results_hits >= background_hits and results_hits > 0:
        return "results"
    if background_hits > 0:
        return "background"
    return "other"


def split_abstract_by_role(abstract: str) -> dict[str, list[str]]:
    """Split abstract text into sentences grouped by inferred role."""
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())
    by_role: dict[str, list[str]] = {
        "background": [], "method": [], "results": [], "other": []
    }
    for sent in sentences:
        sent = sent.strip()
        if sent:
            by_role[classify_sentence(sent)].append(sent)
    return by_role


def _section_texts(paper: dict) -> dict[str, str]:
    """Build the four text variants for a paper (whole + three role-specific)."""
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    by_role = split_abstract_by_role(abstract)

    return {
        "whole": f"{title} {abstract}",
        "method": f"{title} {' '.join(by_role['method'])}",
        "results": f"{title} {' '.join(by_role['results'])}",
        "background": f"{title} {' '.join(by_role['background'])}",
    }


class SectionAwareRetriever:
    """Retrieval using per-section abstract embeddings with configurable score weights.

    Default weights follow the plan: whole(0.4) + method(0.3) + results(0.2) + background(0.1).
    """

    DEFAULT_WEIGHTS = {"whole": 0.4, "method": 0.3, "results": 0.2, "background": 0.1}

    def __init__(self, embedding_model=None, weights: dict[str, float] | None = None):
        if embedding_model is None:
            from .embeddings import EmbeddingRetriever
            self._emb = EmbeddingRetriever()
        else:
            self._emb = embedding_model
        self._weights = weights or self.DEFAULT_WEIGHTS
        # paper_id -> {section: np.ndarray}
        self._section_embs: dict[str, dict[str, np.ndarray]] = {}
        self._papers: list[dict] = []

    def fit(self, papers: list[dict], show_progress: bool = True) -> "SectionAwareRetriever":
        self._papers = list(papers)
        model = self._emb._model

        # Build all four text variants for every paper at once (efficient batching)
        sections = ["whole", "method", "results", "background"]
        all_texts = {s: [] for s in sections}
        for paper in self._papers:
            texts = _section_texts(paper)
            for s in sections:
                all_texts[s].append(texts[s])

        encoded: dict[str, np.ndarray] = {}
        for s in sections:
            encoded[s] = model.encode(
                all_texts[s],
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        for i, paper in enumerate(self._papers):
            self._section_embs[paper["paper_id"]] = {
                s: encoded[s][i] for s in sections
            }
        return self

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        model = self._emb._model
        sections = ["whole", "method", "results", "background"]

        # Encode query once per section (same text, different weight application)
        q_emb = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        scores = []
        for paper in self._papers:
            pid = paper["paper_id"]
            embs = self._section_embs[pid]
            score = sum(
                self._weights[s] * float(np.dot(q_emb, embs[s]))
                for s in sections
            )
            scores.append((paper, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def name(self) -> str:
        return "section_aware"
