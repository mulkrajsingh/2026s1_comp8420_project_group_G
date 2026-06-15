"""Offline BM25 retrieval and RagEvidencePack construction for project arXiv RAG."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from rank_bm25 import BM25Okapi

from app.schemas import validate_rag_evidence_pack

from lora_dataset.io import truncate


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _aps_authors(authors: list[str]) -> str:
    cleaned = [str(author).strip() for author in authors if str(author).strip()]
    if not cleaned:
        return "Unknown author"
    formatted = [_aps_author_name(author) for author in cleaned]
    if len(formatted) == 1:
        return formatted[0]
    if len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    return f"{formatted[0]} et al."


def _aps_author_name(author: str) -> str:
    if " and " in author.lower():
        return author
    if "," in author:
        surname, given = (part.strip() for part in author.split(",", 1))
        tokens = given.split()
    else:
        tokens = author.split()
        if len(tokens) < 2:
            return author
        surname = tokens.pop()
    initials = " ".join(f"{token[0].upper()}." for token in tokens if token)
    return f"{initials} {surname}".strip()


def _publication_year(paper: dict[str, Any]) -> str | None:
    value = str(paper.get("published_date") or "")
    try:
        return str(datetime.strptime(value, "%Y-%m-%d").year)
    except ValueError:
        return None


def aps_citation(paper: dict[str, Any]) -> str:
    """Format available metadata in APS-style order without filling missing fields."""
    authors = paper.get("authors") or []
    author_part = _aps_authors(authors)
    year = _publication_year(paper)
    title = paper.get("title") or "Untitled"
    venue = str(paper.get("venue") or "").strip()
    doi = str(paper.get("doi") or "").strip()
    arxiv_id = str(paper.get("arxiv_id") or "").strip()
    suffix = f" ({year})" if year and year not in venue else ""
    if venue:
        citation = f"{author_part}, {title}, {venue}{suffix}."
    elif arxiv_id:
        citation = f"{author_part}, {title}, arXiv:{arxiv_id}{suffix}."
    else:
        citation = f"{author_part}, {title}{suffix}."
    if doi:
        citation = f"{citation[:-1]}, https://doi.org/{doi}."
    return citation


def rank_papers(
    query: str,
    papers: list[dict[str, Any]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[int, float]]:
    query_tokens = tokenize(query)
    if not query_tokens or not papers:
        return []

    corpus = [
        tokenize(f"{paper.get('title', '')} {paper.get('abstract', '')}")
        for paper in papers
    ]
    if not any(corpus):
        return []

    scores = BM25Okapi(corpus, k1=k1, b=b).get_scores(query_tokens)
    ranked = [
        (doc_id, float(score))
        for doc_id, score in enumerate(scores)
        if score > 0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def relation_label(score: float, rank: int) -> str:
    if rank == 0:
        return "method_related"
    if score > 0.85:
        return "similar"
    if rank <= 2:
        return "foundational"
    return "recent"


def build_rag_evidence_pack(
    query: str,
    papers: list[dict[str, Any]],
    *,
    top_k: int = 5,
    retrieval_mode: str = "offline",
) -> dict[str, Any]:
    ranked = rank_papers(query, papers)[:top_k]
    if not ranked:
        raise ValueError(f"No retrieval hits for query: {query!r}")

    max_score = ranked[0][1]
    candidates: list[dict[str, Any]] = []
    evidence_snippets: list[dict[str, Any]] = []

    for rank, (doc_id, raw_score) in enumerate(ranked):
        paper = papers[doc_id]
        source_id = f"S{rank + 1}"
        normalized = raw_score / max_score if max_score else 0.0
        snippet_text = truncate(paper.get("abstract") or paper.get("title") or "", 420)
        evidence_snippets.append(
            {
                "source_id": source_id,
                "title": paper.get("title", ""),
                "snippet": snippet_text,
                "metadata": {
                    "year": (paper.get("published_date") or "")[:4] or None,
                    "authors": paper.get("authors") or [],
                    "venue": paper.get("venue"),
                    "doi": paper.get("doi"),
                    "categories": paper.get("categories") or [],
                    "citation_count": None,
                    "url": paper.get("url"),
                },
            }
        )
        candidates.append(
            {
                "paper": paper,
                "score": round(min(0.99, 0.55 + normalized * 0.44), 2),
                "reason": (
                    f"Retrieved for query '{query}' because title/abstract overlap suggests "
                    f"relevance to the user's topic (rank {rank + 1})."
                ),
                "evidence": [source_id],
                # Shared schema keeps this historical key; the value is APS-formatted.
                "apa_citation": aps_citation(paper),
                "relation": relation_label(normalized, rank),
            }
        )

    pack = {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "candidates": candidates,
        "evidence_snippets": evidence_snippets,
    }
    validate_rag_evidence_pack(pack)
    return pack
