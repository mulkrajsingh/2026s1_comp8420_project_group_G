"""Shared dataclasses exchanged between integration modules.

The five contract types are ``PaperRecord`` (dataset), ``ParsedPaper`` (PDF-NLP),
``RagEvidencePack`` and ``Recommendation`` (retrieval), and ``AnalysisResult``
(the integrated output rendered by the CLI, API, and web UI). The canonical
contract definitions live in this module; ``ParsedPaper`` validation is also
implemented in ``modules/llm/app/schemas.py::validate_parsed_paper``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CANONICAL_SECTION_KEYS = (
    "abstract",
    "introduction",
    "method",
    "results",
    "conclusion",
)
EMPTY_ENTITIES = {
    "methods": [],
    "datasets": [],
    "tasks": [],
    "metrics": [],
    "institutions": [],
}


def _from(cls, data: dict[str, Any]):
    """Build a dataclass from a dict, ignoring unknown keys (forward-compatible)."""
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in data.items() if key in fields})


def _empty_sections() -> dict[str, str]:
    return {key: "" for key in CANONICAL_SECTION_KEYS}


def _empty_entities() -> dict[str, list[str]]:
    return {key: [] for key in EMPTY_ENTITIES}


def _empty_metadata(paper_id: str = "") -> dict[str, Any]:
    return {
        "paper_id": paper_id,
        "title": "",
        "abstract": "",
        "authors": [],
        "categories": [],
        "published_date": None,
        "venue": None,
        "doi": None,
        "arxiv_id": None,
        "url": None,
        "source": "uploaded_pdf",
    }


# --- Dataset module -------------------------------------------------------
@dataclass
class PaperRecord:
    """One arXiv-derived paper record from the dataset module."""

    id: str
    title: str
    abstract: str
    authors: str = ""               # raw arXiv author string
    categories: str = ""            # space-separated; first is primary
    update_date: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _from(cls, data)


# --- PDF-NLP module -------------------------------------------------------
@dataclass
class ParsedPaper:
    """Standard ParsedPaper shape plus optional deterministic PDF-NLP analysis."""

    metadata: dict[str, Any] = field(default_factory=_empty_metadata)
    sections: dict[str, str] = field(default_factory=_empty_sections)
    references: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=_empty_entities)
    analysis: dict[str, Any] = field(default_factory=dict)
    source_path: str | None = None

    @property
    def paper_id(self) -> str:
        return str(self.metadata.get("paper_id") or "")

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or "")

    @property
    def abstract(self) -> str:
        return str(
            self.metadata.get("abstract")
            or self.sections.get("abstract")
            or ""
        )

    @property
    def authors(self) -> list[str]:
        authors = self.metadata.get("authors") or []
        if isinstance(authors, list):
            return [str(a) for a in authors]
        if isinstance(authors, str) and authors.strip():
            return [authors]
        return []

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": dict(self.metadata),
            "sections": dict(self.sections),
            "references": list(self.references),
            "keywords": list(self.keywords),
            "entities": {key: list(self.entities.get(key, [])) for key in EMPTY_ENTITIES},
            **({"analysis": dict(self.analysis)} if self.analysis else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedPaper":
        if "metadata" in data and "sections" in data:
            metadata = dict(data.get("metadata") or {})
            sections = _empty_sections()
            raw_sections = data.get("sections") or {}
            if isinstance(raw_sections, dict):
                for key, value in raw_sections.items():
                    if isinstance(value, str):
                        sections[str(key)] = value
            entities = _empty_entities()
            raw_entities = data.get("entities") or {}
            if isinstance(raw_entities, dict):
                for key in EMPTY_ENTITIES:
                    value = raw_entities.get(key) or []
                    entities[key] = list(value) if isinstance(value, list) else []
            return cls(
                metadata=metadata,
                sections=sections,
                references=list(data.get("references") or []),
                keywords=list(data.get("keywords") or []),
                entities=entities,
                analysis=dict(data.get("analysis") or {}),
                source_path=data.get("source_path"),
            )
        raise ValueError(
            "ParsedPaper.from_dict requires canonical keys 'metadata' and 'sections'"
        )

# --- Retrieval module -----------------------------------------------------
@dataclass
class RagEvidencePack:
    """Retrieved snippets that ground synthesis for a query."""

    query: str
    snippets: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str = "offline"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _from(cls, data)


@dataclass
class Recommendation:
    """Ranked related papers for a query or parsed title."""

    query: str
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _from(cls, data)


# --- Returned result ------------------------------------------------------
@dataclass
class AnalysisResult:
    """The integrated output. Field set mirrors the frontend results panel
    (metadata, summary, key findings, gaps, recommendations, APA citations,
    evidence, peer review) so the UI maps 1:1 onto this object."""
    input_type: str                 # "pdf" | "topic"
    input_ref: str                  # file path or query text
    metadata: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    research_gaps: list[str] = field(default_factory=list)
    recommended_papers: list[dict[str, Any]] = field(default_factory=list)
    apa_citations: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    peer_review: str | None = None
    paper_analysis: dict[str, Any] = field(default_factory=dict)
    # transparency flags: privacy disclosure, retrieval mode, model toggles
    flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return _from(cls, data)
