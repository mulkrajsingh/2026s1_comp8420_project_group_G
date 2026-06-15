"""Shared contract validators for integration JSON objects.

The project schemas are defined in Markdown rather than code. These validators
enforce required keys without adding a pydantic dependency.
"""

from __future__ import annotations

from typing import Any


PAPER_RECORD_KEYS = {
    "paper_id",
    "title",
    "abstract",
    "authors",
    "categories",
    "published_date",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "source",
}

SECTION_KEYS = {"abstract", "introduction", "method", "results", "conclusion"}
ENTITY_KEYS = {"methods", "datasets", "tasks", "metrics", "institutions"}
RECOMMENDATION_KEYS = {"paper", "score", "reason", "evidence", "apa_citation", "relation"}
RAG_PACK_KEYS = {"query", "retrieval_mode", "candidates", "evidence_snippets"}
ANALYSIS_RESULT_KEYS = {
    "input_type",
    "paper",
    "query",
    "summary",
    "key_findings",
    "research_gaps",
    "recommendations",
    "peer_review",
    "evidence_sources",
}


class SchemaError(ValueError):
    """Raised when a shared-contract object is missing required fields."""


def _missing(record: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required.difference(record.keys()))


def validate_paper_record(record: dict[str, Any]) -> None:
    """Validate a ``PaperRecord`` object."""
    if not isinstance(record, dict):
        raise SchemaError("PaperRecord must be a JSON object")
    missing = _missing(record, PAPER_RECORD_KEYS)
    if missing:
        raise SchemaError(f"PaperRecord missing keys: {missing}")
    if not isinstance(record["authors"], list):
        raise SchemaError("PaperRecord.authors must be a list")
    if not isinstance(record["categories"], list):
        raise SchemaError("PaperRecord.categories must be a list")


def validate_parsed_paper(record: dict[str, Any]) -> None:
    """Validate a ``ParsedPaper`` object."""
    if not isinstance(record, dict):
        raise SchemaError("ParsedPaper must be a JSON object")
    if "metadata" not in record or "sections" not in record:
        raise SchemaError("ParsedPaper requires metadata and sections")
    if not isinstance(record["metadata"], dict):
        raise SchemaError("ParsedPaper.metadata must be a JSON object")
    if not isinstance(record["sections"], dict):
        raise SchemaError("ParsedPaper.sections must be a JSON object")
    validate_paper_record(record["metadata"])
    missing_sections = _missing(record["sections"], SECTION_KEYS)
    if missing_sections:
        raise SchemaError(f"ParsedPaper.sections missing keys: {missing_sections}")
    if "entities" not in record:
        raise SchemaError("ParsedPaper requires entities")
    if not isinstance(record["entities"], dict):
        raise SchemaError("ParsedPaper.entities must be a JSON object")
    missing_entities = _missing(record["entities"], ENTITY_KEYS)
    if missing_entities:
        raise SchemaError(f"ParsedPaper.entities missing keys: {missing_entities}")
    if "analysis" in record and not isinstance(record["analysis"], dict):
        raise SchemaError("ParsedPaper.analysis must be a JSON object when present")


def validate_recommendation(record: dict[str, Any]) -> None:
    """Validate a ``Recommendation`` object."""
    missing = _missing(record, RECOMMENDATION_KEYS)
    if missing:
        raise SchemaError(f"Recommendation missing keys: {missing}")
    validate_paper_record(record["paper"])
    if not isinstance(record["evidence"], list):
        raise SchemaError("Recommendation.evidence must be a list of source ids")


def validate_rag_evidence_pack(record: dict[str, Any]) -> None:
    """Validate a ``RagEvidencePack`` and recommendation evidence links."""
    missing = _missing(record, RAG_PACK_KEYS)
    if missing:
        raise SchemaError(f"RagEvidencePack missing keys: {missing}")
    for candidate in record["candidates"]:
        validate_recommendation(candidate)
    source_ids = {snippet.get("source_id") for snippet in record["evidence_snippets"]}
    for candidate in record["candidates"]:
        for source_id in candidate["evidence"]:
            if source_id not in source_ids:
                raise SchemaError(f"Recommendation cites missing evidence source: {source_id}")


def validate_analysis_result(record: dict[str, Any]) -> None:
    """Validate an ``AnalysisResult`` object."""
    missing = _missing(record, ANALYSIS_RESULT_KEYS)
    if missing:
        raise SchemaError(f"AnalysisResult missing keys: {missing}")
    for recommendation in record["recommendations"]:
        validate_recommendation(recommendation)
