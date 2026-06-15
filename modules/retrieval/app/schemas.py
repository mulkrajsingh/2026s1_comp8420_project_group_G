"""Validators for PaperRecord, Recommendation, and RagEvidencePack shapes.

Field names and allowed values follow the integration contracts in
``integration/app/contracts.py``.
"""

from __future__ import annotations

from typing import Any

PAPER_RECORD_KEYS = {
    "paper_id", "title", "abstract", "authors", "categories",
    "published_date", "venue", "doi", "arxiv_id", "url", "source",
}
RECOMMENDATION_KEYS = {"paper", "score", "reason", "evidence", "apa_citation", "relation"}
RAG_PACK_KEYS = {"query", "retrieval_mode", "candidates", "evidence_snippets"}
VALID_RELATIONS = {"similar", "foundational", "recent", "method_related", "missing_citation", "same_topic"}
VALID_MODES = {"offline", "cached_live", "live", "hybrid"}


class SchemaError(ValueError):
    """Raised when a record fails contract validation."""


def _missing(record: dict[str, Any], required: set[str]) -> list[str]:
    return sorted(required.difference(record.keys()))


def validate_paper_record(record: dict[str, Any]) -> None:
    """Check required PaperRecord keys and list field types."""
    missing = _missing(record, PAPER_RECORD_KEYS)
    if missing:
        raise SchemaError(f"PaperRecord missing keys: {missing}")
    if not isinstance(record["authors"], list):
        raise SchemaError("PaperRecord.authors must be a list")
    if not isinstance(record["categories"], list):
        raise SchemaError("PaperRecord.categories must be a list")


def validate_recommendation(record: dict[str, Any]) -> None:
    """Check Recommendation fields and nested PaperRecord shape."""
    missing = _missing(record, RECOMMENDATION_KEYS)
    if missing:
        raise SchemaError(f"Recommendation missing keys: {missing}")
    validate_paper_record(record["paper"])
    if not isinstance(record["evidence"], list):
        raise SchemaError("Recommendation.evidence must be a list of source ids")
    if record["relation"] not in VALID_RELATIONS:
        raise SchemaError(f"Recommendation.relation must be one of {VALID_RELATIONS}")


def validate_rag_evidence_pack(record: dict[str, Any]) -> None:
    """Check RagEvidencePack fields and evidence source linkage."""
    missing = _missing(record, RAG_PACK_KEYS)
    if missing:
        raise SchemaError(f"RagEvidencePack missing keys: {missing}")
    if record["retrieval_mode"] not in VALID_MODES:
        raise SchemaError(f"retrieval_mode must be one of {VALID_MODES}")
    for candidate in record["candidates"]:
        validate_recommendation(candidate)
    source_ids = {s.get("source_id") for s in record["evidence_snippets"]}
    for candidate in record["candidates"]:
        for sid in candidate["evidence"]:
            if sid not in source_ids:
                raise SchemaError(f"Candidate references missing evidence source: {sid}")
