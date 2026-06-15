"""Evidence and citation-safety checks for generated LLM outputs.

Compares cited source IDs and known APA strings against a ``RagEvidencePack``.
"""

from __future__ import annotations

import re
from typing import Any


_SOURCE_TOKEN = re.compile(
    r"^(?:S\d+|\d{4}\.\d{4,5}(?:v\d+)?|"
    r"(?=[A-Za-z0-9._:-]*[_:-])[A-Za-z0-9][A-Za-z0-9._:-]+)$"
)


def evidence_source_ids(pack: dict[str, Any]) -> set[str]:
    """Return source IDs available in an evidence pack."""
    return {str(snippet["source_id"]) for snippet in pack.get("evidence_snippets", [])}


def output_source_ids(text: str) -> set[str]:
    """Extract source IDs and arXiv-style identifiers from generated text."""
    source_ids = set(
        re.findall(
            r"(?<![A-Za-z0-9._:-])\d{4}\.\d{4,5}(?:v\d+)?(?![A-Za-z0-9._:-])",
            text,
        )
    )
    for bracketed in re.findall(r"\[([^\[\]\n]{1,160})\]", text):
        for token in re.split(r"\s*[,;]\s*", bracketed):
            candidate = token.strip()
            if _SOURCE_TOKEN.fullmatch(candidate):
                source_ids.add(candidate)
    return source_ids


def citation_strings(pack: dict[str, Any]) -> set[str]:
    """Return non-empty APA citation strings from recommendation metadata."""
    return {
        str(candidate.get("apa_citation", "")).strip()
        for candidate in pack.get("candidates", [])
        if str(candidate.get("apa_citation", "")).strip()
    }


def check_generation(text: str, pack: dict[str, Any]) -> dict[str, Any]:
    """Audit source-ID usage and basic citation grounding for one output."""
    available = evidence_source_ids(pack)
    used = output_source_ids(text)
    unsupported = sorted(used.difference(available))
    known_citations = citation_strings(pack)
    citation_mentions = [citation for citation in known_citations if citation in text]
    return {
        "available_source_ids": sorted(available),
        "used_source_ids": sorted(used),
        "unsupported_source_ids": unsupported,
        "uses_at_least_one_source": bool(used),
        "all_used_sources_exist": not unsupported,
        "metadata_citations_found": len(citation_mentions),
        "known_metadata_citations": len(known_citations),
        "passes_basic_faithfulness": bool(used) and not unsupported,
    }
