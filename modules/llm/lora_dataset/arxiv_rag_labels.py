"""Template assistant targets for project arXiv RAG training rows.

Produces faithfulness-safe labels from retrieval packs without calling a teacher model.
"""

from __future__ import annotations

import json
import re
from typing import Any

def _source_ids(pack: dict[str, Any]) -> list[str]:
    return [snippet["source_id"] for snippet in pack.get("evidence_snippets", [])]


def _reference_lines(pack: dict[str, Any], limit: int = 5) -> list[str]:
    return [
        f"[{index}] {candidate['apa_citation']}"
        for index, candidate in enumerate(pack["candidates"][:limit], start=1)
    ]


def label_topic_synthesis(pack: dict[str, Any]) -> str:
    snippets = pack["evidence_snippets"][:3]
    payload = {
        "summary": (
            f"The retrieved literature addresses {pack['query']}. "
            f"The leading abstract provides the strongest direct evidence "
            f"[{snippets[0]['source_id']}][1]."
        ),
        "key_findings": [
            f"{snippet['title']}: {snippet['snippet']} "
            f"[{snippet['source_id']}][{index}]"
            for index, snippet in enumerate(snippets, start=1)
        ],
        "research_gaps": [
            "Full-text methods and user-study outcomes are not established by "
            "abstract-level evidence alone."
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def label_citation_recommendation(pack: dict[str, Any]) -> str:
    lines = ["## Recommended citations"]
    for index, candidate in enumerate(pack["candidates"][:5], start=1):
        evidence = ", ".join(f"[{sid}]" for sid in candidate["evidence"])
        lines.append(f"- [{index}] {candidate['paper']['title']} {evidence}")
    lines.extend(["", "## References", *_reference_lines(pack)])
    return "\n".join(lines)


def label_research_gaps(pack: dict[str, Any]) -> str:
    first = pack["evidence_snippets"][0]["source_id"]
    second = pack["evidence_snippets"][1]["source_id"] if len(pack["evidence_snippets"]) > 1 else first
    return "\n".join(
        [
            "## Evidence-supported gaps",
            f"- Compare section-aware retrieval with whole-abstract retrieval [{first}][1].",
            f"- Add faithfulness audits for generated synthesis [{second}][2].",
            "",
            "## Assumptions",
            "User-satisfaction gains require a dedicated study not present in the supplied snippets.",
            "",
            "## References",
            *_reference_lines(pack, limit=2),
        ]
    )


def label_explain_top_recommendation(pack: dict[str, Any]) -> str:
    top = pack["candidates"][0]
    sid = top["evidence"][0]
    return "\n".join(
        [
            "## Why this paper is recommended",
            f"- {top['paper']['title']} is ranked first with score {top['score']}.",
            f"- Reason: {top['reason']}",
            f"- Supporting evidence: [{sid}][1]",
            "",
            "## References",
            f"[1] {top['apa_citation']}",
        ]
    )


def label_follow_up(pack: dict[str, Any], *, question: str) -> str:
    snippet = pack["evidence_snippets"][0]
    sid = snippet["source_id"]
    return "\n".join(
        [
            f"Question: {question}",
            f"Answer: Based on [{sid}][1], {snippet['snippet']}",
            "If the pack lacks direct evidence, the answer should stay cautious; here only supplied snippets are used.",
            "",
            "## References",
            f"[1] {pack['candidates'][0]['apa_citation']}",
        ]
    )


def label_beginner(pack: dict[str, Any]) -> str:
    snippet = pack["evidence_snippets"][0]
    return "\n".join(
        [
            "## Beginner explanation",
            (
                f"The retrieved set addresses '{pack['query']}'. "
                f"A central paper [{snippet['source_id']}][1] explains: {snippet['snippet']}"
            ),
            "Claims about external papers use source IDs; details not in snippets are not asserted.",
            "",
            "## References",
            f"[1] {pack['candidates'][0]['apa_citation']}",
        ]
    )


TASK_BUILDERS = {
    "topic_search_synthesis": lambda pack, _: label_topic_synthesis(pack),
    "citation_recommendation": lambda pack, _: label_citation_recommendation(pack),
    "research_gap_identification": lambda pack, _: label_research_gaps(pack),
    "explain_recommendation": lambda pack, _: label_explain_top_recommendation(pack),
    "follow_up_qa": lambda pack, _: label_follow_up(
        pack, question="What is the main idea of the top retrieved paper?"
    ),
    "beginner_explanation": lambda pack, _: label_beginner(pack),
}


def build_assistant_label(pack: dict[str, Any], task: str) -> str | None:
    """Return the template assistant target for a RAG task, if supported."""
    builder = TASK_BUILDERS.get(task)
    if builder is None:
        return None
    text = builder(pack, task)
    available = set(_source_ids(pack))
    used = set(re.findall(r"\[(S\d+)\]", text))
    if not used or not used.issubset(available):
        return None
    return text
