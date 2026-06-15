"""Unit tests for topic synthesis JSON parsing (no Ollama)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runtime import GenerationResult
from app.synthesis import (
    PARSE_FAILED_SUMMARY,
    parse_paper_review,
    parse_paper_summary,
    parse_topic_synthesis,
)


def test_parse_valid_json():
    text = (
        '{"summary": "RAG helps ground answers.", '
        '"key_findings": ["Finding one [0812.4614]"], '
        '"research_gaps": ["Need human eval"]}'
    )
    result = GenerationResult(
        text=text,
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )
    out = parse_topic_synthesis(result)
    assert out["summary"] == "RAG helps ground answers."
    assert len(out["key_findings"]) == 1
    assert len(out["research_gaps"]) == 1


def test_parse_json_prefix_noise():
    text = 'Here is the answer:\n{"summary": "Topic overview.", "key_findings": [], "research_gaps": []}'
    result = GenerationResult(
        text=text,
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )
    out = parse_topic_synthesis(result)
    assert out["summary"] == "Topic overview."


def test_parse_nested_findings_dict():
    text = (
        '{"findings": {"0812.4614": "Quantum paper discusses X.", '
        '"0705.3360": "Another finding on Y."}, "research_gaps": []}'
    )
    result = GenerationResult(
        text=text,
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )
    out = parse_topic_synthesis(result)
    assert len(out["key_findings"]) == 2
    assert "Quantum" in out["summary"]


def test_parse_markdown_fallback():
    text = (
        "### Findings\n\n"
        "#### 1. **Source ID: 0704.3395**\n"
        "- Summary: Semantic network computing model.\n\n"
        "### Assumptions\n\n"
        "- Quantum models may not be validated in practice.\n\n"
        "### Conclusion\n\n"
        "The data highlights quantum computing and AI research directions."
    )
    result = GenerationResult(
        text=text,
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )
    out = parse_topic_synthesis(result)
    assert "quantum computing" in out["summary"].lower()
    assert len(out["key_findings"]) >= 1
    assert len(out["research_gaps"]) >= 1


def test_parse_bold_heading_fallback_does_not_truncate_summary():
    introduction = (
        "The papers explain that LLM hallucinations arise when generated tokens are "
        "plausible under the model but are not supported by reliable evidence. "
        "Grounding and knowledge-boundary checks can reduce this failure mode."
    )
    result = GenerationResult(
        text=(
            f"{introduction}\n\n"
            "**Key Findings:**\n"
            "- Retrieval grounding reduces unsupported claims [2410.02825].\n"
            "- Knowledge-boundary checks expose uncertain answers [2405.14383].\n\n"
            "**Research Gaps:**\n"
            "- Cross-domain evaluation remains limited."
        ),
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )

    out = parse_topic_synthesis(result)

    assert out["summary"] == introduction
    assert out["key_findings"] == [
        "Retrieval grounding reduces unsupported claims [2410.02825].",
        "Knowledge-boundary checks expose uncertain answers [2405.14383].",
    ]
    assert out["research_gaps"] == ["Cross-domain evaluation remains limited."]


def test_parse_plain_prose_verifier_revision_as_summary():
    text = (
        "The papers discuss how LLMs access and process data through retrieval, "
        "fine-tuning, and structured extraction methods. Retrieved context can "
        "supply relevant external evidence, while task-specific training changes "
        "the knowledge and behavior encoded in the model."
    )
    result = GenerationResult(
        text=text,
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )

    out = parse_topic_synthesis(result)

    assert out == {
        "summary": text,
        "key_findings": [],
        "research_gaps": [],
    }


def test_parse_failure():
    result = GenerationResult(
        text="{",
        backend="ollama",
        model="qwen3:8b",
        prompt_id="P02",
        task="topic_search_synthesis",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )
    out = parse_topic_synthesis(result)
    assert out["summary"] == PARSE_FAILED_SUMMARY
    assert out["key_findings"] == []


def test_parse_paper_summary_keeps_clean_text_and_extracts_sections():
    result = GenerationResult(
        text=(
            "## Scope\nA scoped paper.\n\n"
            "## Core Contribution\nA new method.\n\n"
            "## Method\nTechnical details.\n\n"
            "## Results\n- Improves the main metric.\n\n"
            "## Limitations\n- No cross-domain evaluation."
        ),
        backend="ollama",
        model="qwen3:8b",
        prompt_id="paper_only_summary",
        task="uploaded_paper_summary",
        strategy="zero_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )

    out = parse_paper_summary(result)
    assert out["summary"].startswith("## Scope")
    assert "Core contribution: A new method." in out["key_findings"]
    assert "Improves the main metric." in out["key_findings"]
    assert out["research_gaps"] == ["No cross-domain evaluation."]


def test_parse_paper_review_returns_only_generated_review():
    result = GenerationResult(
        text="```markdown\n## Strengths\nGrounded strength.\n```",
        backend="ollama",
        model="qwen3:8b",
        prompt_id="paper_only_peer_review",
        task="peer_review_critique",
        strategy="few_shot",
        latency_seconds=1.0,
        error=None,
        evidence_ids_used=[],
        run_metadata={},
    )

    assert parse_paper_review(result) == {
        "peer_review": "## Strengths\nGrounded strength."
    }


if __name__ == "__main__":
    test_parse_valid_json()
    test_parse_json_prefix_noise()
    test_parse_failure()
    print("ok")
