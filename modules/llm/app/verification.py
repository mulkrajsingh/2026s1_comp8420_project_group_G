"""Same-model LLM verification for generated outputs.

Supports a second-pass verifier call and an inline same-call instruction path.
Deterministic source-ID auditing supplements both modes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any

from .faithfulness import check_generation
from .runtime import GenerationConfig, GenerationResult, OllamaRuntime, source_ids_used

INLINE_VERIFICATION_INSTRUCTION = (
    "Before returning the final answer, verify every factual claim against the "
    "supplied evidence snippets. Remove unsupported claims, preserve valid source "
    "IDs, and do not describe this verification step."
)


def verification_enabled() -> bool:
    """Return whether the second-pass verifier is enabled via environment."""
    value = os.getenv("COMP8420_LLM_VERIFY", "1")
    return value.strip().lower() not in {"0", "false", "no", "off"}


def with_inline_verification(prompt_record: dict[str, Any]) -> dict[str, Any]:
    """Add a same-call evidence verification instruction to a prompt record."""
    updated = dict(prompt_record)
    for key in ("zero_shot_prompt", "few_shot_prompt"):
        prompt = str(updated.get(key) or "").strip()
        updated[key] = f"{prompt}\n\n{INLINE_VERIFICATION_INSTRUCTION}".strip()
    return updated


def apply_inline_verification(
    result: GenerationResult,
    *,
    pack: dict[str, Any] | None = None,
) -> tuple[GenerationResult, dict[str, Any]]:
    """Record deterministic source-ID validation for inline self-verification."""
    if result.error:
        return result, {
            "verification_used": False,
            "verification_revised": False,
            "verification_mode": "inline_same_model",
            "supported": False,
            "unsupported_claims": [],
            "revised_answer": "",
            "error": result.error,
        }
    heuristic = check_generation(result.text, pack or {})
    audit = {
        "verification_used": True,
        "verification_revised": False,
        "verification_mode": "inline_same_model",
        "supported": heuristic.get("passes_basic_faithfulness", False),
        "unsupported_claims": heuristic.get("unsupported_source_ids", []),
        "revised_answer": "",
        "error": None,
        "deterministic_source_audit": heuristic,
    }
    metadata = dict(result.run_metadata)
    metadata["llm_verification"] = audit
    return replace(result, run_metadata=metadata), audit


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence = re.search(
        r"```(?:json)?\s*\n(?P<body>.*?)\n```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        stripped = fence.group("body").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _paper_context(paper: dict[str, Any]) -> str:
    metadata = paper.get("metadata") or {}
    sections = paper.get("sections") or {}
    parts = [
        f"Title: {metadata.get('title', '')}",
        f"Abstract: {str(sections.get('abstract') or metadata.get('abstract') or '')[:1200]}",
    ]
    for key in ("method", "results", "limitations"):
        text = str(sections.get(key) or "").strip()
        if text:
            parts.append(f"{key.title()}: {text[:800]}")
    return "\n".join(parts)


def _evidence_context(pack: dict[str, Any]) -> str:
    lines = [f"Query: {pack.get('query', '')}"]
    for snippet in pack.get("evidence_snippets") or []:
        lines.append(
            f"[{snippet.get('source_id')}] {snippet.get('title', '')}: "
            f"{str(snippet.get('snippet', ''))[:400]}"
        )
    return "\n".join(lines)


def _build_verifier_prompt(
    draft: str,
    *,
    task: str,
    pack: dict[str, Any] | None,
    paper: dict[str, Any] | None,
) -> str:
    if pack is not None:
        context = _evidence_context(pack)
        scope = (
            "Verify the draft against ONLY the evidence snippets below. "
            "Remove or rewrite unsupported claims. Preserve source IDs for "
            "supported claims."
        )
    elif paper is not None:
        context = _paper_context(paper)
        scope = (
            "Verify the draft against ONLY the supplied paper fields below. "
            "Remove invented citations, experiments, or external comparisons."
        )
    else:
        context = "No external evidence pack supplied."
        scope = "Check internal consistency only; do not add new factual claims."

    return (
        "You are a verification assistant reviewing an AI draft.\n"
        f"Task: {task}\n"
        f"{scope}\n\n"
        "Return a single JSON object with keys:\n"
        '  "supported": boolean,\n'
        '  "unsupported_claims": array of strings,\n'
        '  "revised_answer": string (full corrected answer; empty string if unchanged)\n\n'
        "List the unsupported claims needed to explain the verification result.\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Draft to verify:\n"
        f"{draft}\n"
    )


def verify_with_llm(
    draft: str,
    *,
    task: str,
    model: str,
    runtime: OllamaRuntime,
    config: GenerationConfig | None = None,
    pack: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one verifier pass with the same Ollama model."""
    if not draft.strip():
        return {
            "verification_used": False,
            "verification_revised": False,
            "supported": False,
            "unsupported_claims": [],
            "revised_answer": "",
            "error": "empty_draft",
        }

    prompt = _build_verifier_prompt(
        draft,
        task=task,
        pack=pack,
        paper=paper,
    )
    verify_record = {
        "prompt_id": "llm_self_verification",
        "task": "self_verification",
        "style": "technical",
        "zero_shot_prompt": prompt,
        "few_shot_prompt": prompt,
        "input": {"parsed_paper": paper, "rag_evidence_pack": pack},
        "expected_output_contract": {"must_return_json": True},
    }
    config = config or GenerationConfig(max_new_tokens=2048)
    result = runtime.generate(verify_record, model, "zero_shot", config)
    payload = _extract_json_object(result.text or "")
    heuristic = check_generation(draft, pack or {})

    if payload is None:
        return {
            "verification_used": True,
            "verification_revised": False,
            "supported": heuristic.get("passes_basic_faithfulness", False),
            "unsupported_claims": [],
            "revised_answer": "",
            "error": result.error or "invalid_verifier_json",
            "latency_seconds": result.latency_seconds,
        }

    unsupported = [
        str(item).strip()
        for item in (payload.get("unsupported_claims") or [])
        if str(item).strip()
    ]
    revised = str(payload.get("revised_answer") or "").strip()
    supported = bool(payload.get("supported", not unsupported))
    use_revised = bool(revised) and revised != draft.strip()

    return {
        "verification_used": True,
        "verification_revised": use_revised,
        "supported": supported,
        "unsupported_claims": unsupported,
        "revised_answer": revised if use_revised else "",
        "error": result.error,
        "latency_seconds": result.latency_seconds,
    }


def apply_verification(
    result: GenerationResult,
    *,
    runtime: OllamaRuntime,
    model: str,
    config: GenerationConfig | None = None,
    pack: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
) -> tuple[GenerationResult, dict[str, Any]]:
    """Optionally revise a generation result using the verifier pass."""
    if not verification_enabled() or result.error:
        return result, {"verification_used": False, "verification_revised": False}

    audit = verify_with_llm(
        result.text,
        task=result.task,
        model=model,
        runtime=runtime,
        config=config,
        pack=pack,
        paper=paper,
    )
    if not audit.get("revised_answer"):
        metadata = dict(result.run_metadata)
        metadata["llm_verification"] = audit
        return replace(result, run_metadata=metadata), audit

    revised = audit["revised_answer"]
    metadata = dict(result.run_metadata)
    metadata["llm_verification"] = audit
    return replace(
        result,
        text=revised,
        evidence_ids_used=source_ids_used(revised),
        run_metadata=metadata,
        latency_seconds=round(
            result.latency_seconds + float(audit.get("latency_seconds") or 0.0),
            4,
        ),
    ), audit
