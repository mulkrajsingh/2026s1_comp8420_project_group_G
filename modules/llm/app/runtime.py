"""Runtime backend for local Ollama generation.

Selects token and reasoning budgets per task, compacts prompt inputs, and records
latency and source-ID usage for each ``/api/generate`` call.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from . import ollama_transport
from .faithfulness import output_source_ids


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_ADAPTER_MODEL = "qwen3-research-lora:latest"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
QWEN3_MAX_CONTEXT_TOKENS = 40960
TECHNICAL_CONTEXT_TOKENS = 16384
LONG_CONTEXT_TOKENS = 24576
TECHNICAL_MAX_NEW_TOKENS = 8192
LONG_CONTEXT_MAX_NEW_TOKENS = 4096
LONG_CONTEXT_THINKING_MAX_NEW_TOKENS = 4096
CHIT_CHAT_CONTEXT_TOKENS = 8192
CHIT_CHAT_MAX_NEW_TOKENS = 512
TECHNICAL_TIMEOUT_SECONDS = 600
CHIT_CHAT_TIMEOUT_SECONDS = 180
STRUCTURED_DIRECT_TASKS = frozenset(
    {
        "beginner_explanation",
        "citation_recommendation",
        "paper_recommendation",
        "topic_search_synthesis",
        "react_tool_plan",
        "self_verification",
    }
)
STRUCTURED_TASK_TOKEN_CAPS = {
    "beginner_explanation": 768,
    "citation_recommendation": 1024,
    "paper_recommendation": 2048,
    "topic_search_synthesis": 768,
    "react_tool_plan": 128,
    "self_verification": 2048,
}
BOUNDED_REASONING_TASK_TOKENS = {
    "research_gap_identification": 4096,
}

# Tasks that embed a full ParsedPaper. Use Ollama thinking with a raised output
# budget so reasoning can complete before the visible answer.
LONG_CONTEXT_THINKING_TASKS = frozenset(
    {"uploaded_paper_summary", "paper_question_answer", "peer_review_critique"}
)

# Alias kept for tests and policy branches that refer to long-context PDF tasks.
LONG_CONTEXT_TASKS = LONG_CONTEXT_THINKING_TASKS


def _env_context_tokens(env_var: str, default: int) -> int:
    raw = os.getenv(env_var)
    if raw is None or raw == "":
        return default
    return int(raw)


def _ollama_keep_alive() -> str | int:
    value = os.getenv(
        "COMP8420_OLLAMA_KEEP_ALIVE",
        DEFAULT_OLLAMA_KEEP_ALIVE,
    )
    if value in {"-1", "0"}:
        return int(value)
    return value


@dataclass(frozen=True)
class ModelVariant:
    """Describes one deployable Ollama model tag used in comparisons."""

    name: str
    role: str
    quantization: str
    adapter: str
    notes: str
    ollama_tag: str
    is_adapter: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling parameters passed through to Ollama ``options``."""

    temperature: float = 0.2
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    top_p: float = 0.9


@dataclass(frozen=True)
class GenerationPolicy:
    """Resolved thinking, context, timeout, and output budgets for one prompt."""

    thinking_enabled: bool
    context_window: int
    max_new_tokens: int
    timeout_seconds: int
    reason: str


@dataclass(frozen=True)
class GenerationResult:
    """Normalized result from one Ollama generation call."""

    text: str
    backend: str
    model: str
    prompt_id: str
    task: str
    strategy: str
    latency_seconds: float
    error: str | None
    evidence_ids_used: list[str]
    run_metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_VARIANTS = [
    ModelVariant(
        name="qwen3_8b",
        role="local foundation model",
        quantization="fp16_or_q8_when_available",
        adapter="none",
        notes="Ollama qwen3:8b; HF Qwen/Qwen3-8B for training.",
        ollama_tag=DEFAULT_OLLAMA_MODEL,
    ),
    ModelVariant(
        name="qwen3_8b_lora",
        role="team-trained LoRA/QLoRA adapter",
        quantization="qlora_4bit_training_then_adapter_inference",
        adapter="models/adapters/research_lora_adapter/",
        notes="Merge adapter into qwen3:8b and deploy as qwen3-research-lora:latest.",
        ollama_tag=DEFAULT_OLLAMA_ADAPTER_MODEL,
        is_adapter=True,
    ),
]


def source_ids_used(text: str) -> list[str]:
    """Return sorted source IDs cited in generated text."""
    return sorted(output_source_ids(text))


def build_runtime(ollama_host: str = DEFAULT_OLLAMA_HOST) -> "OllamaRuntime":
    """Construct an Ollama runtime client for the given host URL."""
    return OllamaRuntime(ollama_host)


def generation_policy_for_record(
    prompt_record: dict[str, Any],
    config: GenerationConfig,
) -> GenerationPolicy:
    """Choose reasoning and token budgets from the resolved task and intent."""
    task = str(prompt_record.get("task", ""))
    query_analysis = prompt_record.get("query_analysis") or {}
    is_chit_chat = (
        task == "direct_text_chat"
        and query_analysis.get("intent") == "chit_chat"
    )
    if is_chit_chat:
        return GenerationPolicy(
            thinking_enabled=False,
            context_window=_env_context_tokens(
                "COMP8420_OLLAMA_NUM_CTX_CHIT_CHAT",
                CHIT_CHAT_CONTEXT_TOKENS,
            ),
            max_new_tokens=min(config.max_new_tokens, CHIT_CHAT_MAX_NEW_TOKENS),
            timeout_seconds=CHIT_CHAT_TIMEOUT_SECONDS,
            reason="classified_chit_chat",
        )
    if task in STRUCTURED_DIRECT_TASKS:
        return GenerationPolicy(
            thinking_enabled=False,
            context_window=_env_context_tokens(
                "COMP8420_OLLAMA_NUM_CTX_TECHNICAL",
                TECHNICAL_CONTEXT_TOKENS,
            ),
            max_new_tokens=min(
                config.max_new_tokens,
                STRUCTURED_TASK_TOKEN_CAPS[task],
            ),
            timeout_seconds=TECHNICAL_TIMEOUT_SECONDS,
            reason=f"structured_direct:{task}",
        )
    if task in BOUNDED_REASONING_TASK_TOKENS:
        return GenerationPolicy(
            thinking_enabled=True,
            context_window=_env_context_tokens(
                "COMP8420_OLLAMA_NUM_CTX_TECHNICAL",
                TECHNICAL_CONTEXT_TOKENS,
            ),
            max_new_tokens=BOUNDED_REASONING_TASK_TOKENS[task],
            timeout_seconds=TECHNICAL_TIMEOUT_SECONDS,
            reason=f"bounded_reasoning:{task}",
        )
    long_context_window = _env_context_tokens(
        "COMP8420_OLLAMA_NUM_CTX_LONG",
        LONG_CONTEXT_TOKENS,
    )
    technical_context_window = _env_context_tokens(
        "COMP8420_OLLAMA_NUM_CTX_TECHNICAL",
        TECHNICAL_CONTEXT_TOKENS,
    )
    if task in LONG_CONTEXT_THINKING_TASKS:
        return GenerationPolicy(
            thinking_enabled=True,
            context_window=long_context_window,
            max_new_tokens=min(
                config.max_new_tokens,
                LONG_CONTEXT_THINKING_MAX_NEW_TOKENS,
            ),
            timeout_seconds=TECHNICAL_TIMEOUT_SECONDS,
            reason=f"long_context_thinking:{task or 'unknown'}",
        )
    return GenerationPolicy(
        thinking_enabled=True,
        context_window=technical_context_window,
        max_new_tokens=max(config.max_new_tokens, TECHNICAL_MAX_NEW_TOKENS),
        timeout_seconds=TECHNICAL_TIMEOUT_SECONDS,
        reason=f"technical_task:{task or 'unknown'}",
    )


def _policy_metadata(policy: GenerationPolicy) -> dict[str, Any]:
    return {
        "thinking_enabled": policy.thinking_enabled,
        "thinking_policy_reason": policy.reason,
        "context_window": policy.context_window,
        "max_new_tokens": policy.max_new_tokens,
        "timeout_seconds": policy.timeout_seconds,
    }


def _answer_format_instruction(task: str) -> str:
    if task == "topic_search_synthesis":
        return (
            'Return the final answer as a single JSON object only (keys: "summary", '
            '"key_findings", "research_gaps"). No markdown fences or extra prose.'
        )
    if task == "paper_recommendation":
        return (
            'Return the final answer as a single JSON object only with key '
            '"paper_summaries": an array of {"source_id": string, "summary": string}. '
            "No markdown fences or extra prose."
        )
    if task == "uploaded_paper_summary":
        return (
            "Return compact Markdown with headings for Scope, Core Contribution, Method, "
            "Results, and Limitations. Use only the supplied ParsedPaper."
        )
    if task == "paper_question_answer":
        return (
            "Answer the user's specific paper question directly in compact Markdown. "
            "Name supporting supplied sections and state when evidence is missing."
        )
    if task == "peer_review_critique":
        return (
            "Return compact Markdown with headings for Strengths, Weaknesses, Missing "
            "Evidence, Suggested Improvements, and Evidence Basis. Use only the supplied "
            "ParsedPaper and identify the supporting section or structural check."
        )
    return "Return the final user-visible answer only. Use compact Markdown sections."


def _query_adaptation_instruction(query_analysis: dict[str, Any] | None) -> str:
    if not query_analysis:
        return ""
    emotion = query_analysis.get("emotion")
    expertise = query_analysis.get("topic_expertise")
    verbosity = query_analysis.get("verbosity")
    intent = query_analysis.get("intent")
    style = query_analysis.get("style")

    instructions = [f"Resolved response style: {style}."]
    if intent == "chit_chat":
        instructions.append("Reply briefly and warmly without retrieval or technical exposition.")
    elif emotion == "frustrated":
        instructions.append("Acknowledge the difficulty briefly, then give concise actionable steps.")
    elif emotion == "confused" and style == "beginner":
        instructions.append("Use supportive beginner-friendly language and explain unfamiliar terms.")
    if expertise == "advanced" or style == "technical":
        instructions.append("Use precise technical language and include mathematical detail when requested.")
    if verbosity == "concise":
        instructions.append("Keep the answer short.")
    elif verbosity == "detailed":
        instructions.append("Give a structured, detailed explanation.")
    return " ".join(instructions)


def _prompt_query_analysis(
    query_analysis: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep model-facing adaptation fields; observability scores stay in metadata."""
    if not query_analysis:
        return None
    keys = ("intent", "emotion", "topic_expertise", "verbosity", "style")
    return {
        key: query_analysis[key]
        for key in keys
        if key in query_analysis
    }


# Parsed-paper fields that are large, token-heavy NLP dumps with no value for
# generation. They balloon the prompt (tens of thousands of tokens) and slow
# prompt evaluation; the model only needs sections, metadata, and compact
# enrichment. The parser output that backs the UI is left untouched.
_PROMPT_DROP_ANALYSIS_FIELDS = ("pos", "entity_mentions")
_PROMPT_DROP_PARSED_FIELDS = ("references",)
# Cap each section so one very long section cannot dominate the prompt. The most
# informative content is at the start of a section; local 8B prompt-eval scales
# with token count (~3.4k tokens ≈ 19s, full paper ≈ minutes on a laptop).
PROMPT_SECTION_CHAR_CAP = 4000
PEER_REVIEW_SECTION_CHAR_CAP = 2200
_SECTION_TRUNCATION_NOTICE = " […section truncated for summarization…]"


def _cap_sections(
    sections: Any,
    *,
    char_cap: int = PROMPT_SECTION_CHAR_CAP,
    preserve_tail: bool = False,
) -> Any:
    if not isinstance(sections, dict):
        return sections
    capped: dict[str, Any] = {}
    for name, text in sections.items():
        if isinstance(text, str) and len(text) > char_cap:
            if preserve_tail:
                head_chars = char_cap // 2
                tail_chars = char_cap - head_chars
                capped[name] = (
                    text[:head_chars]
                    + _SECTION_TRUNCATION_NOTICE
                    + text[-tail_chars:]
                )
            else:
                capped[name] = text[:char_cap] + _SECTION_TRUNCATION_NOTICE
        else:
            capped[name] = text
    return capped


def compact_parsed_paper_for_prompt(
    parsed_paper: Any,
    *,
    section_char_cap: int = PROMPT_SECTION_CHAR_CAP,
    preserve_section_tails: bool = False,
) -> Any:
    """Strip token-heavy NLP dumps and cap sections before prompting."""
    if not isinstance(parsed_paper, dict):
        return parsed_paper
    compact = {
        key: value
        for key, value in parsed_paper.items()
        if key not in _PROMPT_DROP_PARSED_FIELDS
    }
    if "sections" in compact:
        compact["sections"] = _cap_sections(
            compact["sections"],
            char_cap=section_char_cap,
            preserve_tail=preserve_section_tails,
        )
    analysis = compact.get("analysis")
    if isinstance(analysis, dict):
        compact["analysis"] = {
            key: value
            for key, value in analysis.items()
            if key not in _PROMPT_DROP_ANALYSIS_FIELDS
        }
    return compact


_CITATION_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "approach",
    "for",
    "in",
    "improved",
    "mastering",
    "method",
    "of",
    "on",
    "paper",
    "study",
    "the",
    "to",
    "using",
    "with",
}


def _citation_terms(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) > 2 and token not in _CITATION_QUERY_STOPWORDS
    }


def _compact_citation_pack(pack: Any) -> Any:
    """Keep citation prompts focused on evidence and supplied metadata."""
    if not isinstance(pack, dict):
        return pack
    query_terms = _citation_terms(pack.get("query"))
    grouped: dict[str, list[dict[str, Any]]] = {
        "eligible_candidates": [],
        "nearby_candidates": [],
        "rejected_candidates": [],
    }
    for candidate in pack.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        paper = candidate.get("paper") or {}
        candidate_terms = _citation_terms(
            f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
        )
        matched_terms = sorted(query_terms.intersection(candidate_terms))
        missing_terms = sorted(query_terms.difference(candidate_terms))
        coverage = len(matched_terms) / len(query_terms) if query_terms else 0.0
        if coverage >= 0.6:
            group = "eligible_candidates"
        elif coverage >= 0.25:
            group = "nearby_candidates"
        else:
            group = "rejected_candidates"
        grouped[group].append(
            {
                "source_id": str(
                    paper.get("paper_id")
                    or paper.get("arxiv_id")
                    or ""
                ),
                "title": paper.get("title"),
                "abstract": paper.get("abstract"),
                "authors": paper.get("authors"),
                "published_date": paper.get("published_date"),
                "venue": paper.get("venue"),
                "doi": paper.get("doi"),
                "url": paper.get("url"),
                "apa_citation": candidate.get("apa_citation"),
                "query_term_coverage": round(coverage, 3),
                "matched_query_terms": matched_terms,
                "missing_query_terms": missing_terms,
            }
        )
    compact = {
        "query": pack.get("query"),
        "eligibility_rule": (
            "Only eligible_candidates may be recommended. nearby_candidates may "
            "be reported as nearby leads, never as direct recommendations."
        ),
        "evidence_snippets": pack.get("evidence_snippets", []),
    }
    compact.update(grouped)
    return compact


def _compact_topic_pack(pack: Any) -> Any:
    """Remove duplicate abstracts while retaining source and ranking metadata."""
    if not isinstance(pack, dict):
        return pack
    candidates: list[dict[str, Any]] = []
    for candidate in pack.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        paper = candidate.get("paper") or {}
        candidates.append(
            {
                "paper": {
                    key: paper.get(key)
                    for key in (
                        "paper_id",
                        "title",
                        "authors",
                        "published_date",
                        "venue",
                        "doi",
                        "url",
                    )
                },
                "score": candidate.get("score"),
                "relation": candidate.get("relation"),
                "evidence": candidate.get("evidence"),
            }
        )
    return {
        "query": pack.get("query"),
        "retrieval_mode": pack.get("retrieval_mode"),
        "candidates": candidates,
        "evidence_snippets": pack.get("evidence_snippets", []),
    }


def _apply_citation_eligibility_guard(
    text: str,
    prompt_input: dict[str, Any],
) -> tuple[str, bool]:
    compact_pack = _compact_citation_pack(prompt_input.get("rag_evidence_pack"))
    if not isinstance(compact_pack, dict):
        return text, False
    if compact_pack.get("eligible_candidates"):
        return text, False

    lines = [
        "No directly relevant citation was found in the supplied evidence.",
    ]
    nearby = compact_pack.get("nearby_candidates") or []
    if nearby:
        lines.extend(["", "Nearby leads (not recommended as direct citations):"])
        for candidate in nearby[:3]:
            matched = ", ".join(candidate.get("matched_query_terms") or []) or "none"
            missing = ", ".join(candidate.get("missing_query_terms") or []) or "none"
            lines.append(
                f"- [{candidate['source_id']}] {candidate.get('apa_citation') or candidate.get('title')}. "
                f"Supported query terms: {matched}. Missing core query terms: {missing}."
            )
    else:
        lines.append("All supplied candidates were rejected by the evidence-coverage check.")
    return "\n".join(lines), True


def _compact_input_for_prompt(prompt_input: Any, *, task: str) -> Any:
    if not isinstance(prompt_input, dict) or "parsed_paper" not in prompt_input:
        return prompt_input
    compact = dict(prompt_input)
    if task == "citation_recommendation":
        compact["rag_evidence_pack"] = _compact_citation_pack(
            prompt_input.get("rag_evidence_pack")
        )
    elif task in {"topic_search_synthesis", "paper_recommendation"}:
        compact["rag_evidence_pack"] = _compact_topic_pack(
            prompt_input.get("rag_evidence_pack")
        )
    section_char_cap = (
        PEER_REVIEW_SECTION_CHAR_CAP
        if task == "peer_review_critique"
        else PROMPT_SECTION_CHAR_CAP
    )
    compact["parsed_paper"] = compact_parsed_paper_for_prompt(
        prompt_input["parsed_paper"],
        section_char_cap=section_char_cap,
        preserve_section_tails=task == "peer_review_critique",
    )
    return compact


def prompt_text_for_record(prompt_record: dict[str, Any], strategy: str) -> str:
    """Serialize a prompt record into the text sent to Ollama."""
    prompt_key = "few_shot_prompt" if strategy == "few_shot" else "zero_shot_prompt"
    payload = {
        "task": prompt_record["task"],
        "style": prompt_record["style"],
        "input": _compact_input_for_prompt(
            prompt_record["input"],
            task=prompt_record["task"],
        ),
        "expected_output_contract": prompt_record["expected_output_contract"],
    }
    if prompt_record.get("query_analysis"):
        payload["query_analysis"] = _prompt_query_analysis(
            prompt_record["query_analysis"]
        )
    parts = [
        prompt_record[prompt_key],
        _answer_format_instruction(prompt_record["task"]),
    ]
    adaptation = _query_adaptation_instruction(prompt_record.get("query_analysis"))
    if adaptation:
        parts.append(adaptation)
    parts.extend(
        [
            "Input JSON:",
            json.dumps(payload, indent=2, ensure_ascii=True),
        ]
    )
    return "\n\n".join(parts)


class OllamaRuntime:
    """Calls a local Ollama daemon through `/api/generate`."""

    backend_name = "ollama"

    def __init__(self, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.host = host.rstrip("/")

    def _call(
        self,
        payload: dict[str, Any],
        timeout_seconds: int,
    ) -> tuple[str, str | None, dict[str, Any]]:
        """POST one /api/generate call; return (text, error, response body)."""
        return ollama_transport.generate(self.host, payload, timeout_seconds)

    def generate(
        self,
        prompt_record: dict[str, Any],
        model: str,
        strategy: str,
        config: GenerationConfig,
    ) -> GenerationResult:
        started = time.monotonic()
        prompt = prompt_text_for_record(prompt_record, strategy)
        policy = generation_policy_for_record(prompt_record, config)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": policy.thinking_enabled,
            "keep_alive": _ollama_keep_alive(),
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_ctx": policy.context_window,
                "num_predict": policy.max_new_tokens,
            },
        }
        text, error, body = self._call(payload, policy.timeout_seconds)

        # Extended thinking can consume the entire token budget before any answer
        # is emitted (empty response, done_reason="length"). Retry once with
        # thinking disabled so the budget is spent on the user-visible answer.
        thinking_fallback_used = False
        if not text and policy.thinking_enabled:
            fallback_payload = {**payload, "think": False}
            fb_text, fb_error, fb_body = self._call(
                fallback_payload,
                policy.timeout_seconds,
            )
            if fb_text:
                text, error, body = fb_text, None, fb_body
                thinking_fallback_used = True
            else:
                error = fb_error or error
        text = re.sub(
            r"^\s*<think>\s*</think>\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
        citation_guard_applied = False
        if prompt_record["task"] == "citation_recommendation" and not error:
            text, citation_guard_applied = _apply_citation_eligibility_guard(
                text,
                prompt_record["input"],
            )

        return GenerationResult(
            text=text,
            backend=self.backend_name,
            model=model,
            prompt_id=prompt_record["prompt_id"],
            task=prompt_record["task"],
            strategy=strategy,
            latency_seconds=round(time.monotonic() - started, 4),
            error=error,
            evidence_ids_used=source_ids_used(text),
            run_metadata={
                "host": self.host,
                "keep_alive": payload["keep_alive"],
                "temperature": config.temperature,
                "top_p": config.top_p,
                "thinking_fallback_used": thinking_fallback_used,
                "citation_eligibility_guard_applied": citation_guard_applied,
                **_policy_metadata(policy),
                "ollama_load_duration_ms": round(
                    float(body.get("load_duration", 0)) / 1_000_000,
                    4,
                ),
                "ollama_prompt_eval_count": body.get("prompt_eval_count"),
                "ollama_prompt_eval_duration_ms": round(
                    float(body.get("prompt_eval_duration", 0)) / 1_000_000,
                    4,
                ),
                "ollama_eval_count": body.get("eval_count"),
                "ollama_eval_duration_ms": round(
                    float(body.get("eval_duration", 0)) / 1_000_000,
                    4,
                ),
            },
        )


def runtime_notes_markdown() -> str:
    """Return a Markdown table of configured model variants for comparison runs."""
    rows = [
        "| Variant | Ollama tag | Role | Quantization | Adapter | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for variant in MODEL_VARIANTS:
        rows.append(
            f"| `{variant.name}` | `{variant.ollama_tag}` | {variant.role} | "
            f"{variant.quantization} | {variant.adapter} | {variant.notes} |"
        )
    return "\n".join(
        [
            "# Runtime Notes",
            "",
            "Current backend: `ollama`.",
            "",
            "Ollama calls a local daemon and records latency, model tag, prompt id, error",
            "status, and source IDs used.",
            "",
            *rows,
            "",
            "Required measurements: latency, memory notes, output path, quantization level,",
            "structure compliance, evidence faithfulness, and human spot-check notes.",
            "",
        ]
    )
