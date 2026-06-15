"""Bounded ReAct planning for topic RAG (search_offline tool only)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .runtime import GenerationConfig, OllamaRuntime


SearchOfflineFn = Callable[[str, int], dict[str, Any]]


REACT_TOOL_SCHEMA = {
    "tool": "search_offline",
    "arguments": {
        "query": "string — retrieval query for the offline paper corpus",
        "top_k": "integer — number of papers to retrieve (1-20)",
    },
}


def build_react_plan_prompt(user_query: str) -> str:
    return (
        "You are a research assistant with access to one tool.\n"
        "Decide whether offline paper retrieval is needed, then respond with "
        "ONLY a JSON object (no markdown fences):\n"
        '{"tool":"search_offline","arguments":{"query":"<retrieval query>","top_k":5}}\n\n'
        "Rules:\n"
        "- Use search_offline for substantive research questions needing evidence.\n"
        "- The backend executes the tool; you do not fetch papers directly.\n"
        "- Prefer a concise retrieval query focused on methods, problems, or topics.\n\n"
        f"User query: {user_query}\n"
    )


def parse_tool_call(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
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
    if not isinstance(payload, dict):
        return None
    if payload.get("tool") != "search_offline":
        return None
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        return None
    query = str(arguments.get("query") or "").strip()
    if not query:
        return None
    try:
        top_k = int(arguments.get("top_k", 5))
    except (TypeError, ValueError):
        top_k = 5
    top_k = max(1, min(top_k, 20))
    return {"tool": "search_offline", "arguments": {"query": query, "top_k": top_k}}


@dataclass(frozen=True)
class ReactTopicRagResult:
    tool_call: dict[str, Any] | None
    retrieval_query: str
    top_k: int
    used_react: bool
    fallback_reason: str | None
    plan_text: str
    plan_latency_seconds: float
    plan_error: str | None


def plan_search_offline(
    user_query: str,
    *,
    runtime: OllamaRuntime,
    model: str,
    config: GenerationConfig | None = None,
) -> ReactTopicRagResult:
    """Ask the model which offline retrieval query to run."""
    prompt = build_react_plan_prompt(user_query)
    record = {
        "prompt_id": "react_topic_plan",
        "task": "react_tool_plan",
        "style": "technical",
        "zero_shot_prompt": prompt,
        "few_shot_prompt": prompt,
        "input": {"user_query": user_query},
        "expected_output_contract": {"must_return_tool_json": True},
    }
    config = config or GenerationConfig(max_new_tokens=128, temperature=0.1)
    result = runtime.generate(record, model, "zero_shot", config)
    tool_call = parse_tool_call(result.text)
    if tool_call is None:
        return ReactTopicRagResult(
            tool_call=None,
            retrieval_query=user_query,
            top_k=5,
            used_react=False,
            fallback_reason=result.error or "invalid_tool_json",
            plan_text=result.text or "",
            plan_latency_seconds=result.latency_seconds,
            plan_error=result.error,
        )
    args = tool_call["arguments"]
    return ReactTopicRagResult(
        tool_call=tool_call,
        retrieval_query=args["query"],
        top_k=args["top_k"],
        used_react=True,
        fallback_reason=None,
        plan_text=result.text or "",
        plan_latency_seconds=result.latency_seconds,
        plan_error=result.error,
    )


def run_react_topic_rag(
    user_query: str,
    *,
    search_offline: SearchOfflineFn,
    runtime: OllamaRuntime,
    model: str,
    config: GenerationConfig | None = None,
    default_top_k: int = 5,
) -> tuple[dict[str, Any], ReactTopicRagResult]:
    """Plan retrieval with ReAct, execute search_offline, return RagEvidencePack dict."""
    plan = plan_search_offline(
        user_query,
        runtime=runtime,
        model=model,
        config=config,
    )
    retrieval_query = plan.retrieval_query or user_query
    top_k = plan.top_k or default_top_k
    if not plan.used_react:
        retrieval_query = user_query
        top_k = default_top_k
    pack = search_offline(retrieval_query, top_k)
    return pack, plan
