"""Render an AnalysisResult into a human-readable markdown report.

Used by analyze-pdf and topic search to produce `outputs/demo_report.md` — the
readable artifact for the video/report. Mirrors the frontend results panel.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from .contracts import AnalysisResult

_ARXIV_URL_RE = re.compile(r"https?://arxiv\.org/abs/[^\s)>\"]+", re.I)


def _paper_link(it: dict) -> str:
    """Resolve a clickable arXiv URL from recommendation fields."""
    url = (it.get("url") or "").strip()
    if url:
        return url
    pid = (it.get("paper_id") or "").strip()
    if pid:
        return f"https://arxiv.org/abs/{pid}"
    apa = it.get("apa_citation") or ""
    m = _ARXIV_URL_RE.search(apa)
    return m.group(0).rstrip(".") if m else ""


def render_markdown(r: AnalysisResult) -> str:
    lines = [
        f"# Analysis Report",
        "",
        f"- **Input type:** {r.input_type}",
        f"- **Input:** {r.input_ref}",
    ]
    if r.metadata:
        title = r.metadata.get("title", "")
        authors = r.metadata.get("authors", [])
        if isinstance(authors, list):
            authors = ", ".join(authors)
        lines += [f"- **Title:** {title}", f"- **Authors:** {authors}"]

    flags = r.flags or {}
    lines += [
        f"- **Retrieval mode:** {flags.get('retrieval_mode', '-')}",
        f"- **Model mode:** {flags.get('model_mode', '-')}",
        f"- **Module sources:** {flags.get('provider_sources', {})}",
        "",
        "## Summary",
        "",
        r.summary,
        "",
        "## Key findings",
        "",
    ]
    lines += [f"- {f}" for f in r.key_findings] or ["- (none)"]
    lines += ["", "## Research gaps", ""]
    lines += [f"- {g}" for g in r.research_gaps] or ["- (none)"]

    lines += ["", "## Recommended papers", ""]
    if r.recommended_papers:
        for it in r.recommended_papers:
            lines.append(f"- **{it.get('title','')}** "
                         f"(score {it.get('score','?')}) — {it.get('why','')}")
    else:
        lines.append("- (none)")

    lines += ["", "## APA citations", ""]
    lines += [f"- {c}" for c in r.apa_citations] or ["- (none)"]

    lines += ["", "## Evidence", ""]
    if r.evidence:
        for s in r.evidence:
            lines.append(f"- [{s.get('paper_id','?')}] "
                         f"(score {s.get('score','?')}): {s.get('text','')}")
    else:
        lines.append("- (none)")

    analysis = r.paper_analysis or {}
    if analysis:
        lines += ["", "## Deterministic PDF-NLP analysis", ""]
        extractive = analysis.get("extractive_summary") or {}
        if extractive.get("text"):
            lines += [
                "### Extractive summary",
                "",
                str(extractive["text"]),
                "",
            ]
        keyphrases = analysis.get("keyphrases") or []
        lines += ["### Keyphrases", ""]
        lines += [
            f"- {item.get('text', '')} ({item.get('section', '-')}, "
            f"score {item.get('score', '-')})"
            for item in keyphrases
        ] or ["- (none)"]
        entities = analysis.get("entity_mentions") or []
        lines += ["", "### Scientific entities", ""]
        lines += [
            f"- [{item.get('type', '')}] {item.get('text', '')} "
            f"({item.get('source', '')})"
            for item in entities[:40]
        ] or ["- (none)"]
        checks = analysis.get("structural_checks") or []
        lines += ["", "### Structural checks", ""]
        lines += [
            f"- **{item.get('severity', '')}:** {item.get('message', '')}"
            for item in checks
        ] or ["- (none)"]

    if r.peer_review:
        lines += ["", "## Peer-review assistance", "", r.peer_review]

    lines += ["", "---",
              "",
              ""]
    return "\n".join(lines)


def format_topic_cli_command(
    command: str,
    *,
    query: str,
    corpus: Optional[str] = None,
    corpus_limit: Optional[int] = None,
    llm_model: str = "qwen3:8b",
    style: str = "technical",
    retrieval_mode: str = "offline",
    model_mode: str = "base",
    top_n: Optional[int] = None,
) -> str:
    """Rebuild CLI invocation string for terminal display."""
    parts = ["python", "-m", "app.cli", command]
    q = query.replace("\\", "\\\\").replace('"', '\\"')
    parts.append(f'--query "{q}"')
    if corpus:
        c = corpus.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'--corpus "{c}"')
    if corpus_limit is not None:
        parts.append(f"--corpus-limit {corpus_limit}")
    parts.append(f"--llm-model {llm_model}")
    parts.append(f"--style {style}")
    parts.append(f"--retrieval-mode {retrieval_mode}")
    parts.append(f"--model-mode {model_mode}")
    if top_n is not None:
        parts.append(f"--top-n {top_n}")
    return " ".join(parts)


def _format_score(score: Any) -> str:
    if isinstance(score, (int, float)):
        return f"{score:.4f}".rstrip("0").rstrip(".")
    return str(score)


def _provider_line(sources: dict[str, str]) -> str:
    order = ("paper_source", "recommender", "synthesizer", "parser")
    parts = []
    for role in order:
        if role in sources:
            parts.append(f"{role}={sources[role]}")
    for role, src in sorted(sources.items()):
        if role not in order:
            parts.append(f"{role}={src}")
    return " ".join(parts)


def _format_trace_steps(trace: list[dict[str, Any]]) -> list[str]:
    lines = []
    for step in trace:
        name = step.get("step", "")
        if name.endswith(" START"):
            continue
        ms = step.get("elapsed_ms", 0)
        if isinstance(ms, (int, float)):
            elapsed = f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms:.0f}ms"
        else:
            elapsed = str(ms)
        lines.append(f"  {name:<36} {elapsed:>8}")
    return lines


def format_run_output(
    r: AnalysisResult,
    *,
    trace: Optional[list[dict[str, Any]]] = None,
    session_path: Optional[str] = None,
    command_line: Optional[str] = None,
    top_n: int = 5,
) -> str:
    """Terminal output from live AnalysisResult only (no display fallbacks)."""
    flags = r.flags or {}
    sources = flags.get("provider_sources") or {}
    query = r.input_ref or "—"
    cmd = command_line or format_topic_cli_command("run", query=query, top_n=top_n)

    lines = [
        "Topic search — local integration",
        "=" * 40,
        f"Query: {query}",
        f"Command: {cmd}",
        "",
        "Providers",
        _provider_line(sources) if sources else "(unknown)",
        "",
    ]

    if trace:
        lines += ["Pipeline steps (elapsed)", "-" * 22]
        lines.extend(_format_trace_steps(trace))
        if trace:
            total = trace[-1].get("elapsed_ms", 0)
            if isinstance(total, (int, float)):
                total_s = f"{total / 1000:.1f}s" if total >= 1000 else f"{total:.0f}ms"
                lines.append(f"  {'TOTAL':<36} {total_s:>8}")
        lines.append("")

    lines += [
        f"Evidence snippets: {len(r.evidence or [])}",
        f"Recommendations: {len(r.recommended_papers or [])}",
        "",
        "Summary",
        "-" * 7,
        (r.summary or "").strip() or "(empty)",
        "",
        "Key findings",
        "-" * 12,
    ]
    if r.key_findings:
        lines.extend(f"- {f}" for f in r.key_findings)
    else:
        lines.append("(none)")

    lines += ["", "Research gaps", "-" * 14]
    if r.research_gaps:
        lines.extend(f"- {g}" for g in r.research_gaps)
    else:
        lines.append("(none)")

    papers = (r.recommended_papers or [])[:top_n]
    lines += ["", f"Top recommendations ({len(papers)})", "-" * 24]
    if papers:
        for i, it in enumerate(papers, 1):
            lines.append(f"{i}. [{_format_score(it.get('score', '?'))}] {(it.get('title') or '—').strip()}")
            pid = (it.get("paper_id") or "").strip()
            if pid:
                lines.append(f"   ID: {pid}")
            link = _paper_link(it)
            if link:
                lines.append(f"   Link: {link}")
            why = (it.get("why") or "").strip()
            if why:
                lines.append(f"   Reason: {why}")
    else:
        lines.append("(none)")

    lines += [
        "",
        "Artifacts",
        "  outputs/analysis_result.json",
        "  outputs/demo_report.md",
        "  outputs/demo_trace.json",
        "  outputs/llm_analysis.md",
    ]
    if session_path:
        lines.append(f"  session: {session_path}")

    return "\n".join(lines)
