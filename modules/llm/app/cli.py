"""Command-line entry points for paper summarization and RAG synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import compare_models, compare_prompts, ensure_fixed_prompts, parse_model_specs
from .io_utils import read_json, write_json, write_text
from .prompt_library import (
    direct_chat_prompt_record,
    paper_recommendation_prompt_record,
    paper_review_prompt_record,
    paper_summary_prompt_record,
    topic_synthesis_prompt_record,
)
from .query_understanding import QueryAnalysis, VALID_STYLES, analyze_query
from .runtime import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    GenerationConfig,
    GenerationResult,
    build_runtime,
)
from .schemas import SchemaError, validate_parsed_paper, validate_rag_evidence_pack
from .verification import (
    apply_inline_verification,
    apply_verification,
    with_inline_verification,
)
from .session_append import append_session_event
from .synthesis import (
    analysis_result_from_pack,
    build_recommended_paper_cards,
    handoff_markdown,
    parse_paper_review,
    parse_paper_recommendation_summaries,
    parse_paper_summary,
    review_paper_markdown,
    summarize_paper_markdown,
    synthesize_markdown,
)


def _read_required_json(path: Path, label: str) -> Any:
    """Read a required JSON input and translate file errors into CLI messages."""
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")
    try:
        return read_json(path)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {label} file {path}: {exc}") from exc


def _generation_config(args: argparse.Namespace) -> GenerationConfig:
    """Map shared CLI generation flags to the runtime configuration."""
    return GenerationConfig(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )


def _query_analysis(args: argparse.Namespace, query: str) -> QueryAnalysis:
    """Reuse upstream analysis when supplied; otherwise classify the query once."""
    analysis_path = getattr(args, "query_analysis", None)
    if not analysis_path:
        return analyze_query(query, style_override=args.style)

    payload = _read_required_json(Path(analysis_path), "QueryAnalysis")
    if not isinstance(payload, dict):
        raise SystemExit(f"QueryAnalysis must be a JSON object: {analysis_path}")
    try:
        analysis = QueryAnalysis.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid QueryAnalysis in {analysis_path}: {exc}") from exc
    if args.style != "auto" and analysis.style != args.style:
        raise SystemExit(
            "Precomputed QueryAnalysis style does not match --style: "
            f"{analysis.style!r} != {args.style!r}"
        )
    return analysis


def _generation_metadata(
    result: GenerationResult,
    *,
    requested_style: str,
    query_analysis: QueryAnalysis,
    **extra: Any,
) -> dict[str, Any]:
    """Build consistent metadata for paper-only and RAG generation commands."""
    return {
        **extra,
        "requested_style": requested_style,
        "resolved_style": query_analysis.style,
        "query_analysis": query_analysis.as_dict(),
        **result.as_dict(),
    }


PROMPT_STRATEGIES = ("zero_shot", "few_shot")


def _prompt_strategy_for(args: argparse.Namespace, *, command: str) -> str:
    """Resolve prompt strategy; review keeps few-shot default when unset."""
    raw = getattr(args, "prompt_strategy", None)
    if raw in PROMPT_STRATEGIES:
        return raw
    if command == "review":
        return "few_shot"
    return "zero_shot"


def _generate_and_verify(
    runtime,
    prompt_record: dict,
    model: str,
    strategy: str,
    config: GenerationConfig,
    *,
    pack: dict | None = None,
    paper: dict | None = None,
    verify: bool = True,
    fast_verify: bool = False,
) -> tuple[GenerationResult, dict[str, Any]]:
    effective_record = (
        with_inline_verification(prompt_record)
        if fast_verify and verify
        else prompt_record
    )
    result = runtime.generate(effective_record, model, strategy, config)
    if not verify:
        return result, {
            "verification_used": False,
            "verification_revised": False,
            "verification_reason": "disabled_by_cli",
        }
    if fast_verify:
        return apply_inline_verification(result, pack=pack)
    return apply_verification(
        result,
        runtime=runtime,
        model=model,
        config=config,
        pack=pack,
        paper=paper,
    )


def _verification_event(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "verification_used": audit.get("verification_used"),
        "verification_revised": audit.get("verification_revised"),
        "verification_mode": audit.get("verification_mode", "second_pass"),
        "verification_supported": audit.get("supported"),
        "verification_error": audit.get("error"),
    }


def _generation_policy_event(result: GenerationResult) -> dict[str, Any]:
    """Expose selected inference budgets without recording hidden reasoning."""
    metadata = result.run_metadata
    return {
        "thinking_enabled": metadata.get("thinking_enabled"),
        "thinking_policy_reason": metadata.get("thinking_policy_reason"),
        "context_window": metadata.get("context_window"),
        "max_new_tokens": metadata.get("max_new_tokens"),
    }


def cmd_synthesize(args: argparse.Namespace) -> None:
    """Generate related-paper analysis from an existing retrieval evidence pack."""
    evidence_path = Path(args.evidence)
    out_path = Path(args.out)
    metadata_path = out_path.parent / "llm_generation.json"
    handoff_path = out_path.parent / "llm_handoff.md"
    append_session_event(
        "user_input",
        {
            "command": "synthesize",
            "evidence": str(evidence_path),
            "backend": "ollama",
            "model": args.model,
            "style": args.style,
        },
    )
    pack = _read_required_json(evidence_path, "RagEvidencePack")
    validate_rag_evidence_pack(pack)
    query_analysis = _query_analysis(args, pack["query"])
    runtime = build_runtime(args.ollama_host)
    strategy = _prompt_strategy_for(args, command="synthesize")
    result, audit = _generate_and_verify(
        runtime,
        topic_synthesis_prompt_record(
            pack,
            query_analysis.style,
            query_analysis=query_analysis.as_dict(),
        ),
        args.model,
        strategy,
        _generation_config(args),
        pack=pack,
        verify=not args.no_verify,
        fast_verify=args.fast_verify,
    )
    markdown = synthesize_markdown(
        pack,
        query_analysis.style,
        result,
        verification_audit=audit,
    )
    write_text(out_path, markdown)
    json_out = (
        Path(args.json_out)
        if args.json_out
        else out_path.parent / "analysis_result_from_llm.json"
    )
    write_json(json_out, analysis_result_from_pack(pack, result))
    write_json(
        metadata_path,
        _generation_metadata(
            result,
            requested_style=args.style,
            query_analysis=query_analysis,
            input_contract="rag_evidence_pack",
            evidence_path=str(evidence_path),
            retrieval_used=True,
            external_evidence_used=True,
            prompt_strategy=strategy,
        ),
    )
    write_text(handoff_path, handoff_markdown())
    append_session_event(
        "synthesis",
        {
            "backend": "ollama",
            "model": args.model,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
            "evidence_ids_used": result.evidence_ids_used,
            "prompt_strategy": strategy,
            **_generation_policy_event(result),
            **_verification_event(audit),
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(json_out), str(metadata_path)],
            "summary_preview": markdown[:500],
        },
    )
    print(f"Wrote synthesis markdown: {out_path}")
    print(f"Wrote AnalysisResult JSON: {json_out}")
    print(f"Wrote generation metadata: {metadata_path}")
    print(f"Wrote handoff: {handoff_path}")
    if result.error:
        print(f"Generation backend reported an error: {result.error}")


def cmd_chat(args: argparse.Namespace) -> None:
    """Generate a direct conversational response without retrieval."""
    out_path = Path(args.out)
    query_analysis = _query_analysis(args, args.query)
    append_session_event(
        "user_input",
        {
            "command": "chat",
            "query": args.query,
            "backend": "ollama",
            "model": args.model,
            "style": args.style,
            "route": "direct_llm",
        },
    )
    runtime = build_runtime(args.ollama_host)
    result = runtime.generate(
        direct_chat_prompt_record(
            args.query,
            query_analysis.style,
            query_analysis=query_analysis.as_dict(),
        ),
        args.model,
        "zero_shot",
        _generation_config(args),
    )
    answer = result.text.strip()
    if result.error:
        answer = f"Generation failed: {result.error}"
    write_text(out_path, answer + "\n")
    metadata_path = (
        Path(args.metadata_out)
        if args.metadata_out
        else out_path.parent / f"{out_path.stem}_generation.json"
    )
    write_json(
        metadata_path,
        _generation_metadata(
            result,
            requested_style=args.style,
            query_analysis=query_analysis,
            input_contract="text_only",
            retrieval_used=False,
            external_evidence_used=False,
        ),
    )
    append_session_event(
        "synthesis",
        {
            "backend": "ollama",
            "model": args.model,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
            "evidence_ids_used": [],
            "route": "direct_llm",
            **_generation_policy_event(result),
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(metadata_path)],
            "summary_preview": answer[:500],
            "route": "direct_llm",
        },
    )
    print(f"Wrote direct chat response: {out_path}")
    print(f"Wrote generation metadata: {metadata_path}")
    if result.error:
        raise SystemExit(f"Generation backend reported an error: {result.error}")


def cmd_recommend_summaries(args: argparse.Namespace) -> None:
    """Generate per-paper summaries for structured recommendation responses."""
    evidence_path = Path(args.evidence)
    out_path = Path(args.out)
    metadata_path = (
        Path(args.metadata_out)
        if args.metadata_out
        else out_path.parent / "paper_recommendation_generation.json"
    )
    append_session_event(
        "user_input",
        {
            "command": "recommend-summaries",
            "evidence": str(evidence_path),
            "backend": "ollama",
            "model": args.model,
            "style": args.style,
        },
    )
    pack = _read_required_json(evidence_path, "RagEvidencePack")
    validate_rag_evidence_pack(pack)
    query_analysis = _query_analysis(args, pack["query"])
    runtime = build_runtime(args.ollama_host)
    strategy = _prompt_strategy_for(args, command="recommend-summaries")
    result = runtime.generate(
        paper_recommendation_prompt_record(
            pack,
            query_analysis.style,
            query_analysis=query_analysis.as_dict(),
        ),
        args.model,
        strategy,
        _generation_config(args),
    )
    summaries = parse_paper_recommendation_summaries(result)
    cards = build_recommended_paper_cards(
        pack.get("candidates", []),
        summaries,
        pack,
        max_papers=args.max_papers,
    )
    payload = {
        "query": pack["query"],
        "summaries": summaries,
        "recommended_papers": cards,
    }
    write_json(out_path, payload)
    write_json(
        metadata_path,
        _generation_metadata(
            result,
            requested_style=args.style,
            query_analysis=query_analysis,
            input_contract="rag_evidence_pack",
            evidence_path=str(evidence_path),
            retrieval_used=True,
            external_evidence_used=True,
            route="paper_recommendation",
        ),
    )
    append_session_event(
        "synthesis",
        {
            "backend": "ollama",
            "model": args.model,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
            "evidence_ids_used": result.evidence_ids_used,
            "route": "paper_recommendation",
            **_generation_policy_event(result),
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(metadata_path)],
            "recommended_count": len(cards),
            "citation_count": sum(1 for card in cards if card.get("apa_citation")),
        },
    )
    print(f"Wrote recommendation summaries: {out_path}")
    print(f"Wrote generation metadata: {metadata_path}")
    if result.error:
        print(f"Generation backend reported an error: {result.error}")


def cmd_summarize(args: argparse.Namespace) -> None:
    """Generate a summary using only one validated parsed paper."""
    paper_path = Path(args.paper)
    append_session_event(
        "user_input",
        {
            "command": "summarize",
            "paper": str(paper_path),
            "query": args.query,
            "backend": "ollama",
            "model": args.model,
            "style": args.style,
        },
    )
    paper = _read_required_json(paper_path, "ParsedPaper")
    validate_parsed_paper(paper)

    query_analysis = _query_analysis(args, args.query)
    out_path = Path(args.out)
    runtime = build_runtime(args.ollama_host)
    strategy = _prompt_strategy_for(args, command="summarize")
    result, audit = _generate_and_verify(
        runtime,
        paper_summary_prompt_record(
            paper,
            query_analysis.style,
            user_query=args.query,
            query_analysis=query_analysis.as_dict(),
        ),
        args.model,
        strategy,
        _generation_config(args),
        paper=paper,
    )
    markdown = summarize_paper_markdown(
        paper,
        query_analysis.style,
        result,
        verification_audit=audit,
    )
    write_text(out_path, markdown)
    json_out = (
        Path(args.json_out)
        if args.json_out
        else out_path.parent / f"{out_path.stem}_result.json"
    )
    write_json(json_out, parse_paper_summary(result))
    metadata_path = (
        Path(args.metadata_out)
        if args.metadata_out
        else out_path.parent / f"{out_path.stem}_generation.json"
    )
    write_json(
        metadata_path,
        _generation_metadata(
            result,
            requested_style=args.style,
            query_analysis=query_analysis,
            input_contract="parsed_paper_only",
            paper_path=str(paper_path),
            retrieval_used=False,
            external_evidence_used=False,
            prompt_strategy=strategy,
        ),
    )
    append_session_event(
        "synthesis",
        {
            "backend": "ollama",
            "model": args.model,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
            "evidence_ids_used": [],
            "prompt_strategy": strategy,
            **_generation_policy_event(result),
            **_verification_event(audit),
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(json_out), str(metadata_path)],
            "summary_preview": markdown[:500],
        },
    )
    print(f"Wrote paper-only summary: {out_path}")
    print(f"Wrote structured paper summary: {json_out}")
    print(f"Wrote generation metadata: {metadata_path}")
    if result.error:
        raise SystemExit(f"Generation backend reported an error: {result.error}")


def cmd_review(args: argparse.Namespace) -> None:
    """Generate a constructive review using only one validated parsed paper."""
    paper_path = Path(args.paper)
    append_session_event(
        "user_input",
        {
            "command": "review",
            "paper": str(paper_path),
            "query": args.query,
            "backend": "ollama",
            "model": args.model,
            "style": args.style,
        },
    )
    paper = _read_required_json(paper_path, "ParsedPaper")
    validate_parsed_paper(paper)
    query_analysis = analyze_query(args.query, style_override=args.style)
    out_path = Path(args.out)
    runtime = build_runtime(args.ollama_host)
    strategy = _prompt_strategy_for(args, command="review")
    result, audit = _generate_and_verify(
        runtime,
        paper_review_prompt_record(
            paper,
            query_analysis.style,
            user_query=args.query,
            query_analysis=query_analysis.as_dict(),
        ),
        args.model,
        strategy,
        _generation_config(args),
        paper=paper,
    )
    write_text(
        out_path,
        review_paper_markdown(
            paper,
            query_analysis.style,
            result,
            verification_audit=audit,
        ),
    )
    json_out = (
        Path(args.json_out)
        if args.json_out
        else out_path.parent / f"{out_path.stem}_result.json"
    )
    write_json(json_out, parse_paper_review(result))
    metadata_path = (
        Path(args.metadata_out)
        if args.metadata_out
        else out_path.parent / f"{out_path.stem}_generation.json"
    )
    write_json(
        metadata_path,
        _generation_metadata(
            result,
            requested_style=args.style,
            query_analysis=query_analysis,
            input_contract="parsed_paper_only",
            paper_path=str(paper_path),
            retrieval_used=False,
            external_evidence_used=False,
            prompt_strategy=strategy,
        ),
    )
    append_session_event(
        "synthesis",
        {
            "backend": "ollama",
            "model": args.model,
            "latency_seconds": result.latency_seconds,
            "error": result.error,
            "evidence_ids_used": [],
            "route": "paper_peer_review",
            "prompt_strategy": strategy,
            **_generation_policy_event(result),
            **_verification_event(audit),
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(json_out), str(metadata_path)],
            "summary_preview": result.text[:500],
            "route": "paper_peer_review",
        },
    )
    print(f"Wrote paper-only peer review: {out_path}")
    print(f"Wrote structured peer review: {json_out}")
    print(f"Wrote generation metadata: {metadata_path}")
    if result.error:
        raise SystemExit(f"Generation backend reported an error: {result.error}")


def cmd_analyze_query(args: argparse.Namespace) -> None:
    """Write the semantic structured analysis for a user query."""
    analysis = analyze_query(args.query, style_override=args.style)
    payload = analysis.as_dict()
    if args.out:
        write_json(Path(args.out), payload)
        print(f"Wrote query analysis: {args.out}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


def cmd_compare_prompts(args: argparse.Namespace) -> None:
    """Run the fixed evaluation set across supported prompt strategies."""
    test_set = Path(args.test_set)
    out_dir = Path(args.out)
    ensure_fixed_prompts(test_set)
    compare_prompts(
        test_set,
        out_dir,
        model=args.model,
        ollama_host=args.ollama_host,
        config=_generation_config(args),
    )
    print(f"Wrote prompt comparison artifacts under: {out_dir}")


def cmd_compare_models(args: argparse.Namespace) -> None:
    """Run the fixed evaluation set across configured model variants."""
    test_set = Path(args.test_set)
    out_dir = Path(args.out)
    ensure_fixed_prompts(test_set)
    compare_models(
        test_set,
        out_dir,
        model_specs=parse_model_specs(args.models),
        ollama_host=args.ollama_host,
        config=_generation_config(args),
    )
    print(f"Wrote model comparison artifacts under: {out_dir}")


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    """Attach Ollama generation options shared by commands that invoke a runtime."""
    parser.add_argument(
        "--model",
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model tag.",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help="Local Ollama host.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--prompt-strategy",
        choices=PROMPT_STRATEGIES,
        default=None,
        help="Zero-shot or few-shot prompting; review defaults to few_shot when omitted.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the public LLM command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Mulkraj LLM, prompt, adapter, and evaluation stage commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_query_parser = subparsers.add_parser(
        "analyze-query",
        help=(
            "Classify a query with embedding cosine similarity and a "
            "low-confidence TinyBERT fallback."
        ),
    )
    analyze_query_parser.add_argument("--query", required=True, help="User query text.")
    analyze_query_parser.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Explicit response style override; auto uses query analysis.",
    )
    analyze_query_parser.add_argument("--out", default=None, help="Optional JSON output path.")
    analyze_query_parser.set_defaults(func=cmd_analyze_query)

    chat = subparsers.add_parser(
        "chat",
        help="Answer conversational text without retrieval or paper evidence.",
    )
    chat.add_argument("--query", required=True, help="User message text.")
    chat.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Response style; auto derives it from --query.",
    )
    chat.add_argument("--out", required=True, help="Output text or Markdown path.")
    chat.add_argument(
        "--metadata-out",
        default=None,
        help="Optional generation metadata JSON path. Defaults beside --out.",
    )
    chat.add_argument(
        "--query-analysis",
        default=None,
        help="Optional precomputed QueryAnalysis JSON; avoids duplicate classification.",
    )
    _add_generation_options(chat)
    chat.set_defaults(func=cmd_chat)

    summarize = subparsers.add_parser(
        "summarize",
        help="Summarize one supplied ParsedPaper without retrieval or external evidence.",
    )
    summarize.add_argument("--paper", required=True, help="Path to a ParsedPaper JSON object.")
    summarize.add_argument(
        "--query",
        default="Summarize the supplied paper.",
        help="User instruction used for automatic style analysis.",
    )
    summarize.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Response style; auto derives it from --query.",
    )
    summarize.add_argument("--out", required=True, help="Output Markdown summary path.")
    summarize.add_argument(
        "--json-out",
        default=None,
        help="Optional structured summary JSON path. Defaults beside --out.",
    )
    summarize.add_argument(
        "--metadata-out",
        default=None,
        help="Optional generation metadata JSON path. Defaults beside --out.",
    )
    summarize.add_argument(
        "--query-analysis",
        default=None,
        help="Optional precomputed QueryAnalysis JSON; avoids duplicate classification.",
    )
    _add_generation_options(summarize)
    summarize.set_defaults(func=cmd_summarize)

    review = subparsers.add_parser(
        "review",
        help="Peer-review one supplied ParsedPaper without retrieval or external evidence.",
    )
    review.add_argument("--paper", required=True, help="Path to a ParsedPaper JSON object.")
    review.add_argument(
        "--query",
        default="Write a constructive peer review of the supplied paper.",
        help="User instruction used for automatic style analysis.",
    )
    review.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Response style; auto derives it from --query.",
    )
    review.add_argument("--out", required=True, help="Output Markdown review path.")
    review.add_argument(
        "--json-out",
        default=None,
        help="Optional structured review JSON path. Defaults beside --out.",
    )
    review.add_argument(
        "--metadata-out",
        default=None,
        help="Optional generation metadata JSON path. Defaults beside --out.",
    )
    _add_generation_options(review)
    review.set_defaults(func=cmd_review)

    synthesize = subparsers.add_parser(
        "synthesize",
        help="Synthesize a retrieved RagEvidencePack for related-paper analysis.",
    )
    synthesize.add_argument("--evidence", required=True, help="Path to RagEvidencePack JSON.")
    synthesize.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Response style; auto derives it from RagEvidencePack.query.",
    )
    synthesize.add_argument("--out", required=True, help="Output markdown path.")
    synthesize.add_argument("--json-out", default=None, help="Optional AnalysisResult JSON output path.")
    synthesize.add_argument(
        "--query-analysis",
        default=None,
        help="Optional precomputed QueryAnalysis JSON; avoids duplicate classification.",
    )
    synthesize.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "Skip same-model verification. Intended for diagnostics; production "
            "and evaluation runs verify by default."
        ),
    )
    synthesize.add_argument(
        "--fast-verify",
        action="store_true",
        help=(
            "Use inline same-model evidence verification plus deterministic "
            "source-ID auditing for interactive chat. Full evaluation runs use "
            "the default second-pass verifier."
        ),
    )
    _add_generation_options(synthesize)
    synthesize.set_defaults(func=cmd_synthesize)

    recommend_summaries = subparsers.add_parser(
        "recommend-summaries",
        help="Generate per-paper summaries for structured recommendation responses.",
    )
    recommend_summaries.add_argument(
        "--evidence",
        required=True,
        help="Path to RagEvidencePack JSON.",
    )
    recommend_summaries.add_argument(
        "--style",
        default="auto",
        choices=VALID_STYLES,
        help="Response style; auto derives it from RagEvidencePack.query.",
    )
    recommend_summaries.add_argument(
        "--out",
        required=True,
        help="Output JSON path for recommended paper cards.",
    )
    recommend_summaries.add_argument(
        "--metadata-out",
        default=None,
        help="Optional generation metadata JSON path.",
    )
    recommend_summaries.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of recommendation cards to return.",
    )
    recommend_summaries.add_argument(
        "--query-analysis",
        default=None,
        help="Optional precomputed QueryAnalysis JSON; avoids duplicate classification.",
    )
    _add_generation_options(recommend_summaries)
    recommend_summaries.set_defaults(func=cmd_recommend_summaries)

    compare_prompt_parser = subparsers.add_parser(
        "compare-prompts",
        help="Compare zero-shot vs few-shot prompts; writes static ReAct examples.",
    )
    compare_prompt_parser.add_argument("--test-set", required=True, help="Fixed prompt JSONL path.")
    compare_prompt_parser.add_argument("--out", required=True, help="Output directory.")
    _add_generation_options(compare_prompt_parser)
    compare_prompt_parser.set_defaults(func=cmd_compare_prompts)

    compare_model_parser = subparsers.add_parser(
        "compare-models",
        help="Create model/runtime/adapter comparison artifacts.",
    )
    compare_model_parser.add_argument("--test-set", required=True, help="Fixed prompt JSONL path.")
    compare_model_parser.add_argument("--out", required=True, help="Output directory.")
    compare_model_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated name=tag entries. Adapter rows are inferred from names containing lora or adapter.",
    )
    _add_generation_options(compare_model_parser)
    compare_model_parser.set_defaults(func=cmd_compare_models)

    return parser


def main() -> None:
    """Parse arguments and dispatch one CLI command."""
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except SchemaError as exc:
        raise SystemExit(f"Schema validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
