"""CLI for production PDF parsing, NLP enrichment, evaluation, and model setup."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from model_assets import (  # noqa: E402
    ModelAssetError,
    install_model_archive,
    validate_model_assets,
)
from paper_analysis import (  # noqa: E402
    analyze_parsed_paper,
    structural_checks,
)
from pdf_parser import PdfParserError, build_parsed_paper  # noqa: E402

from .session_append import append_session_event, _verbose_session_log  # noqa: E402


def _import_validator():
    llm_schemas = (
        Path(__file__).resolve().parents[2] / "llm" / "app" / "schemas.py"
    )
    if not llm_schemas.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("llm_schemas", llm_schemas)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_parsed_paper


def cmd_parse_pdf(args: argparse.Namespace) -> None:
    """Parse one PDF without loading any NLP models."""
    pdf_path = Path(args.pdf).resolve()
    out_path = Path(args.out).resolve()
    debug_path = Path(args.debug_out).resolve()

    if _verbose_session_log():
        append_session_event(
            "user_input",
            {"command": "parse-pdf", "pdf_path": str(pdf_path), "out": str(out_path)},
            phase="parse",
            status="started",
            message="Parsing uploaded PDF",
        )

    try:
        parsed, debug = build_parsed_paper(pdf_path)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    except PdfParserError as exc:
        raise SystemExit(str(exc)) from exc

    validator = _import_validator()
    if validator is not None:
        validator(parsed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    debug_path.write_text(json.dumps(debug, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    append_session_event(
        "parse_complete",
        {
            "title": parsed["metadata"].get("title", ""),
            "ref_count": len(parsed.get("references") or []),
            "section_keys": list((parsed.get("sections") or {}).keys()),
        },
        phase="parse",
        status="completed",
        metrics={
            "page_count": debug["page_count"],
            "reference_count": len(parsed.get("references") or []),
        },
        artifacts=[str(out_path), str(debug_path)],
        message="PDF parsing completed",
    )
    if _verbose_session_log():
        append_session_event(
            "user_output",
            {
                "artifact_paths": [str(out_path), str(debug_path)],
                "summary_preview": (parsed["metadata"].get("title") or "")[:500],
            },
            phase="parse",
            status="completed",
            artifacts=[str(out_path), str(debug_path)],
        )

    print(f"Wrote ParsedPaper: {out_path}")
    print(f"Wrote parse debug: {debug_path}")
    print(f"pages={debug['page_count']} title={parsed['metadata']['title']!r}")


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _analysis_event(phase: str, status: str, details: dict) -> None:
    if status == "started" and not _verbose_session_log():
        return
    append_session_event(
        f"{phase}_{status}",
        {},
        phase=phase,
        status=status,
        duration_ms=details.get("duration_ms"),
        error=details.get("error"),
        message=f"{phase.replace('_', ' ')} {status}",
    )


def cmd_basic_nlp(args: argparse.Namespace) -> None:
    """Enrich one existing ParsedPaper with production NLP evidence."""
    paper_path = Path(args.paper_json).resolve()
    out_path = Path(args.out).resolve()
    paper = _read_json(paper_path, "ParsedPaper")
    validator = _import_validator()
    if validator is not None:
        validator(paper)
    try:
        enriched, analysis = analyze_parsed_paper(
            paper,
            event_callback=_analysis_event,
        )
    except (ModelAssetError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    _write_json(out_path, analysis)
    enriched_path = Path(args.enriched_paper_out).resolve() if args.enriched_paper_out else None
    artifacts = [str(out_path)]
    if enriched_path:
        _write_json(enriched_path, enriched)
        artifacts.append(str(enriched_path))
    if _verbose_session_log():
        append_session_event(
            "user_output",
            {"artifact_paths": artifacts},
            phase="basic_nlp",
            status="completed",
            artifacts=artifacts,
            metrics={
                "keyword_count": len(enriched.get("keywords") or []),
                "entity_count": sum(
                    len(values) for values in (enriched.get("entities") or {}).values()
                ),
            },
        )
    print(f"Wrote basic NLP analysis: {out_path}")
    if enriched_path:
        print(f"Wrote enriched ParsedPaper: {enriched_path}")


def cmd_peer_review_checks(args: argparse.Namespace) -> None:
    """Run deterministic structural checks without invoking an LLM."""
    paper_path = Path(args.paper_json).resolve()
    out_path = Path(args.out).resolve()
    paper = _read_json(paper_path, "ParsedPaper")
    if _verbose_session_log():
        append_session_event(
            "structural_checks_started",
            {},
            phase="structural_checks",
            status="started",
        )
    checks = structural_checks(paper)
    _write_json(out_path, {"checks": checks, "count": len(checks)})
    append_session_event(
        "structural_checks_completed",
        {},
        phase="structural_checks",
        status="completed",
        metrics={"finding_count": len(checks)},
        artifacts=[str(out_path)],
    )
    print(f"Wrote structural review: {out_path}")


def cmd_analyze_paper(args: argparse.Namespace) -> None:
    """Parse a real PDF and enrich the resulting ParsedPaper in one command."""
    pdf_path = Path(args.pdf).resolve()
    out_path = Path(args.out).resolve()
    debug_path = Path(args.debug_out).resolve()
    analysis_path = Path(args.analysis_out).resolve()
    review_path = Path(args.review_out).resolve()
    if _verbose_session_log():
        append_session_event(
            "analyze_paper_started",
            {"command": "analyze-paper", "pdf_path": str(pdf_path)},
            phase="analyze_paper",
            status="started",
            message="Starting production PDF-NLP analysis",
        )
    try:
        parsed, debug = build_parsed_paper(pdf_path)
        append_session_event(
            "parse_complete",
            {},
            phase="parse",
            status="completed",
            metrics={
                "page_count": debug["page_count"],
                "reference_count": len(parsed.get("references") or []),
            },
        )
        enriched, analysis = analyze_parsed_paper(
            parsed,
            event_callback=_analysis_event,
        )
    except (FileNotFoundError, PdfParserError, ModelAssetError, RuntimeError) as exc:
        append_session_event(
            "analyze_paper_failed",
            {},
            phase="analyze_paper",
            status="failed",
            error=str(exc),
        )
        raise SystemExit(str(exc)) from exc
    validator = _import_validator()
    if validator is not None:
        validator(enriched)
    _write_json(out_path, enriched)
    _write_json(debug_path, debug)
    _write_json(analysis_path, analysis)
    _write_json(
        review_path,
        {
            "checks": analysis["structural_checks"],
            "count": len(analysis["structural_checks"]),
        },
    )
    artifacts = [str(out_path), str(debug_path), str(analysis_path), str(review_path)]
    append_session_event(
        "analyze_paper_completed",
        {},
        phase="analyze_paper",
        status="completed",
        metrics={
            "page_count": debug["page_count"],
            "keyword_count": len(enriched["keywords"]),
            "entity_count": sum(len(values) for values in enriched["entities"].values()),
            "structural_finding_count": len(analysis["structural_checks"]),
        },
        artifacts=artifacts,
    )
    if _verbose_session_log():
        append_session_event(
            "user_output",
            {"artifact_paths": artifacts},
            phase="analyze_paper",
            status="completed",
            artifacts=artifacts,
        )
    print(f"Wrote enriched ParsedPaper: {out_path}")
    print(f"Wrote basic NLP analysis: {analysis_path}")
    print(f"Wrote structural review: {review_path}")


def cmd_model_assets(args: argparse.Namespace) -> None:
    """Install a supplied model archive or verify the current runtime assets."""
    try:
        result = (
            install_model_archive(Path(args.archive))
            if args.archive
            else validate_model_assets()
        )
    except ModelAssetError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, ensure_ascii=True))
    if not result["valid"]:
        raise SystemExit("PDF-NLP model assets are not ready")


def cmd_evaluate_real_papers(args: argparse.Namespace) -> None:
    """Evaluate production PDF-NLP outputs on the declared real-paper corpus."""
    from evaluation import evaluate_real_papers

    try:
        report = evaluate_real_papers(
            Path(args.manifest).resolve(),
            Path(args.annotations).resolve(),
            Path(args.out).resolve(),
        )
    except (ModelAssetError, RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote real-paper evaluation: {report}")


def build_parser() -> argparse.ArgumentParser:
    """Build the module command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Nadiyah PDF parser commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_pdf = subparsers.add_parser("parse-pdf", help="Parse a PDF into ParsedPaper JSON.")
    parse_pdf.add_argument("--pdf", required=True, help="Path to the uploaded PDF.")
    parse_pdf.add_argument(
        "--out",
        default="outputs/parsed_paper.json",
        help="Canonical ParsedPaper output path.",
    )
    parse_pdf.add_argument(
        "--debug-out",
        default="outputs/pdf_parse_debug.json",
        help="Debug JSON with page text and headings.",
    )
    parse_pdf.set_defaults(func=cmd_parse_pdf)

    basic_nlp = subparsers.add_parser(
        "basic-nlp",
        help="Run POS, NER, keyphrases, and extractive summary on ParsedPaper JSON.",
    )
    basic_nlp.add_argument("--paper-json", required=True)
    basic_nlp.add_argument("--out", default="outputs/basic_nlp.json")
    basic_nlp.add_argument(
        "--enriched-paper-out",
        default=None,
        help="Optional path for ParsedPaper with populated keywords/entities/analysis.",
    )
    basic_nlp.set_defaults(func=cmd_basic_nlp)

    review = subparsers.add_parser(
        "peer-review-checks",
        help="Run deterministic structural checks on ParsedPaper JSON.",
    )
    review.add_argument("--paper-json", required=True)
    review.add_argument("--out", default="outputs/structural_review.json")
    review.set_defaults(func=cmd_peer_review_checks)

    analyze = subparsers.add_parser(
        "analyze-paper",
        help="Parse and enrich one real PDF with the production PDF-NLP pipeline.",
    )
    analyze.add_argument("--pdf", required=True)
    analyze.add_argument("--out", default="outputs/parsed_paper.json")
    analyze.add_argument("--debug-out", default="outputs/pdf_parse_debug.json")
    analyze.add_argument("--analysis-out", default="outputs/basic_nlp.json")
    analyze.add_argument("--review-out", default="outputs/structural_review.json")
    analyze.set_defaults(func=cmd_analyze_paper)

    assets = subparsers.add_parser(
        "model-assets",
        help="Install a local team ZIP or verify required PDF-NLP model assets.",
    )
    assets.add_argument("--archive", default=None)
    assets.set_defaults(func=cmd_model_assets)

    evaluate = subparsers.add_parser(
        "evaluate-real-papers",
        help="Run production evaluation on the five declared real PDFs.",
    )
    evaluate.add_argument(
        "--manifest",
        default="data/evaluation/real_papers.json",
    )
    evaluate.add_argument(
        "--annotations",
        default="data/evaluation/annotations.json",
    )
    evaluate.add_argument("--out", default="results/pdf_nlp")
    evaluate.set_defaults(func=cmd_evaluate_real_papers)
    return parser


def main() -> None:
    """Parse command-line arguments and execute the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
