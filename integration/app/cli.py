"""CLI entry point for the Use Case 3 system.

    python -m app.cli analyze-pdf ../tests/papers/drq_v2/2107.09645v1.pdf
    python -m app.cli search-topic "retrieval augmented generation for science"
    python -m app.cli recommend --paper-json ../tests/papers/artifacts/drq_v2_parsed.json
    python -m app.cli peer-review ../tests/papers/siga/SIGA_....pdf
    python -m app.cli chat --paper-json ../tests/papers/artifacts/drq_v2_parsed.json "what is the main idea?"

Production commands use live module providers (analyze-paper, recommend-topic, synthesize).
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import typer
except ImportError:  # clear setup message (Stage 01 requirement)
    raise SystemExit(
        "Missing dependency 'typer'. Install with:\n"
        "    pip install -r requirements.txt  # from the repository root\n"
        "or: pip install typer"
    )

from .contracts import ParsedPaper
from .io_paths import write_json, write_text
from .render import format_run_output, format_topic_cli_command, render_markdown

app = typer.Typer(add_completion=False,
                  help="Research Paper Analysis & Recommendation — local CLI.")


def _persist(result, also_report=False):
    """Write analysis_result.json (and optionally demo_report.md)."""
    json_path = write_json("analysis_result.json", result)
    typer.echo(f"wrote {json_path}")
    if also_report:
        md_path = write_text("demo_report.md", render_markdown(result))
        typer.echo(f"wrote {md_path}")


@app.command("analyze-pdf")
def analyze_pdf(
    pdf_path: str,
    style: str = "auto",
    retrieval_mode: str = "offline",
    model_mode: str = "base",
    llm_model: str = typer.Option("qwen3:8b", "--llm-model"),
    no_related_papers: bool = typer.Option(
        False,
        "--no-related-papers",
        help="Skip retrieval; use paper-only LLM synthesis.",
    ),
):
    """Analyze an uploaded PDF -> AnalysisResult."""
    from . import service as service

    out = service.run_analyze_pdf(
        pdf_path,
        style=style,
        retrieval_mode="none" if no_related_papers else retrieval_mode,
        model_mode=model_mode,
        llm_model=llm_model,
        with_related_papers=not no_related_papers,
    )
    _persist(out["result"], also_report=True)


@app.command("search-topic")
def search_topic(query: str, style: str = "auto",
                 retrieval_mode: str = "offline", model_mode: str = "base"):
    """Topic/text search -> AnalysisResult (related work + summary)."""
    from . import service
    out = service.run_topic(
        query=query,
        style=style,
        retrieval_mode=retrieval_mode,
        model_mode=model_mode,
    )
    _persist(out["result"], also_report=True)


@app.command("recommend")
def recommend(paper_json: str = typer.Option(..., "--paper-json"),
              retrieval_mode: str = "offline"):
    """Recommend related papers for a parsed paper JSON file."""
    with open(paper_json) as f:
        parsed = ParsedPaper.from_dict(json.load(f))
    from . import service
    rec = service.recommend_for_parsed(parsed, retrieval_mode=retrieval_mode)
    path = write_json("recommendations.json", rec)
    typer.echo(f"wrote {path}")
    for it in rec.items:
        typer.echo(f"  {it['score']:.2f}  {it['title']}")


@app.command("peer-review")
def peer_review(
    pdf_path: str,
    model_mode: str = "base",
    llm_model: str = typer.Option("qwen3:8b", "--llm-model"),
    style: str = typer.Option("auto", "--style"),
):
    """Peer-review assistance for an uploaded PDF."""
    from . import service as service

    out = service.run_peer_review_pdf(
        pdf_path,
        model_mode=model_mode,
        llm_model=llm_model,
        style=style,
    )
    typer.echo(f"wrote outputs/peer_review.md")
    typer.echo(out["feedback"])


@app.command("chat")
def chat(
    question: str,
    paper_json: str = typer.Option(None, "--paper-json"),
    llm_model: str = typer.Option("qwen3:8b", "--llm-model"),
    style: str = typer.Option("auto", "--style"),
):
    """Ask a follow-up question (optionally grounded in a parsed paper)."""
    from . import service as service

    if paper_json:
        response = {
            "answer": service.run_pdf_chat(
                question,
                paper_json,
                llm_model=llm_model,
                style=style,
            )
        }
    else:
        response = service.chat_topic(
            question,
            llm_model=llm_model,
            style=style,
        )
    typer.echo(response.get("answer", ""))
    for index, paper in enumerate(response.get("recommended_papers") or [], start=1):
        typer.echo(f"\n{index}. {paper.get('title', 'Untitled paper')}")
        if paper.get("apa_citation"):
            typer.echo(f"   {paper['apa_citation']}")
        if paper.get("summary"):
            typer.echo(f"   {paper['summary']}")


@app.command("run")
def run_cmd(
    query: str = typer.Option(
        "retrieval augmented generation for scientific literature",
        "--query",
        help="Topic query (Use Case 3 text input mode).",
    ),
    corpus: str = typer.Option(
        None,
        "--corpus",
        help="Path to dev_5k.jsonl corpus (default: ../modules/dataset/.../dev_5k.jsonl).",
    ),
    corpus_limit: int = typer.Option(
        0,
        "--corpus-limit",
        help="Use first N real PaperRecords for demo latency (0 = no limit).",
    ),
    llm_model: str = typer.Option("qwen3:8b", "--llm-model"),
    style: str = typer.Option("auto", "--style"),
    retrieval_mode: str = typer.Option("offline", "--retrieval-mode"),
    retrieval_strategy: str = typer.Option(
        "hybrid_rrf",
        "--retrieval-strategy",
        help="Ranking strategy: hybrid_rrf (default) or tfidf baseline.",
    ),
    model_mode: str = typer.Option("base", "--model-mode"),
    top_n: int = typer.Option(5, "--top-n", help="Recommendations shown in terminal output."),
):
    """Topic search: real corpus + retrieval + LLM synthesis (Ollama) + session JSONL."""
    from . import service as service

    limit = None if corpus_limit == 0 else corpus_limit
    out = service.run_topic(
        query=query,
        corpus=corpus,
        corpus_limit=limit,
        llm_model=llm_model,
        style=style,
        retrieval_mode=retrieval_mode,
        retrieval_strategy=retrieval_strategy,
        model_mode=model_mode,
    )
    cmd = format_topic_cli_command(
        "run",
        query=query,
        corpus=str(out["corpus_path"]) if corpus else None,
        corpus_limit=limit,
        llm_model=llm_model,
        style=style,
        retrieval_mode=retrieval_mode,
        model_mode=model_mode,
        top_n=top_n,
    )
    typer.echo("")
    typer.echo(
        format_run_output(
            out["result"],
            trace=out.get("trace"),
            session_path=out.get("session_path"),
            command_line=cmd,
            top_n=top_n,
        )
    )
    typer.echo("")
    typer.echo(f"session log: {out['session_path']}")
    typer.echo(
        "wrote outputs/analysis_result.json, demo_report.md, demo_trace.json, "
        "outputs/llm_analysis.md"
    )


@app.command("integration-status")
def integration_status():
    """Report file-backed and live provider status."""
    from .integration import write_integration_status
    typer.echo(f"wrote {write_integration_status()}")


@app.command("build-artifacts")
def build_artifacts():
    """Regenerate documentation artifacts under outputs/."""
    from . import commands_members, privacy, api as api_mod, integration, docs_gen, frontend
    paths = [
        commands_members.write_command_matrix(),
        privacy.write_privacy_check(),
        api_mod.write_api_contract(),
        frontend.write_frontend_notes(),
        integration.write_integration_status(),
        docs_gen.write_video_script(),
        docs_gen.write_packaging_checklist(),
    ]
    for p in paths:
        typer.echo(f"wrote {p}")


@app.command("web")
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Address to bind."),
    port: int = typer.Option(8000, "--port", min=1, max=65535, help="Port to bind."),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Rebuild frontend assets even when the current build is fresh.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Restart the backend automatically when Python files change.",
    ),
):
    """Build and serve the web UI and API at one local URL."""
    from .web import run_web

    typer.echo(f"Serving web app at http://{host}:{port}")
    try:
        run_web(host=host, port=port, rebuild=rebuild, reload=reload)
    except RuntimeError as exc:
        raise typer.ClickException(str(exc)) from exc


@app.command("session-inspect")
def session_inspect(
    run_id: str = typer.Argument(None),
    component: str = typer.Option(None, "--component"),
    status: str = typer.Option(None, "--status"),
):
    """Inspect a structured session timeline with optional filters."""
    sessions = Path(__file__).resolve().parents[1] / "data" / "sessions"
    if run_id:
        session_path = sessions / run_id / "session.jsonl"
    else:
        candidates = sorted(sessions.glob("*/session.jsonl"))
        if not candidates:
            raise typer.BadParameter("No integration sessions found")
        session_path = candidates[-1]
    if not session_path.is_file():
        raise typer.BadParameter(f"Session not found: {session_path}")
    rows = []
    for line in session_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if component and row.get("component") != component:
            continue
        if status and row.get("status") != status:
            continue
        rows.append(row)
    typer.echo(f"session: {session_path}")
    for row in rows:
        duration = row.get("duration_ms")
        elapsed = f" {float(duration):.1f}ms" if duration is not None else ""
        typer.echo(
            f"{row.get('timestamp', row.get('ts', ''))} "
            f"[{row.get('component', '-')}/{row.get('phase', '-')}] "
            f"{row.get('status', '-')}: {row.get('message', '')}{elapsed}"
        )


# Stage 02: attach every member-owned command to this same entry point.
from . import commands_members  # noqa: E402
commands_members.register(app)


if __name__ == "__main__":
    app()
