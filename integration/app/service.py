"""Production service layer shared by the CLI and FastAPI.

Constructs request-scoped providers, manages session logging, and persists
analysis artifacts under ``integration/outputs/``. Each public entry point
mirrors one user-facing command (analyze PDF, topic search, chat, peer review).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from . import pipeline
from .contracts import ParsedPaper
from .io_paths import get_trace, log, reset_trace, write_json, write_text
from .providers.container import Providers
from .providers.live_providers import (
    LivePaperSource,
    SubprocessPdfParser,
    SubprocessRecommender,
    SubprocessSynthesizer,
    _integration_root,
    _repo_root,
    write_corpus_slice,
)
from .render import render_markdown
from .session_log import SessionLogger, new_timestamp_id, set_active

DEFAULT_CORPUS = (
    _repo_root()
    / "modules"
    / "dataset"
    / "data"
    / "processed"
    / "dev_5k_balanced.jsonl"
)
DEFAULT_QUERY = "retrieval augmented generation for scientific literature"
DEFAULT_RETRIEVAL_STRATEGY = "hybrid_rrf"
VALID_RETRIEVAL_STRATEGIES = frozenset({"hybrid_rrf", "tfidf"})


def _outputs_dir() -> Path:
    return _integration_root() / "outputs"


def _output_artifact(name: str) -> str:
    return str(_outputs_dir() / name)


def production_provider_sources() -> dict[str, str]:
    """Describe the source labels used by production requests."""
    return {
        "paper_source": "file-backed",
        "parser": "live",
        "recommender": "live",
        "synthesizer": "live",
    }


def _request_session(
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    redact_values: list[str] | None = None,
) -> tuple[SessionLogger, bool]:
    turn_id = new_timestamp_id()
    if session_id:
        return (
            SessionLogger.resume(
                root=_integration_root(),
                session_id=session_id,
                redact_values=redact_values,
                turn_id=turn_id,
            ),
            True,
        )
    return (
        SessionLogger.create(
            root=_integration_root(),
            run_id=run_id,
            redact_values=redact_values,
            turn_id=turn_id,
        ),
        False,
    )


def _finish_request_session(session: SessionLogger, persistent: bool) -> None:
    if persistent:
        session.checkpoint()
    else:
        session.close()


def _temporary_upload_redactions(pdf_path: str) -> list[str]:
    """Return path aliases only for temporary uploads."""
    original = Path(pdf_path)
    resolved = original.resolve()
    if original.parent.resolve() != Path(tempfile.gettempdir()).resolve():
        return []
    aliases = [str(original), str(original.absolute()), str(resolved)]
    return list(dict.fromkeys(value for value in aliases if value))


def _resolve_corpus(corpus: str | None) -> Path:
    if corpus:
        path = Path(corpus)
        if not path.is_absolute():
            path = (_integration_root() / path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Dataset production corpus missing: {path}. "
                "Expected a PaperRecord JSONL input"
            )
        return path
    path = DEFAULT_CORPUS.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset production corpus missing: {path}. "
            "Expected modules/dataset/data/processed/dev_5k_balanced.jsonl"
        )
    return path


def _parser_provider(session: SessionLogger) -> SubprocessPdfParser:
    return SubprocessPdfParser(
        session=session,
        integration_outputs=_outputs_dir(),
    )


def _synthesizer_provider(
    *,
    llm_model: str,
    style: str,
    session: SessionLogger,
    query_analysis: dict | None = None,
    recommendation_limit: int = 5,
    prompt_strategy: str = "zero_shot",
) -> SubprocessSynthesizer:
    return SubprocessSynthesizer(
        model=llm_model,
        style=style,
        recommendation_limit=recommendation_limit,
        query_analysis=query_analysis,
        prompt_strategy=prompt_strategy,
        session=session,
        integration_outputs=_outputs_dir(),
    )


def _retrieval_providers(
    *,
    query: str,
    corpus_jsonl: Path,
    corpus_limit: int | None,
    session: SessionLogger,
    embedding_model: str | None = None,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    top_k: int = 10,
    providers: Providers | None = None,
) -> tuple[Providers, Path]:
    providers = providers or Providers()
    poc_corpus = (
        _integration_root()
        / "data"
        / "processed"
        / "poc_corpus.jsonl"
    )
    write_corpus_slice(corpus_jsonl, poc_corpus, corpus_limit)
    providers.add("paper_source", LivePaperSource(poc_corpus), "file-backed")
    providers.add(
        "recommender",
        SubprocessRecommender(
            query=query,
            corpus_jsonl=poc_corpus,
            embedding_model=embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            session=session,
            integration_outputs=_outputs_dir(),
        ),
        "live",
    )
    return providers, poc_corpus


def _topic_providers(
    *,
    query: str,
    corpus_jsonl: Path,
    corpus_limit: int | None,
    llm_model: str,
    style: str,
    session: SessionLogger,
    query_analysis: dict | None = None,
    embedding_model: str | None = None,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    top_k: int = 5,
    prompt_strategy: str = "zero_shot",
) -> tuple[Providers, Path]:
    providers, poc_corpus = _retrieval_providers(
        query=query,
        corpus_jsonl=corpus_jsonl,
        corpus_limit=corpus_limit,
        session=session,
        embedding_model=embedding_model,
        retrieval_strategy=retrieval_strategy,
        top_k=top_k,
    )
    providers.add(
        "synthesizer",
        _synthesizer_provider(
            llm_model=llm_model,
            style=style,
            session=session,
            query_analysis=query_analysis,
            recommendation_limit=top_k,
            prompt_strategy=prompt_strategy,
        ),
        "live",
    )
    return providers, poc_corpus


def recommend_for_parsed(
    parsed: ParsedPaper,
    *,
    corpus: str | None = None,
    retrieval_embedding_model: str | None = None,
    retrieval_top_k: int = 10,
    retrieval_mode: str = "offline",
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
):
    """Run the retrieval recommender using the parsed paper title."""
    session = SessionLogger.create(root=_integration_root())
    set_active(session)
    try:
        providers, _ = _retrieval_providers(
            query=parsed.title,
            corpus_jsonl=_resolve_corpus(corpus),
            corpus_limit=None,
            session=session,
            embedding_model=retrieval_embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=retrieval_top_k,
        )
        return pipeline.recommend_for_parsed(
            providers,
            parsed,
            retrieval_mode=retrieval_mode,
        )
    finally:
        set_active(None)
        session.close()


def chat_topic(
    question: str,
    *,
    corpus: str | None = None,
    corpus_limit: int | None = None,
    retrieval_embedding_model: str | None = None,
    retrieval_top_k: int = 5,
    llm_model: str = "qwen3:8b",
    style: str = "auto",
    retrieval_mode: str = "offline",
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    session_id: str | None = None,
    prompt_strategy: str = "zero_shot",
) -> dict:
    """Route and answer one text turn."""
    session, persistent = _request_session(session_id=session_id)
    set_active(session)
    try:
        session.log_user_input(
            "chat",
            {
                "input_type": "text",
                "query": question,
                "llm_model": llm_model,
                "requested_style": style,
                "retrieval_mode": retrieval_mode,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_embedding_model": retrieval_embedding_model,
                "retrieval_top_k": retrieval_top_k,
                "prompt_strategy": prompt_strategy,
            },
        )
        session.log_user(question)
        analysis = pipeline.analyze_query(question, style_override=style)
        session.log_query_analysis(analysis.as_dict())

        is_recommendation = analysis.is_paper_recommendation
        retrieval_used = analysis.should_use_retrieval
        retrieval_query = question
        if is_recommendation:
            retrieval_query = (
                pipeline._query_understanding_module()
                .extract_recommendation_topic(question)
                or question
            )
        route = (
            "paper_recommendation_chat"
            if is_recommendation
            else "retrieval_augmented_chat"
            if retrieval_used
            else "direct_llm_chat"
        )
        session.log_route(
            input_type="text",
            route=route,
            retrieval_used=retrieval_used,
        )

        if retrieval_used:
            providers, _ = _topic_providers(
                query=retrieval_query,
                corpus_jsonl=_resolve_corpus(corpus),
                corpus_limit=None if is_recommendation else corpus_limit,
                llm_model=llm_model,
                style=analysis.style,
                session=session,
                query_analysis=analysis.as_dict(),
                embedding_model=retrieval_embedding_model,
                retrieval_strategy=retrieval_strategy,
                top_k=retrieval_top_k,
                prompt_strategy=prompt_strategy,
            )
            artifacts = [
                _output_artifact("recommendations.json"),
                _output_artifact("rag_evidence_pack.json"),
            ]
            if is_recommendation:
                artifacts.extend(
                    [
                        _output_artifact("paper_recommendation_cards.json"),
                        _output_artifact("paper_recommendation_generation.json"),
                    ]
                )
            else:
                artifacts.extend(
                    [
                        _output_artifact("llm_analysis.md"),
                        _output_artifact("analysis_result_from_llm.json"),
                        _output_artifact("llm_generation.json"),
                    ]
                )
        else:
            providers = Providers().add(
                "synthesizer",
                _synthesizer_provider(
                    llm_model=llm_model,
                    style=analysis.style,
                    session=session,
                    query_analysis=analysis.as_dict(),
                    prompt_strategy=prompt_strategy,
                ),
                "live",
            )
            artifacts = [
                _output_artifact("llm_chat_answer.md"),
                _output_artifact("llm_chat_answer_generation.json"),
            ]

        response = pipeline.chat_response(
            providers,
            question,
            retrieval_mode=retrieval_mode,
            query_analysis=analysis,
        )
        recommended = response.get("recommended_papers") or []
        citations = response.get("apa_citations") or [
            paper["apa_citation"]
            for paper in recommended
            if paper.get("apa_citation")
        ]
        session.log_outputs_recorded(
            artifact_paths=artifacts,
            recommended_count=len(recommended),
            citation_count=len(citations),
            route=route,
        )
        if response.get("kind") == "paper_recommendations":
            session.log_assistant_recommendations(response)
        elif route == "retrieval_augmented_chat" and (
            recommended or citations
        ):
            session.log_assistant_rag_message(response)
        else:
            session.log_assistant_text(response.get("answer") or "")
        return response
    except Exception as exc:
        session.log_request_failure(operation="chat", error=str(exc))
        raise
    finally:
        set_active(None)
        _finish_request_session(session, persistent)


def _pdf_providers(
    *,
    parser: SubprocessPdfParser,
    parsed: ParsedPaper,
    corpus_jsonl: Path,
    corpus_limit: int | None,
    llm_model: str,
    style: str,
    session: SessionLogger,
    with_related_papers: bool,
    embedding_model: str | None,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    top_k: int,
    prompt_strategy: str = "zero_shot",
) -> tuple[Providers, Path | None]:
    providers = Providers().add("parser", parser, "live")
    poc_corpus = None
    if with_related_papers:
        providers, poc_corpus = _retrieval_providers(
            query=pipeline.pdf_analysis_query(parsed),
            corpus_jsonl=corpus_jsonl,
            corpus_limit=corpus_limit,
            session=session,
            embedding_model=embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=top_k,
            providers=providers,
        )
    providers.add(
        "synthesizer",
        _synthesizer_provider(
            llm_model=llm_model,
            style=style,
            session=session,
            recommendation_limit=top_k,
            prompt_strategy=prompt_strategy,
        ),
        "live",
    )
    return providers, poc_corpus


def _log_analysis_output(session: SessionLogger, result) -> None:
    session.log_outputs_recorded(
        artifact_paths=[
            _output_artifact("analysis_result.json"),
            _output_artifact("demo_report.md"),
            _output_artifact("demo_trace.json"),
            _output_artifact("basic_nlp.json"),
            _output_artifact("structural_review.json"),
        ],
        recommended_count=len(result.recommended_papers or []),
        citation_count=len(result.apa_citations or []),
    )
    session.log_assistant_analysis(result.to_dict())


def run_analyze_pdf(
    pdf_path: str,
    *,
    corpus: str | None = None,
    corpus_limit: int | None = None,
    retrieval_embedding_model: str | None = None,
    retrieval_top_k: int = 5,
    llm_model: str = "qwen3:8b",
    style: str = "auto",
    retrieval_mode: str = "offline",
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    model_mode: str = "base",
    with_related_papers: bool | None = None,
    with_peer_review: bool = False,
    run_id: str | None = None,
    session_id: str | None = None,
    upload_name: str | None = None,
    prompt_strategy: str = "zero_shot",
) -> dict:
    """Parse first, then optionally retrieve related papers and synthesize."""
    if with_related_papers is None:
        with_related_papers = retrieval_mode != "none"
    corpus_path = _resolve_corpus(corpus) if with_related_papers else None
    session, persistent = _request_session(
        session_id=session_id,
        run_id=run_id,
        redact_values=_temporary_upload_redactions(pdf_path),
    )
    set_active(session)
    try:
        session.log_pdf_attachment(upload_name or Path(pdf_path).name)
        session.log_route(
            input_type="pdf",
            route=(
                "pdf_analysis_with_related_papers"
                if with_related_papers
                else "pdf_analysis_paper_only"
            ),
            retrieval_used=with_related_papers,
        )

        parser = _parser_provider(session)
        log("parse uploaded PDF", f"source=live path={pdf_path!r}")
        parsed = parser.parse(pdf_path)
        retrieval_query = pipeline.pdf_analysis_query(parsed)
        session.log_user_input(
            "analyze-pdf",
            {
                "pdf_path": pdf_path,
                "style": style,
                "retrieval_mode": retrieval_mode,
                "llm_model": llm_model,
                "with_related_papers": with_related_papers,
                "with_peer_review": with_peer_review,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_embedding_model": retrieval_embedding_model,
                "retrieval_top_k": retrieval_top_k,
                "prompt_strategy": prompt_strategy,
                "model_mode": model_mode,
                "retrieval_query": retrieval_query,
            },
        )
        providers, _ = _pdf_providers(
            parser=parser,
            parsed=parsed,
            corpus_jsonl=corpus_path or DEFAULT_CORPUS,
            corpus_limit=corpus_limit,
            llm_model=llm_model,
            style=style,
            session=session,
            with_related_papers=with_related_papers,
            embedding_model=retrieval_embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=retrieval_top_k,
            prompt_strategy=prompt_strategy,
        )

        reset_trace()
        log(
            "analyze-pdf START",
            f"path={pdf_path!r} parser=live q={retrieval_query!r}",
        )
        result = pipeline.analyze_parsed(
            providers,
            parsed,
            pdf_path,
            style=style,
            retrieval_mode=retrieval_mode,
            model_mode=model_mode,
            retrieval_strategy=retrieval_strategy,
            with_related_papers=with_related_papers,
            with_peer_review=with_peer_review,
        )
        trace = get_trace()
        write_json("analysis_result.json", result)
        write_text("demo_report.md", render_markdown(result))
        write_json(
            "demo_trace.json",
            {"steps": trace, "session": session.session_path},
        )
        _log_analysis_output(session, result)
        return {
            "result": result,
            "session_path": session.session_path,
            "trace": trace,
        }
    except Exception as exc:
        session.log_request_failure(operation="analyze-pdf", error=str(exc))
        raise
    finally:
        set_active(None)
        _finish_request_session(session, persistent)


def run_peer_review_pdf(
    pdf_path: str,
    *,
    llm_model: str = "qwen3:8b",
    style: str = "auto",
    model_mode: str = "base",
    prompt_strategy: str = "few_shot",
) -> dict:
    """Parse a PDF and return peer-review feedback plus session metadata."""
    session = SessionLogger.create(root=_integration_root())
    set_active(session)
    try:
        session.log_user_input(
            "peer-review",
            {
                "pdf_path": pdf_path,
                "style": style,
                "llm_model": llm_model,
                "model_mode": model_mode,
            },
        )
        session.log_route(
            input_type="pdf",
            route="pdf_peer_review",
            retrieval_used=False,
        )
        providers = Providers()
        providers.add("parser", _parser_provider(session), "live")
        providers.add(
            "synthesizer",
            _synthesizer_provider(
                llm_model=llm_model,
                style=style,
                session=session,
                prompt_strategy=prompt_strategy,
            ),
            "live",
        )
        reset_trace()
        feedback = pipeline.peer_review(
            providers,
            pdf_path,
            model_mode=model_mode,
        )
        trace = get_trace()
        peer_path = write_text(
            "peer_review.md",
            f"# Peer-review assistance\n\n{feedback}\n",
        )
        session.log_outputs_recorded(
            artifact_paths=[peer_path],
            recommended_count=0,
            citation_count=0,
        )
        return {
            "feedback": feedback,
            "session_path": session.session_path,
            "trace": trace,
        }
    except Exception as exc:
        session.log_request_failure(operation="peer-review", error=str(exc))
        raise
    finally:
        set_active(None)
        session.close()


def run_pdf_chat(
    question: str,
    paper_json: str,
    *,
    llm_model: str = "qwen3:8b",
    style: str = "auto",
    session_id: str | None = None,
    prompt_strategy: str = "zero_shot",
) -> str:
    """Answer a question grounded in a previously parsed paper JSON file."""
    with open(paper_json, encoding="utf-8") as handle:
        parsed = ParsedPaper.from_dict(json.load(handle))
    session, persistent = _request_session(session_id=session_id)
    set_active(session)
    try:
        session.log_user_input(
            "chat-pdf",
            {
                "question": question,
                "paper_json": paper_json,
                "llm_model": llm_model,
                "style": style,
            },
        )
        session.log_user(question)
        session.log_route(
            input_type="pdf",
            route="pdf_grounded_chat",
            retrieval_used=False,
        )
        analysis = pipeline.analyze_query(question, style_override=style)
        query_analysis = analysis.as_dict()
        query_analysis["intent"] = "question"
        query_analysis["confidence"] = 1.0
        query_analysis.setdefault("field_confidences", {})["intent"] = 1.0
        query_analysis.setdefault("field_sources", {})["intent"] = (
            "pdf_grounded_route"
        )
        session.log_query_analysis(query_analysis)
        providers = Providers().add(
            "synthesizer",
            _synthesizer_provider(
                llm_model=llm_model,
                style=style,
                session=session,
                query_analysis=query_analysis,
                prompt_strategy=prompt_strategy,
            ),
            "live",
        )
        answer = pipeline.chat(providers, question, parsed=parsed)
        session.log_assistant_text(answer)
        return answer
    except Exception as exc:
        session.log_request_failure(operation="chat-pdf", error=str(exc))
        raise
    finally:
        set_active(None)
        _finish_request_session(session, persistent)


def run_topic(
    *,
    query: str = DEFAULT_QUERY,
    corpus: str | None = None,
    corpus_limit: int | None = None,
    retrieval_embedding_model: str | None = None,
    retrieval_top_k: int = 5,
    llm_model: str = "qwen3:8b",
    style: str = "auto",
    retrieval_mode: str = "offline",
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    model_mode: str = "base",
    run_id: str | None = None,
    prompt_strategy: str = "zero_shot",
) -> dict:
    """Execute production topic search and persist its trace."""
    corpus_path = _resolve_corpus(corpus)
    session = SessionLogger.create(root=_integration_root(), run_id=run_id)
    set_active(session)
    try:
        session.log_user_input(
            "run",
            {
                "mode": "topic_search",
                "query": query,
                "corpus": str(corpus_path),
                "corpus_limit": corpus_limit,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_embedding_model": retrieval_embedding_model,
                "retrieval_top_k": retrieval_top_k,
                "llm_model": llm_model,
                "style": style,
                "retrieval_mode": retrieval_mode,
                "model_mode": model_mode,
                "prompt_strategy": prompt_strategy,
            },
        )
        session.log_user(query)
        providers, poc_corpus = _topic_providers(
            query=query,
            corpus_jsonl=corpus_path,
            corpus_limit=corpus_limit,
            llm_model=llm_model,
            style=style,
            session=session,
            embedding_model=retrieval_embedding_model,
            retrieval_strategy=retrieval_strategy,
            top_k=retrieval_top_k,
            prompt_strategy=prompt_strategy,
        )

        reset_trace()
        log(
            "run START",
            f"query={query!r} corpus={corpus_path} slice={poc_corpus}",
        )
        result = pipeline.search_topic(
            providers,
            query,
            style=style,
            retrieval_mode=retrieval_mode,
            model_mode=model_mode,
            retrieval_strategy=retrieval_strategy,
        )
        log("run DONE", f"sources={providers.sources()}")
        trace = get_trace()
        write_json("analysis_result.json", result)
        write_text("demo_report.md", render_markdown(result))
        write_json(
            "demo_trace.json",
            {"steps": trace, "session": session.session_path},
        )
        _log_analysis_output(session, result)
        session.log_meta(
            "run_complete",
            {
                "analysis_result": _output_artifact("analysis_result.json"),
                "provider_sources": providers.sources(),
            },
        )
        return {
            "result": result,
            "session_path": session.session_path,
            "trace": trace,
            "corpus_path": corpus_path,
        }
    finally:
        set_active(None)
        session.close()
