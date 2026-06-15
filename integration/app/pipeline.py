"""Pure orchestration for topic, PDF, and conversational analysis."""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

from .contracts import AnalysisResult, ParsedPaper, RagEvidencePack, Recommendation
from .io_paths import log
from .providers.container import Providers
from .providers.live_providers import (
    evidence_pack_to_rag,
    raw_recommendations_to_contract,
)

PDF_PARSER_UNAVAILABLE = (
    "live PDF parser unavailable: PDF analysis, peer review and PDF-grounded "
    "chat are disabled until the PDF-NLP parser satisfies the ParsedPaper contract"
)


@lru_cache(maxsize=1)
def _query_understanding_module():
    """Load the LLM query analyzer without importing the other `app` package."""
    module_path = (
        Path(__file__).resolve().parents[2]
        / "modules"
        / "llm"
        / "app"
        / "query_understanding.py"
    )
    spec = importlib.util.spec_from_file_location("llm_query_understanding", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LLM query analyzer: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def analyze_query(question: str, *, style_override: str = "auto"):
    """Classify a message before configuring expensive providers."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return _query_understanding_module().analyze_query(
        question,
        style_override=style_override,
    )


def _flags(
    providers: Providers,
    retrieval_mode: str,
    model_mode: str,
    *,
    retrieval_strategy: str = "hybrid_rrf",
    include_parser: bool = False,
) -> dict:
    roles = ["paper_source", "recommender", "synthesizer"]
    if include_parser:
        roles.append("parser")
    sources = providers.sources()
    return {
        "retrieval_mode": retrieval_mode,
        "retrieval_strategy": retrieval_strategy,
        "model_mode": model_mode,
        "provider_sources": {
            role: sources[role]
            for role in roles
            if role in sources
        },
        "ai_generated_sections": [
            "summary",
            "key_findings",
            "research_gaps",
            "peer_review",
        ],
    }


def _analyze(
    providers: Providers,
    parsed: ParsedPaper | None,
    query: str,
    input_type: str,
    input_ref: str,
    *,
    candidates=None,
    style: str = "concise",
    retrieval_mode: str = "offline",
    model_mode: str = "base",
    retrieval_strategy: str = "hybrid_rrf",
    with_peer_review: bool = True,
    with_related_papers: bool = True,
) -> AnalysisResult:
    synthesizer = providers.require("synthesizer")
    evidence = RagEvidencePack(query=query, snippets=[], retrieval_mode="paper_only")
    recommendation = Recommendation(query=query, items=[])

    if with_related_papers and retrieval_mode != "none":
        recommender = providers.require("recommender")
        if candidates is None:
            candidates = []
            if providers.source("recommender") != "live":
                source = providers.require("paper_source")
                log(
                    "retrieving candidate corpus",
                    f"source={providers.source('paper_source')}",
                )
                candidates = source.get_corpus()

        log(
            "retrieving evidence (RAG)",
            f"source={providers.source('recommender')}",
        )
        evidence = recommender.retrieve_evidence(query, mode=retrieval_mode)
        log(
            "recommending related papers",
            f"source={providers.source('recommender')}",
        )
        recommendation = recommender.recommend(query, candidates)
    elif parsed is not None:
        log(
            "skipping related-paper retrieval",
            "paper-only synthesis path (no topic RAG replacement)",
        )

    log(
        "synthesizing summary/findings/gaps",
        f"source={providers.source('synthesizer')}",
    )
    synthesis = synthesizer.synthesize(
        parsed,
        query,
        evidence,
        recommendation,
        style=style,
        model_mode=model_mode,
    )

    peer_review = None
    if with_peer_review and parsed is not None:
        log(
            "peer-review assistance",
            f"source={providers.source('synthesizer')}",
        )
        peer_review = synthesizer.peer_review(parsed, model_mode=model_mode)

    metadata = {}
    if parsed is not None:
        metadata = {
            "title": parsed.title,
            "authors": parsed.authors,
            "paper_id": parsed.paper_id,
        }

    return AnalysisResult(
        input_type=input_type,
        input_ref=input_ref,
        metadata=metadata,
        summary=synthesis["summary"],
        key_findings=synthesis["key_findings"],
        research_gaps=synthesis["research_gaps"],
        recommended_papers=recommendation.items,
        apa_citations=[
            item["apa_citation"]
            for item in recommendation.items
            if item.get("apa_citation")
        ],
        evidence=evidence.snippets,
        peer_review=peer_review,
        paper_analysis=dict(parsed.analysis) if parsed is not None else {},
        flags=_flags(
            providers,
            retrieval_mode,
            model_mode,
            retrieval_strategy=retrieval_strategy,
            include_parser=parsed is not None,
        ),
    )


def pdf_analysis_query(parsed: ParsedPaper) -> str:
    """Use parsed title first and a bounded abstract only as fallback."""
    title = parsed.title.strip()
    if title:
        return title
    abstract = parsed.abstract.strip()
    if abstract:
        return abstract[:500]
    return "Summarize the uploaded paper."


def _rag_chat_sources(
    recommender,
    query: str,
    *,
    raw_candidates: list | None = None,
) -> tuple[list[dict], list[str]]:
    """Return deterministic bibliography metadata for the retrieved RAG result."""
    recommendation = (
        raw_recommendations_to_contract(query, raw_candidates)
        if raw_candidates is not None
        else recommender.recommend(query, [])
    )
    papers = list(recommendation.items)
    citations = [
        str(item["apa_citation"])
        for item in papers
        if item.get("apa_citation")
    ]
    return papers, citations


def analyze_parsed(
    providers: Providers,
    parsed: ParsedPaper,
    pdf_path: str,
    **options,
) -> AnalysisResult:
    return _analyze(
        providers,
        parsed,
        pdf_analysis_query(parsed),
        "pdf",
        pdf_path,
        with_peer_review=options.pop("with_peer_review", True),
        with_related_papers=options.pop(
            "with_related_papers",
            options.get("retrieval_mode", "offline") != "none",
        ),
        **options,
    )


def analyze_pdf(
    providers: Providers,
    pdf_path: str,
    **options,
) -> AnalysisResult:
    if providers.source_labels.get("parser") != "live":
        raise RuntimeError(PDF_PARSER_UNAVAILABLE)
    parser = providers.require("parser")
    log("parse uploaded PDF", f"source=live path={pdf_path!r}")
    return analyze_parsed(providers, parser.parse(pdf_path), pdf_path, **options)


def search_topic(providers: Providers, query: str, **options) -> AnalysisResult:
    log(
        "topic search",
        f"source={providers.source('paper_source')} q={query!r}",
    )
    return _analyze(
        providers,
        None,
        query,
        "topic",
        query,
        with_peer_review=False,
        with_related_papers=True,
        **options,
    )


def recommend_for_parsed(
    providers: Providers,
    parsed: ParsedPaper,
    retrieval_mode: str = "offline",
) -> Recommendation:
    recommender = providers.require("recommender")
    candidates = []
    if providers.source("recommender") != "live":
        candidates = providers.require("paper_source").get_corpus()
    log(
        "recommending for parsed paper",
        f"source={providers.source('recommender')}",
    )
    return recommender.recommend(parsed.title, candidates)


def peer_review(
    providers: Providers,
    pdf_path: str,
    model_mode: str = "base",
) -> str:
    if providers.source_labels.get("parser") != "live":
        raise RuntimeError(PDF_PARSER_UNAVAILABLE)
    parser = providers.require("parser")
    log("parse uploaded PDF for peer review", "source=live")
    parsed = parser.parse(pdf_path)
    return providers.require("synthesizer").peer_review(
        parsed,
        model_mode=model_mode,
    )


def chat(
    providers: Providers,
    question: str,
    parsed: ParsedPaper | None = None,
    retrieval_mode: str = "offline",
    query_analysis=None,
) -> str:
    return chat_response(
        providers,
        question,
        parsed=parsed,
        retrieval_mode=retrieval_mode,
        query_analysis=query_analysis,
    )["answer"]


def chat_response(
    providers: Providers,
    question: str,
    parsed: ParsedPaper | None = None,
    retrieval_mode: str = "offline",
    query_analysis=None,
) -> dict:
    synthesizer = providers.require("synthesizer")

    if parsed is not None:
        if providers.source("synthesizer") != "live":
            raise RuntimeError(
                "PDF-grounded chat requires live synthesizer for paper-aware answers"
            )
        log("PDF-grounded chat", "source=live")
        answer = synthesizer.answer(
            parsed,
            question,
            RagEvidencePack(question, []),
        )
        return {"kind": "message", "answer": answer, "recommended_papers": []}

    analysis = query_analysis or analyze_query(question)
    if analysis.is_paper_recommendation:
        topic = (
            _query_understanding_module().extract_recommendation_topic(question)
            or question
        )
        recommender = providers.require("recommender")
        log(
            "paper recommendation chat",
            f"source={providers.source('recommender')} topic={topic!r}",
        )
        recommendation = recommender.recommend(topic, [])
        papers = synthesizer.recommend_papers(topic, recommendation)
        return {
            "kind": "paper_recommendations",
            "answer": f"Here are {len(papers)} papers on {topic}:",
            "recommended_papers": papers,
        }

    if not analysis.should_use_retrieval:
        log("chat routed without retrieval", f"intent={analysis.intent}")
        return {
            "kind": "message",
            "answer": synthesizer.answer_direct(question),
            "recommended_papers": [],
        }

    recommender = providers.require("recommender")
    if hasattr(recommender, "search_offline_pack"):
        try:
            import os

            from .llm_bridge import build_llm_runtime, run_react_topic_rag

            model = getattr(synthesizer, "model", "qwen3:8b")
            host = os.getenv("COMP8420_OLLAMA_HOST", "http://127.0.0.1:11434")
            runtime = build_llm_runtime(host)
            _pack, plan = run_react_topic_rag(
                question,
                search_offline=recommender.search_offline_pack,
                runtime=runtime,
                model=model,
                default_top_k=getattr(recommender, "top_k", 5),
            )
            evidence = evidence_pack_to_rag(_pack)
            retrieval_query = plan.retrieval_query or question
            raw_candidates = (
                _pack.get("candidates")
                if isinstance(_pack, dict)
                and isinstance(_pack.get("candidates"), list)
                else None
            )
            papers, citations = _rag_chat_sources(
                recommender,
                retrieval_query,
                raw_candidates=raw_candidates,
            )
            log(
                "react topic rag",
                f"used_react={plan.used_react} query={plan.retrieval_query!r} "
                f"fallback={plan.fallback_reason or 'none'}",
            )
            answer = synthesizer.answer(
                None,
                question,
                evidence,
            )
            return {
                "kind": "message",
                "answer": answer,
                "recommended_papers": papers,
                "apa_citations": citations,
                "react_used": plan.used_react,
            }
        except Exception as exc:
            log("react fallback", str(exc))

    evidence = recommender.retrieve_evidence(
        question,
        mode=retrieval_mode,
    )
    papers, citations = _rag_chat_sources(recommender, question)
    return {
        "kind": "message",
        "answer": synthesizer.answer(None, question, evidence),
        "recommended_papers": papers,
        "apa_citations": citations,
    }
