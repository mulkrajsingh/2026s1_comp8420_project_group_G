"""FastAPI HTTP layer for the local research assistant.

Exposes thin endpoints that delegate to the same service functions used by the
CLI. The app is intended for local use only (no auth, no deployment). Start it
with ``python rpa.py web`` and open ``http://127.0.0.1:8000``.

Endpoints cover topic search, PDF analysis, recommendations, peer review, chat,
session management, and background jobs.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .contracts import ParsedPaper
from .ollama_runtime import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_WEB_KEEP_ALIVE,
    MODEL_MANAGER,
    PROJECT_MODEL_IDS,
)
from .request_control import (
    RequestCancelledError,
    bind_request,
    cancel as cancel_active_request,
    new_request_id,
    register,
    unregister,
)
from .session_log import (
    InvalidSessionId,
    SessionCompleted,
    SessionNotFound,
    complete_web_session,
    create_web_session,
    session_details,
)


def _warm_query_analyzer() -> dict[str, str]:
    """Load both pinned query-understanding models before serving requests."""
    from .pipeline import analyze_query

    analysis = analyze_query(
        "How does an LLM get relevant data?",
        style_override="auto",
    )
    return {
        "status": "loaded",
        "embedding_model": analysis.embedding_model,
        "fallback_model": analysis.fallback_model or "",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm local inference models before the web app accepts requests."""
    host = os.getenv("COMP8420_OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    model = os.getenv("COMP8420_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    keep_alive = os.getenv(
        "COMP8420_OLLAMA_KEEP_ALIVE",
        DEFAULT_WEB_KEEP_ALIVE,
    )
    os.environ.setdefault("COMP8420_OLLAMA_KEEP_ALIVE", keep_alive)
    MODEL_MANAGER.configure(host=host, keep_alive=keep_alive)
    app.state.ollama_manager = MODEL_MANAGER
    app.state.ollama = await asyncio.to_thread(MODEL_MANAGER.start, model)
    app.state.query_analyzer = await asyncio.to_thread(_warm_query_analyzer)
    try:
        yield
    finally:
        try:
            app.state.ollama = await asyncio.to_thread(MODEL_MANAGER.shutdown)
        except RuntimeError as exc:
            app.state.ollama = {
                "status": "unload_failed",
                "model": MODEL_MANAGER.state().get("model", model),
                "host": host,
                "error": str(exc),
            }
        app.state.ollama_manager = None


api = FastAPI(
    title="Research Paper Analysis API (local)",
    version="0.1.0",
    lifespan=lifespan,
)
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL_EMBEDDING_MODEL = "allenai/specter2_base"
DEFAULT_RETRIEVAL_STRATEGY = "hybrid_rrf"

if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        api.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @api.get("/", include_in_schema=False)
    def frontend_index():
        return FileResponse(FRONTEND_DIST / "index.html")


class TopicReq(BaseModel):
    query: str
    style: str = "auto"
    retrieval_mode: str = "offline"
    model_mode: str = "base"
    retrieval_embedding_model: str = Field(
        default=DEFAULT_RETRIEVAL_EMBEDDING_MODEL,
        min_length=1,
        max_length=200,
    )
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_strategy: str = Field(default=DEFAULT_RETRIEVAL_STRATEGY)
    llm_model: str = "qwen3:8b"
    prompt_strategy: str = Field(default="zero_shot", pattern="^(zero_shot|few_shot)$")


class ChatReq(BaseModel):
    question: str
    llm_model: str = "qwen3:8b"
    style: str = "auto"
    retrieval_mode: str = "offline"
    retrieval_embedding_model: str = Field(
        default=DEFAULT_RETRIEVAL_EMBEDDING_MODEL,
        min_length=1,
        max_length=200,
    )
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_strategy: str = Field(default=DEFAULT_RETRIEVAL_STRATEGY)
    session_id: str | None = None
    prompt_strategy: str = Field(default="zero_shot", pattern="^(zero_shot|few_shot)$")


class RecommendReq(BaseModel):
    parsed: dict
    retrieval_mode: str = "offline"
    retrieval_embedding_model: str = Field(
        default=DEFAULT_RETRIEVAL_EMBEDDING_MODEL,
        min_length=1,
        max_length=200,
    )
    retrieval_top_k: int = Field(default=10, ge=1, le=20)
    retrieval_strategy: str = Field(default=DEFAULT_RETRIEVAL_STRATEGY)


class JobTopicReq(TopicReq):
    """Request body for queued topic analysis."""


@contextmanager
def _model_request(model: str):
    """Switch models once per request while serializing local generation."""
    if model not in PROJECT_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported Ollama model {model!r}. "
                "Use qwen3:8b or qwen3-research-lora:latest."
            ),
        )
    manager = getattr(api.state, "ollama_manager", None)
    if manager is None:
        yield
        return
    with manager.use(backend="ollama", model=model) as state:
        api.state.ollama = state
        try:
            yield
        finally:
            api.state.ollama = manager.state()


def _session_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InvalidSessionId):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, SessionNotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, SessionCompleted):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def request_scope(
    x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
):
    """Bind one cancellable request id for the lifetime of a web action."""
    request_id = x_request_id or new_request_id()
    register(request_id)
    bind_request(request_id)
    try:
        yield request_id
    finally:
        unregister(request_id)


@api.post("/api/requests/{request_id}/cancel")
def cancel_request(request_id: str):
    """Cancel one active chat or PDF analysis request."""
    if not cancel_active_request(request_id):
        raise HTTPException(status_code=404, detail="No active request")
    return {"cancelled": True}


@api.post("/api/sessions")
def create_session():
    """Create one timestamped web conversation."""
    return create_web_session(root=INTEGRATION_ROOT)


@api.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    """Return lifecycle state and the sanitized user-visible transcript."""
    try:
        return session_details(root=INTEGRATION_ROOT, session_id=session_id)
    except (InvalidSessionId, SessionNotFound) as exc:
        raise _session_http_error(exc) from exc


@api.post("/api/sessions/{session_id}/complete")
def complete_session(session_id: str):
    """Complete one web conversation; repeated calls are idempotent."""
    try:
        return complete_web_session(
            root=INTEGRATION_ROOT,
            session_id=session_id,
        )
    except (InvalidSessionId, SessionNotFound) as exc:
        raise _session_http_error(exc) from exc


@api.get("/api/health")
def health():
    from .service import production_provider_sources

    return {
        "status": "ok",
        "provider_sources": production_provider_sources(),
        "ollama": getattr(api.state, "ollama", {"status": "not_started"}),
        "query_analyzer": getattr(
            api.state,
            "query_analyzer",
            {"status": "not_started"},
        ),
    }


@api.get("/api/models")
def models():
    """Return the supported local base and adapter model tags."""
    manager = getattr(api.state, "ollama_manager", None) or MODEL_MANAGER
    try:
        return manager.catalog()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@api.post("/api/search-topic")
def search_topic(req: TopicReq):
    from . import service as service

    try:
        with _model_request(req.llm_model):
            out = service.run_topic(
                query=req.query,
                style=req.style,
                retrieval_mode=req.retrieval_mode,
                model_mode=req.model_mode,
                retrieval_embedding_model=req.retrieval_embedding_model,
                retrieval_top_k=req.retrieval_top_k,
                retrieval_strategy=req.retrieval_strategy,
                llm_model=req.llm_model,
                prompt_strategy=req.prompt_strategy,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return out["result"].to_dict()


def _optional_session_kwargs(session_id: str | None) -> dict:
    return {"session_id": session_id} if session_id is not None else {}


@api.post("/api/analyze-pdf")
async def analyze_pdf(
    file: UploadFile = File(...),
    style: str = "auto",
    retrieval_mode: str = "offline",
    retrieval_embedding_model: str = DEFAULT_RETRIEVAL_EMBEDDING_MODEL,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    retrieval_top_k: int = Query(5, ge=1, le=20),
    no_related_papers: bool = False,
    include_peer_review: bool = False,
    llm_model: str = "qwen3:8b",
    session_id: str | None = None,
    prompt_strategy: str = Query("zero_shot", pattern="^(zero_shot|few_shot)$"),
    _request_id: str = Depends(request_scope),
):
    from . import service as service

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())

    # Run the blocking pipeline in a thread so the asyncio event loop stays
    # free to handle concurrent requests (e.g. the cancel endpoint).
    # bind_request is called inside the thread because asyncio.to_thread copies
    # the context at call-time; the ContextVar may not yet be propagated there.
    def _run():
        bind_request(_request_id)
        with _model_request(llm_model):
            return service.run_analyze_pdf(
                tmp,
                style=style,
                retrieval_mode="none" if no_related_papers else retrieval_mode,
                with_related_papers=not no_related_papers,
                with_peer_review=include_peer_review,
                retrieval_embedding_model=retrieval_embedding_model,
                retrieval_strategy=retrieval_strategy,
                retrieval_top_k=retrieval_top_k,
                llm_model=llm_model,
                session_id=session_id,
                upload_name=Path(file.filename or "uploaded.pdf").name,
                prompt_strategy=prompt_strategy,
            )

    try:
        out = await asyncio.to_thread(_run)
        r = out["result"]
    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except (InvalidSessionId, SessionNotFound, SessionCompleted) as exc:
        raise _session_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return r.to_dict()


@api.post("/api/jobs/analyze-pdf")
async def queue_analyze_pdf(
    file: UploadFile = File(...),
    style: str = "auto",
    retrieval_mode: str = "offline",
    retrieval_embedding_model: str = DEFAULT_RETRIEVAL_EMBEDDING_MODEL,
    retrieval_strategy: str = DEFAULT_RETRIEVAL_STRATEGY,
    retrieval_top_k: int = Query(5, ge=1, le=20),
    no_related_papers: bool = False,
    llm_model: str = "qwen3:8b",
):
    """Queue local PDF analysis and return immediately with a job identifier."""
    from .jobs import submit_pdf_job

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(await file.read())
        record = submit_pdf_job(
            tmp,
            {
                "style": style,
                "retrieval_mode": "none" if no_related_papers else retrieval_mode,
                "with_related_papers": not no_related_papers,
                "retrieval_embedding_model": retrieval_embedding_model,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_top_k": retrieval_top_k,
                "llm_model": llm_model,
            },
        )
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return {"job_id": record.job_id, "state": record.state}


@api.post("/api/jobs/search-topic")
def queue_search_topic(req: JobTopicReq):
    """Queue local topic analysis on the single-worker execution lane."""
    from .jobs import submit_topic_job

    record = submit_topic_job(
        req.query,
        {
            "style": req.style,
            "retrieval_mode": req.retrieval_mode,
            "model_mode": req.model_mode,
            "retrieval_embedding_model": req.retrieval_embedding_model,
            "retrieval_top_k": req.retrieval_top_k,
            "llm_model": req.llm_model,
        },
    )
    return {"job_id": record.job_id, "state": record.state}


@api.get("/api/jobs/{job_id}")
def job_status(job_id: str, after: int = 0):
    """Poll job state and return only events after the supplied cursor."""
    from .jobs import get_job

    try:
        return get_job(job_id, after=after)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}") from exc


@api.post("/api/recommend")
def recommend(req: RecommendReq):
    from . import service as service

    parsed = ParsedPaper.from_dict(req.parsed)
    rec = service.recommend_for_parsed(
        parsed,
        retrieval_mode=req.retrieval_mode,
        retrieval_embedding_model=req.retrieval_embedding_model,
        retrieval_top_k=req.retrieval_top_k,
        retrieval_strategy=req.retrieval_strategy,
    )
    return rec.to_dict()


@api.post("/api/peer-review")
async def peer_review(file: UploadFile = File(...)):
    from . import service as service

    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(await file.read())
    try:
        with _model_request(DEFAULT_OLLAMA_MODEL):
            out = service.run_peer_review_pdf(tmp)
            feedback = out["feedback"]
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {"peer_review": feedback}


class PdfChatReq(BaseModel):
    question: str
    paper_json_path: str | None = None
    llm_model: str = "qwen3:8b"
    style: str = "auto"
    session_id: str | None = None
    prompt_strategy: str = Field(default="zero_shot", pattern="^(zero_shot|few_shot)$")


@api.post("/api/chat")
def chat(req: ChatReq, _request_id: str = Depends(request_scope)):
    from . import service as service

    try:
        with _model_request(req.llm_model):
            answer = service.chat_topic(
                req.question,
                llm_model=req.llm_model,
                style=req.style,
                retrieval_mode=req.retrieval_mode,
                retrieval_embedding_model=req.retrieval_embedding_model,
                retrieval_top_k=req.retrieval_top_k,
                retrieval_strategy=req.retrieval_strategy,
                **_optional_session_kwargs(req.session_id),
                prompt_strategy=req.prompt_strategy,
            )
    except RequestCancelledError as exc:
        raise HTTPException(status_code=499, detail=str(exc)) from exc
    except (InvalidSessionId, SessionNotFound, SessionCompleted) as exc:
        raise _session_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return answer


@api.post("/api/chat-pdf")
def chat_pdf(req: PdfChatReq):
    from . import service as service

    if not req.paper_json_path:
        raise ValueError("paper_json_path is required for PDF-grounded chat")
    session_kwargs = _optional_session_kwargs(req.session_id)
    try:
        with _model_request(req.llm_model):
            answer = service.run_pdf_chat(
                req.question,
                req.paper_json_path,
                llm_model=req.llm_model,
                style=req.style,
                prompt_strategy=req.prompt_strategy,
                **session_kwargs,
            )
    except (InvalidSessionId, SessionNotFound, SessionCompleted) as exc:
        raise _session_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"answer": answer}


def write_api_contract() -> str:
    """Write ``outputs/api_contract.md`` summarising the local HTTP surface."""
    from .io_paths import write_text

    rows = [
        ("GET", "/api/health", "—", "{status, provider_sources, ollama}"),
        (
            "GET",
            "/api/models",
            "—",
            "{default_model, active_model, models[]}",
        ),
        (
            "POST",
            "/api/sessions",
            "—",
            "{session_id, state, created_at}",
        ),
        (
            "GET",
            "/api/sessions/{session_id}",
            "—",
            "{session_id, state, created_at, last_activity_at, transcript}",
        ),
        (
            "POST",
            "/api/sessions/{session_id}/complete",
            "—",
            "Completed session metadata and transcript",
        ),
        (
            "POST",
            "/api/search-topic",
            "{query, style?, retrieval_mode?, retrieval_embedding_model?, "
            "retrieval_embedding_model?, retrieval_top_k?, llm_model?}",
            "AnalysisResult",
        ),
        (
            "POST",
            "/api/analyze-pdf",
            "multipart file + style?, retrieval_mode?, retrieval_embedding_model?, "
            "retrieval_embedding_model?, retrieval_top_k?, no_related_papers?, "
            "llm_model?, session_id?",
            "AnalysisResult",
        ),
        (
            "POST",
            "/api/jobs/analyze-pdf",
            "multipart PDF + analysis options",
            "{job_id, state}",
        ),
        (
            "POST",
            "/api/jobs/search-topic",
            "TopicReq",
            "{job_id, state}",
        ),
        (
            "GET",
            "/api/jobs/{job_id}?after=<cursor>",
            "—",
            "{state, events, cursor, result?, error?}",
        ),
        (
            "POST",
            "/api/recommend",
            "{parsed: ParsedPaper, retrieval_mode?, retrieval_embedding_model?}",
            "Recommendation",
        ),
        ("POST", "/api/peer-review", "multipart file (PDF)", "{peer_review}"),
        (
            "POST",
            "/api/chat",
            "{question, llm_model?, style?, retrieval_mode?, session_id?}",
            "{answer}",
        ),
        (
            "POST",
            "/api/chat-pdf",
            "{question, paper_json_path, llm_model?, style?, session_id?}",
            "{answer}",
        ),
    ]
    md = [
        "# Local API contract",
        "",
        "Run: `python rpa.py web` then open `/docs`.",
        "",
        "| Method | Path | Request | Response |",
        "| --- | --- | --- | --- |",
    ]
    for m, p, req, resp in rows:
        md.append(f"| {m} | `{p}` | {req} | {resp} |")
    md += [
        "",
        "## Recommend request",
        "",
        "```json",
        "{",
        '  "parsed": {',
        '    "metadata": {},',
        '    "sections": {},',
        '    "references": [],',
        '    "keywords": [],',
        '    "entities": {}',
        "  },",
        '  "retrieval_mode": "offline",',
        '  "retrieval_embedding_model": "allenai/specter2_base"',
        "}",
        "```",
        "",
        "All endpoints reuse production CLI orchestration, are local-only, and "
        "uploaded PDFs are deleted after processing.",
        "",
    ]
    return write_text("api_contract.md", "\n".join(md))
