"""Structured JSONL logging for CLI runs and web chat sessions."""

from __future__ import annotations

import json
import os
import re
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"
SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-\d{6}$")
TRANSCRIPT_ENTITY_LIMIT = 40
_TURN_PROGRESS_STAGES = frozenset(
    {"parse", "retrieve", "synthesize", "peer_review", "done"}
)
_ACTIVE: ContextVar["SessionLogger" | None] = ContextVar(
    "comp8420_active_session",
    default=None,
)
_EXECUTION_LOCK = threading.RLock()
_SESSION_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}


class SessionError(RuntimeError):
    """Base class for web-session lifecycle errors."""


class InvalidSessionId(SessionError):
    """Raised when a web session identifier is not a timestamp identifier."""


class SessionNotFound(SessionError):
    """Raised when a requested web session does not exist."""


class SessionCompleted(SessionError):
    """Raised when a completed web session receives another turn."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_timestamp_id() -> str:
    """Return a lexicographically sortable local timestamp identifier."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def validate_session_id(session_id: str) -> str:
    """Validate the exact timestamp identifier accepted by web session APIs."""
    if not SESSION_ID_PATTERN.fullmatch(session_id or ""):
        raise InvalidSessionId(
            "Session ID must use the format YYYYMMDD-HHMMSS-ffffff"
        )
    return session_id


def set_active(logger: "SessionLogger" | None) -> None:
    """Set the logger used by integration pipeline helpers in this context."""
    _ACTIVE.set(logger)


def get_active() -> "SessionLogger" | None:
    """Return the logger active in the current execution context."""
    return _ACTIVE.get()


def _session_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _SESSION_LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(key, threading.RLock())


def _redact(value: Any, redactions: list[str]) -> Any:
    if isinstance(value, str):
        for item in redactions:
            if item:
                value = value.replace(item, "[redacted-upload]")
        return value
    if isinstance(value, list):
        return [_redact(item, redactions) for item in value]
    if isinstance(value, tuple):
        return [_redact(item, redactions) for item in value]
    if isinstance(value, dict):
        return {str(key): _redact(item, redactions) for key, item in value.items()}
    return value


def verbose_session_log() -> bool:
    """Return whether subprocess and started-phase session rows should be logged."""
    return os.environ.get("COMP8420_VERBOSE_SESSION_LOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _pos_summary_for_log(pos: Any) -> dict[str, Any] | None:
    """Keep browser-restore POS stats without token-level paper text."""
    if not isinstance(pos, Mapping):
        return None
    noun_chunks = pos.get("noun_chunks")
    compact_chunks: list[dict[str, Any]] = []
    if isinstance(noun_chunks, list):
        for chunk in noun_chunks[:150]:
            if not isinstance(chunk, Mapping):
                continue
            compact_chunks.append(
                {
                    key: chunk[key]
                    for key in ("root", "section", "start", "end")
                    if key in chunk
                }
            )
    return {
        "token_count": pos.get("token_count", 0),
        "noun_chunks": compact_chunks,
    }


def _entity_mentions_for_log(mentions: Any) -> list[dict[str, Any]]:
    if not isinstance(mentions, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in mentions[:TRANSCRIPT_ENTITY_LIMIT]:
        if not isinstance(item, Mapping):
            continue
        compact.append(
            {
                key: item[key]
                for key in ("text", "type", "source", "section", "score")
                if key in item
            }
        )
    return compact


def _extractive_summary_for_log(extractive: Any) -> dict[str, Any] | None:
    """Keep the user-visible extractive summary without sentence-level dumps."""
    if not isinstance(extractive, Mapping):
        return None
    sentences = extractive.get("sentences")
    return {
        "text": str(extractive.get("text") or ""),
        "candidate_sentence_count": extractive.get("candidate_sentence_count"),
        "source_traceable": extractive.get("source_traceable"),
        "sentence_count": len(sentences) if isinstance(sentences, list) else 0,
    }


def _compact_paper_analysis_for_log(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Retain browser-restore NLP panels without token-level paper dumps."""
    compact = {
        key: analysis[key]
        for key in (
            "keyphrases",
            "structural_checks",
            "timings_seconds",
            "provenance",
        )
        if key in analysis
    }
    pos_summary = _pos_summary_for_log(analysis.get("pos"))
    if pos_summary is not None:
        compact["pos"] = pos_summary
    mentions = _entity_mentions_for_log(analysis.get("entity_mentions"))
    if mentions:
        compact["entity_mentions"] = mentions
    extractive = _extractive_summary_for_log(analysis.get("extractive_summary"))
    if extractive is not None:
        compact["extractive_summary"] = extractive
    return compact


def _trim_recommended_papers(
    papers: list[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for item in papers or []:
        if not isinstance(item, Mapping):
            continue
        trimmed.append(
            {
                key: item[key]
                for key in item
                if key != "snippet"
            }
        )
    return trimmed


def _analysis_result_for_log(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the UI result with paper-body-derived token dumps removed."""
    content = dict(result)
    if content.get("recommended_papers"):
        content["recommended_papers"] = _trim_recommended_papers(
            content.get("recommended_papers")
        )
    analysis = content.get("paper_analysis")
    if isinstance(analysis, Mapping):
        content["paper_analysis"] = _compact_paper_analysis_for_log(analysis)
    return content


def _relative_output_name(path: str) -> str:
    marker = f"{os.sep}outputs{os.sep}"
    if marker in path:
        return path.split(marker, 1)[1]
    return Path(path).name


def _include_in_summary(row: Mapping[str, Any]) -> bool:
    """Filter diagnostic rows that should not appear in summary.md."""
    event = str(row.get("event") or "")
    status = str(row.get("status") or "")
    if event in {"artifact_written", "pipeline_step"}:
        return False
    if event in {"subprocess_start", "subprocess_end"} and not verbose_session_log():
        return False
    if status == "started" and event not in {
        "meta",
        "user_input",
        "subprocess_start",
        "analyze_paper_started",
    }:
        return False
    if event in {"user_input", "user_output"} and row.get("component") != "integration":
        return False
    return True


def _rag_message_for_log(response: Mapping[str, Any]) -> dict[str, Any]:
    """Keep bibliography metadata while excluding retrieved paper text."""
    papers: list[dict[str, Any]] = []
    for item in response.get("recommended_papers") or []:
        if not isinstance(item, Mapping):
            continue
        papers.append(
            {
                key: item[key]
                for key in (
                    "paper_id",
                    "title",
                    "authors",
                    "year",
                    "url",
                    "score",
                    "apa_citation",
                )
                if key in item
            }
        )
    return {
        "answer": str(response.get("answer") or ""),
        "recommended_papers": papers,
        "apa_citations": [
            str(citation)
            for citation in response.get("apa_citations") or []
            if citation
        ],
    }


def read_session_events(session_path: Path) -> list[dict[str, Any]]:
    """Read valid JSON objects from a session stream."""
    events: list[dict[str, Any]] = []
    if not session_path.is_file():
        return events
    for line in session_path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            events.append(row)
    return events


def _event_timestamp(row: Mapping[str, Any]) -> str:
    return str(row.get("timestamp") or row.get("ts") or "")


def session_is_completed(events: list[dict[str, Any]]) -> bool:
    """Return whether the stream contains an explicit session completion."""
    return any(
        row.get("event") == "meta"
        and row.get("phase") == "session_complete"
        for row in events
    )


def session_transcript(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract only sanitized user-visible messages from a session stream."""
    transcript: list[dict[str, Any]] = []
    for row in events:
        event = row.get("event")
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        base = {
            "turn_id": row.get("turn_id"),
            "timestamp": _event_timestamp(row),
        }
        if event == "user":
            transcript.append(
                {
                    **base,
                    "role": "user",
                    "kind": "text",
                    "content": str(payload.get("text") or ""),
                }
            )
        elif event == "pdf_attachment":
            transcript.append(
                {
                    **base,
                    "role": "user",
                    "kind": "pdf_attachment",
                    "content": str(payload.get("filename") or "attached.pdf"),
                }
            )
        elif event == "assistant":
            kind = str(payload.get("kind") or "text")
            content = payload.get("content")
            if kind == "analysis_result" and not isinstance(content, dict):
                continue
            if kind in {"paper_recommendations", "rag_message"} and not isinstance(
                content,
                dict,
            ):
                continue
            transcript.append(
                {
                    **base,
                    "role": "assistant",
                    "kind": kind,
                    "content": content,
                }
            )
    return transcript


def refresh_session_artifacts(
    *,
    session_path: Path,
    run_id: str,
    run_dir: Path,
    started_at: str | None = None,
) -> None:
    """Rebuild the manifest and turn-grouped Markdown timeline."""
    events = read_session_events(session_path)
    components = sorted(
        {
            str(row.get("component"))
            for row in events
            if row.get("component")
        }
    )
    failures = [row for row in events if row.get("status") == "failed"]
    durations = [
        float(row["duration_ms"])
        for row in events
        if isinstance(row.get("duration_ms"), (int, float))
    ]
    turn_ids = list(
        dict.fromkeys(
            str(row["turn_id"])
            for row in events
            if row.get("turn_id")
        )
    )
    transcript = session_transcript(events)
    first_timestamp = _event_timestamp(events[0]) if events else ""
    last_timestamp = _event_timestamp(events[-1]) if events else ""
    completed_row = next(
        (
            row
            for row in reversed(events)
            if row.get("event") == "meta"
            and row.get("phase") == "session_complete"
        ),
        None,
    )
    completed_at = _event_timestamp(completed_row) if completed_row else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "session_log": str(session_path),
        "state": "completed" if completed_at else "active",
        "started_at": started_at or first_timestamp or None,
        "last_activity_at": last_timestamp or None,
        "completed_at": completed_at,
        "turn_count": len(turn_ids),
        "message_count": len(transcript),
        "event_count": len(events),
        "components": components,
        "failure_count": len(failures),
        "recorded_duration_ms": round(sum(durations), 3),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# Session {run_id}",
        "",
        f"- State: {manifest['state']}",
        f"- Turns: {len(turn_ids)}",
        f"- Messages: {len(transcript)}",
        f"- Events: {len(events)}",
        f"- Components: {', '.join(components)}",
        f"- Failures: {len(failures)}",
    ]
    session_events = [row for row in events if not row.get("turn_id")]
    groups = [(None, session_events)]
    groups.extend(
        (turn_id, [row for row in events if row.get("turn_id") == turn_id])
        for turn_id in turn_ids
    )
    for turn_id, rows in groups:
        if not rows:
            continue
        lines.extend(
            [
                "",
                "## Session Events" if turn_id is None else f"## Turn {turn_id}",
                "",
                "| Component | Phase | Status | Message | Duration |",
                "| --- | --- | --- | --- | ---: |",
            ]
        )
        for row in rows:
            if not _include_in_summary(row):
                continue
            duration = row.get("duration_ms")
            duration_text = (
                f"{float(duration):.1f} ms" if duration is not None else ""
            )
            message = str(row.get("message") or "").replace("|", "\\|")
            lines.append(
                f"| {row.get('component', '')} | {row.get('phase', '')} | "
                f"{row.get('status', '')} | {message} | {duration_text} |"
            )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def session_details(*, root: Path, session_id: str) -> dict[str, Any]:
    """Return lifecycle state and sanitized transcript for a web session."""
    validate_session_id(session_id)
    run_dir = (root / "data" / "sessions" / session_id).resolve()
    session_path = run_dir / "session.jsonl"
    if not session_path.is_file():
        raise SessionNotFound(f"Unknown session: {session_id}")
    lock = _session_lock(session_path)
    with _EXECUTION_LOCK, lock:
        events = read_session_events(session_path)
    created_at = _event_timestamp(events[0]) if events else None
    completed = session_is_completed(events)
    return {
        "session_id": session_id,
        "state": "completed" if completed else "active",
        "created_at": created_at,
        "last_activity_at": _event_timestamp(events[-1]) if events else None,
        "transcript": session_transcript(events),
    }


@dataclass
class SessionLogger:
    """Append structured events and manage a run or conversation lifecycle."""

    path: Path | None
    run_id: str = "disabled"
    run_dir: Path | None = None
    mirror_path: Path | None = None
    redactions: list[str] = field(default_factory=list)
    turn_id: str | None = None
    _fh: Any = field(default=None, repr=False)
    _mirror_fh: Any = field(default=None, repr=False)
    _sequence: int = field(default=0, repr=False)
    _started_at: str = field(default_factory=_utc_now_iso, repr=False)
    _previous_env: dict[str, str | None] = field(default_factory=dict, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _session_lock: threading.RLock | None = field(default=None, repr=False)
    _locks_held: bool = field(default=False, repr=False)
    _logged_turn_stages: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def disabled(cls) -> "SessionLogger":
        return cls(path=None)

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        log_dir: str = "data/sessions",
        run_id: str | None = None,
        redact_values: list[str] | None = None,
        turn_id: str | None = None,
    ) -> "SessionLogger":
        """Create a new run directory and expose it to child CLIs."""
        base = (root / log_dir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        if run_id is None:
            while True:
                resolved_id = new_timestamp_id()
                run_dir = base / resolved_id
                try:
                    run_dir.mkdir()
                    break
                except FileExistsError:
                    continue
        else:
            resolved_id = run_id
            run_dir = base / resolved_id
            run_dir.mkdir(parents=True, exist_ok=True)
        logger = cls._build(
            path=run_dir / "session.jsonl",
            run_id=resolved_id,
            run_dir=run_dir,
            redactions=redact_values,
            turn_id=turn_id,
        )
        logger._logged_turn_stages = set()
        logger.log_meta(
            "session_start",
            {
                "session_path": str(logger.path),
                "run_dir": str(run_dir),
            },
            status="started",
            message="Session started",
            include_turn=False,
        )
        return logger

    @classmethod
    def resume(
        cls,
        *,
        root: Path,
        session_id: str,
        redact_values: list[str] | None = None,
        turn_id: str | None = None,
        allow_completed: bool = False,
    ) -> "SessionLogger":
        """Resume a timestamped web session without emitting another start."""
        validate_session_id(session_id)
        run_dir = (root / "data" / "sessions" / session_id).resolve()
        path = run_dir / "session.jsonl"
        if not path.is_file():
            raise SessionNotFound(f"Unknown session: {session_id}")
        logger = cls._build(
            path=path,
            run_id=session_id,
            run_dir=run_dir,
            redactions=redact_values,
            turn_id=turn_id,
        )
        events = read_session_events(path)
        if session_is_completed(events) and not allow_completed:
            logger.checkpoint()
            raise SessionCompleted(f"Session is completed: {session_id}")
        logger._sequence = logger._max_sequence(events)
        if events:
            logger._started_at = _event_timestamp(events[0]) or logger._started_at
        logger._logged_turn_stages = set()
        return logger

    @classmethod
    def _build(
        cls,
        *,
        path: Path,
        run_id: str,
        run_dir: Path,
        redactions: list[str] | None,
        turn_id: str | None,
    ) -> "SessionLogger":
        logger = cls(
            path=path,
            run_id=run_id,
            run_dir=run_dir,
            redactions=list(redactions or []),
            turn_id=turn_id,
            _session_lock=_session_lock(path),
        )
        logger._acquire()
        return logger

    def _acquire(self) -> None:
        if self.path is None or self._locks_held:
            return
        _EXECUTION_LOCK.acquire()
        assert self._session_lock is not None
        self._session_lock.acquire()
        self._locks_held = True
        previous_log = os.environ.get("COMP8420_SESSION_LOG")
        self.mirror_path = (
            Path(previous_log).resolve()
            if previous_log and Path(previous_log).resolve() != self.path
            else None
        )
        self._previous_env = {
            "COMP8420_SESSION_LOG": previous_log,
            "COMP8420_RUN_ID": os.environ.get("COMP8420_RUN_ID"),
            "COMP8420_TURN_ID": os.environ.get("COMP8420_TURN_ID"),
            "COMP8420_REDACT_VALUES": os.environ.get("COMP8420_REDACT_VALUES"),
        }
        os.environ["COMP8420_SESSION_LOG"] = str(self.path)
        os.environ["COMP8420_RUN_ID"] = self.run_id
        if self.turn_id:
            os.environ["COMP8420_TURN_ID"] = self.turn_id
        else:
            os.environ.pop("COMP8420_TURN_ID", None)
        os.environ["COMP8420_REDACT_VALUES"] = json.dumps(self.redactions)

    @staticmethod
    def _max_sequence(events: list[dict[str, Any]]) -> int:
        maximum = 0
        for row in events:
            event_id = str(row.get("event_id") or "")
            match = re.search(r"-(\d{5})$", event_id)
            if match:
                maximum = max(maximum, int(match.group(1)))
        return maximum

    def _append(self, obj: dict[str, Any]) -> None:
        if self.path is None:
            return
        line = json.dumps(obj, ensure_ascii=True) + "\n"
        with self._lock:
            if self._fh is None:
                self._fh = self.path.open("a", encoding="utf-8")
            self._fh.write(line)
            self._fh.flush()
            if self.mirror_path is not None:
                self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
                if self._mirror_fh is None:
                    self._mirror_fh = self.mirror_path.open("a", encoding="utf-8")
                self._mirror_fh.write(line)
                self._mirror_fh.flush()

    def log_event(
        self,
        *,
        event: str,
        component: str,
        phase: str,
        status: str,
        source: str,
        message: str,
        duration_ms: float | None = None,
        metrics: Mapping[str, Any] | None = None,
        artifacts: list[str] | None = None,
        error: str | None = None,
        payload: Mapping[str, Any] | None = None,
        include_turn: bool = True,
    ) -> None:
        """Write one event using the shared report/video schema."""
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        timestamp = _utc_now_iso()
        self._append(
            {
                "schema_version": SCHEMA_VERSION,
                "timestamp": timestamp,
                "run_id": self.run_id,
                "turn_id": self.turn_id if include_turn else None,
                "event_id": f"{self.run_id}-{sequence:05d}",
                "event": event,
                "component": component,
                "phase": phase,
                "status": status,
                "source": source,
                "message": _redact(message, self.redactions),
                "duration_ms": (
                    round(duration_ms, 3) if duration_ms is not None else None
                ),
                "metrics": _redact(dict(metrics or {}), self.redactions),
                "artifacts": _redact(artifacts or [], self.redactions),
                "error": _redact(error, self.redactions),
                "payload": _redact(dict(payload or {}), self.redactions),
            }
        )

    def _log_transcript_item(
        self,
        *,
        event: str,
        source: str,
        message: str,
        kind: str,
        content,
    ) -> None:
        """Write one sanitized user-visible transcript item."""
        payload = (
            {"filename": content}
            if event == "pdf_attachment"
            else {"text": content}
            if event == "user"
            else {"kind": kind, "content": content}
        )
        self.log_event(
            event=event,
            component="integration",
            phase="input" if source == "user" else "output",
            status="completed",
            source=source,
            message=message,
            payload=payload,
        )

    def log_user(self, text: str) -> None:
        self._log_transcript_item(
            event="user",
            source="user",
            message="User message recorded",
            kind="text",
            content=text,
        )

    def log_pdf_attachment(self, filename: str) -> None:
        self._log_transcript_item(
            event="pdf_attachment",
            source="user",
            message="PDF attachment recorded",
            kind="pdf_attachment",
            content=Path(filename).name,
        )

    def log_assistant_text(self, text: str) -> None:
        self._log_transcript_item(
            event="assistant",
            source="live",
            message="Assistant response recorded",
            kind="text",
            content=text,
        )

    def log_assistant_recommendations(self, response: Mapping[str, Any]) -> None:
        self._log_transcript_item(
            event="assistant",
            source="live",
            message="Assistant recommendation response recorded",
            kind="paper_recommendations",
            content={
                "answer": response.get("answer", ""),
                "recommended_papers": list(
                    response.get("recommended_papers") or []
                ),
            },
        )

    def log_assistant_rag_message(self, response: Mapping[str, Any]) -> None:
        self._log_transcript_item(
            event="assistant",
            source="live",
            message="Assistant RAG response recorded",
            kind="rag_message",
            content=_rag_message_for_log(response),
        )

    def log_assistant_analysis(self, result: Mapping[str, Any]) -> None:
        self._log_transcript_item(
            event="assistant",
            source="live",
            message="Assistant analysis result recorded",
            kind="analysis_result",
            content=_analysis_result_for_log(result),
        )

    def log_user_input(
        self,
        command: str,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        self.log_event(
            event="user_input",
            component="integration",
            phase=command,
            status="started",
            source="live",
            message=f"Starting {command}",
            payload={"command": command, **dict(payload or {})},
        )

    def log_parse_complete(
        self,
        *,
        title: str,
        ref_count: int,
        section_keys: list[str],
    ) -> None:
        self.log_event(
            event="parse_complete",
            component="pdf_nlp",
            phase="parse",
            status="completed",
            source="live",
            message="PDF parsing completed",
            metrics={
                "reference_count": ref_count,
                "section_count": len(section_keys),
            },
            payload={"title": title, "section_keys": section_keys},
        )

    def log_user_output(self, payload: Mapping[str, Any]) -> None:
        artifacts = [str(item) for item in payload.get("artifact_paths", [])]
        self.log_event(
            event="user_output",
            component="integration",
            phase="output",
            status="completed",
            source="live",
            message="Run outputs written",
            artifacts=artifacts,
            metrics={
                "recommended_count": payload.get("recommended_count", 0),
                "citation_count": payload.get("citation_count", 0),
            },
            payload=payload,
        )

    def log_meta(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        status: str = "info",
        message: str | None = None,
        include_turn: bool = True,
    ) -> None:
        self.log_event(
            event="meta",
            component="integration",
            phase=kind,
            status=status,
            source="live",
            message=message or kind.replace("_", " "),
            payload={"kind": kind, **dict(payload or {})},
            include_turn=include_turn,
        )

    def log_session_config(self, config: Mapping[str, Any]) -> None:
        self.log_meta("session_config", config, message="Session configuration")

    def log_query_analysis(self, analysis: Mapping[str, Any]) -> None:
        self.log_event(
            event="query_analysis",
            component="integration",
            phase="routing",
            status="completed",
            source="live",
            message="User input analysed",
            payload=dict(analysis),
        )

    def log_route(
        self,
        *,
        input_type: str,
        route: str,
        retrieval_used: bool,
    ) -> None:
        self.log_event(
            event="route_selected",
            component="integration",
            phase="routing",
            status="completed",
            source="live",
            message=f"Selected {route} route",
            payload={
                "input_type": input_type,
                "route": route,
                "retrieval_used": retrieval_used,
            },
        )

    def log_request_failure(self, *, operation: str, error: str) -> None:
        self.log_event(
            event="request_failed",
            component="integration",
            phase=operation,
            status="failed",
            source="live",
            message=f"{operation} failed",
            error=error,
        )

    def log_outputs_recorded(
        self,
        *,
        artifact_paths: list[str] | None = None,
        recommended_count: int = 0,
        citation_count: int = 0,
        route: str | None = None,
    ) -> None:
        artifact_names = [
            _relative_output_name(str(path))
            for path in (artifact_paths or [])
        ]
        payload: dict[str, Any] = {
            "artifact_names": artifact_names,
            "recommended_count": recommended_count,
            "citation_count": citation_count,
        }
        if route:
            payload["route"] = route
        self.log_event(
            event="outputs_recorded",
            component="integration",
            phase="output",
            status="completed",
            source="live",
            message="Run outputs recorded",
            metrics={
                "recommended_count": recommended_count,
                "citation_count": citation_count,
            },
            payload=payload,
        )

    def log_turn_progress(self, stage: str, detail: str = "") -> None:
        """Record one major pipeline stage per turn without step spam."""
        if stage not in _TURN_PROGRESS_STAGES or stage in self._logged_turn_stages:
            return
        self._logged_turn_stages.add(stage)
        self.log_event(
            event="turn_progress",
            component="integration",
            phase=stage,
            status="completed",
            source="live",
            message=stage.replace("_", " "),
            payload={"stage": stage, "detail": detail},
        )

    def log_pipeline_step(self, step: str, detail: str = "") -> None:
        """Backward-compatible alias that maps noisy steps onto turn_progress."""
        stage_map = {
            "parse uploaded PDF": "parse",
            "parse uploaded PDF for peer review": "parse",
            "analyze-pdf START": "parse",
            "retrieving evidence (RAG)": "retrieve",
            "recommending related papers": "retrieve",
            "retrieving candidate corpus": "retrieve",
            "synthesizing summary/findings/gaps": "synthesize",
            "peer-review assistance": "peer_review",
            "analyze-pdf DONE": "done",
            "run START": "retrieve",
            "run DONE": "done",
        }
        stage = stage_map.get(step)
        if stage is not None:
            self.log_turn_progress(stage, detail)
        elif verbose_session_log():
            self.log_event(
                event="pipeline_step",
                component="integration",
                phase=step,
                status="completed",
                source="live",
                message=step,
                payload={"detail": detail},
            )

    def log_retrieval(
        self,
        *,
        query: str,
        retrieval_mode: str,
        top_papers: list[Mapping[str, Any]],
    ) -> None:
        self.log_event(
            event="retrieval",
            component="retrieval",
            phase="retrieval",
            status="completed",
            source="live",
            message="Related-paper retrieval completed",
            metrics={"result_count": len(top_papers)},
            payload={
                "query": query,
                "retrieval_mode": retrieval_mode,
                "top_papers": list(top_papers),
            },
        )

    def log_synthesis(
        self,
        *,
        backend: str,
        model: str,
        latency_seconds: float | None,
        error: str | None,
        evidence_ids_used: list[str] | None = None,
        thinking_enabled: bool | None = None,
        thinking_policy_reason: str | None = None,
        context_window: int | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        self.log_event(
            event="synthesis",
            component="llm",
            phase="synthesis",
            status="failed" if error else "completed",
            source="live",
            message="LLM synthesis failed" if error else "LLM synthesis completed",
            duration_ms=latency_seconds * 1000 if latency_seconds is not None else None,
            metrics={"evidence_id_count": len(evidence_ids_used or [])},
            error=error,
            payload={
                "backend": backend,
                "model": model,
                "thinking_enabled": thinking_enabled,
                "thinking_policy_reason": thinking_policy_reason,
                "context_window": context_window,
                "max_new_tokens": max_new_tokens,
            },
        )

    def log_subprocess_start(self, *, command: list[str], cwd: str) -> None:
        if not verbose_session_log():
            return
        component = self._component_from_command(command)
        self.log_event(
            event="subprocess_start",
            component=component,
            phase="subprocess",
            status="started",
            source="live",
            message=f"Starting {component} subprocess",
            payload={"command": command, "cwd": cwd},
        )

    def log_subprocess_end(
        self,
        *,
        command: list[str],
        returncode: int,
        duration_ms: float,
    ) -> None:
        if not verbose_session_log():
            return
        component = self._component_from_command(command)
        self.log_event(
            event="subprocess_end",
            component=component,
            phase="subprocess",
            status="completed" if returncode == 0 else "failed",
            source="live",
            message=f"{component} subprocess finished",
            duration_ms=duration_ms,
            metrics={"returncode": returncode},
        )

    @staticmethod
    def _component_from_command(command: list[str]) -> str:
        joined = " ".join(command)
        if "pdf_nlp" in joined or "analyze-paper" in command or "parse-pdf" in command:
            return "pdf_nlp"
        if "recommend-topic" in command:
            return "retrieval"
        if "summarize" in command or "synthesize" in command or "chat" in command:
            return "llm"
        return "integration"

    def log_artifact_written(self, path: str) -> None:
        if not verbose_session_log():
            return
        self.log_event(
            event="artifact_written",
            component="integration",
            phase="artifact",
            status="completed",
            source="live",
            message="Artifact written",
            artifacts=[path],
        )

    def _close_handles(self) -> None:
        for handle_name in ("_fh", "_mirror_fh"):
            handle = getattr(self, handle_name)
            if handle is not None:
                try:
                    handle.close()
                finally:
                    setattr(self, handle_name, None)

    def _restore_environment(self) -> None:
        for key, previous in self._previous_env.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    def _release(self) -> None:
        if not self._locks_held:
            return
        self._restore_environment()
        assert self._session_lock is not None
        self._session_lock.release()
        _EXECUTION_LOCK.release()
        self._locks_held = False

    def checkpoint(self) -> None:
        """Flush one turn without completing the surrounding conversation."""
        try:
            self._close_handles()
            if self.run_dir is not None and self.path is not None:
                refresh_session_artifacts(
                    session_path=self.path,
                    run_id=self.run_id,
                    run_dir=self.run_dir,
                    started_at=self._started_at,
                )
        finally:
            self._release()

    def complete(self) -> None:
        """Complete the session exactly once and refresh derived artifacts."""
        if self.path is not None:
            events = read_session_events(self.path)
            if not session_is_completed(events):
                self.log_meta(
                    "session_complete",
                    status="completed",
                    message="Session completed",
                    include_turn=False,
                )
        self.checkpoint()

    def close(self) -> None:
        """Backward-compatible completion for one-request CLI and job runs."""
        self.complete()

    @property
    def session_path(self) -> str | None:
        return str(self.path) if self.path else None


def create_web_session(*, root: Path) -> dict[str, Any]:
    """Create an active web conversation and return its public metadata."""
    logger = SessionLogger.create(root=root)
    session_id = logger.run_id
    logger.checkpoint()
    details = session_details(root=root, session_id=session_id)
    return {
        "session_id": session_id,
        "state": details["state"],
        "created_at": details["created_at"],
    }


def complete_web_session(*, root: Path, session_id: str) -> dict[str, Any]:
    """Complete a web conversation idempotently."""
    details = session_details(root=root, session_id=session_id)
    if details["state"] == "completed":
        return details
    logger = SessionLogger.resume(
        root=root,
        session_id=session_id,
        allow_completed=True,
    )
    logger.complete()
    return session_details(root=root, session_id=session_id)
