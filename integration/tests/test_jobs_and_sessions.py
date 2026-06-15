"""Tests for enriched contracts, structured sessions, and queued job state."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.contracts import ParsedPaper
from app.jobs import JobRecord, _run_job, get_job
from app.service import _temporary_upload_redactions
from app.session_log import (
    InvalidSessionId,
    SessionCompleted,
    SessionLogger,
    complete_web_session,
    create_web_session,
    new_timestamp_id,
    session_details,
)
import app.jobs as jobs
import app.service as service


def _direct_query_analysis() -> Mock:
    analysis = Mock(
        intent="chit_chat",
        should_use_retrieval=False,
        is_paper_recommendation=False,
        style="concise",
    )
    analysis.as_dict.return_value = {
        "intent": "chit_chat",
        "emotion": "positive",
        "topic_expertise": "intermediate",
        "verbosity": "concise",
        "style": "concise",
        "confidence": 0.9,
        "style_source": "explicit_override",
        "fallback_used": False,
    }
    return analysis


def _retrieval_query_analysis() -> Mock:
    analysis = Mock(
        intent="research_question",
        should_use_retrieval=True,
        is_paper_recommendation=False,
        style="concise",
    )
    analysis.as_dict.return_value = {
        "intent": "research_question",
        "emotion": "neutral",
        "topic_expertise": "intermediate",
        "verbosity": "concise",
        "style": "concise",
        "confidence": 0.9,
        "style_source": "cosine_similarity",
        "fallback_used": False,
    }
    return analysis


def _parsed_paper(paper_id: str, title: str, abstract: str) -> ParsedPaper:
    return ParsedPaper(
        metadata={
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": [],
            "source": "uploaded_pdf",
        },
        sections={
            "abstract": abstract,
            "introduction": "",
            "method": "",
            "results": "",
            "conclusion": "",
        },
    )


class ContractAndSessionTests(unittest.TestCase):
    def test_temporary_upload_redactions_include_path_aliases(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
            redactions = _temporary_upload_redactions(handle.name)
            self.assertIn(handle.name, redactions)
            self.assertIn(str(Path(handle.name).resolve()), redactions)

    def test_parsed_paper_round_trip_preserves_analysis_and_optional_sections(self) -> None:
        paper = _parsed_paper("p", "Title", "Abstract")
        paper.sections["limitations"] = "A limitation."
        paper.analysis = {"extractive_summary": {"text": "Abstract"}}
        restored = ParsedPaper.from_dict(paper.to_dict())
        self.assertEqual(restored.sections["limitations"], "A limitation.")
        self.assertEqual(
            restored.analysis["extractive_summary"]["text"],
            "Abstract",
        )

    def test_session_artifacts_use_structured_schema_and_redact_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            secret = "/private/tmp/upload-123.pdf"
            logger = SessionLogger.create(
                root=Path(temp_dir),
                run_id="run-test",
                redact_values=[secret],
            )
            logger.log_user_input("analyze-pdf", {"pdf_path": secret})
            logger.log_event(
                event="pos_complete",
                component="pdf_nlp",
                phase="pos",
                status="completed",
                source="live",
                message="POS complete",
                duration_ms=12.3,
            )
            logger.close()
            session_path = Path(temp_dir) / "data" / "sessions" / "run-test" / "session.jsonl"
            text = session_path.read_text(encoding="utf-8")
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertNotIn(secret, text)
            self.assertIn("[redacted-upload]", text)
            self.assertTrue(all(row["schema_version"] == "1.0" for row in rows))
            self.assertTrue((session_path.parent / "manifest.json").is_file())
            self.assertTrue((session_path.parent / "summary.md").is_file())

    def test_job_runner_records_success_and_deletes_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_root = jobs._integration_root
            jobs._integration_root = lambda: Path(temp_dir)
            try:
                upload = Path(temp_dir) / "upload.pdf"
                upload.write_bytes(b"%PDF")
                record = JobRecord(job_id="job-test", kind="analyze-pdf")
                jobs._JOBS[record.job_id] = record

                class Result:
                    def to_dict(self):
                        return {"summary": "done"}

                _run_job(
                    record,
                    lambda **_: {"result": Result()},
                    {},
                    cleanup_path=str(upload),
                )
                status = get_job(record.job_id)
                self.assertEqual(status["state"], "succeeded")
                self.assertEqual(status["result"]["summary"], "done")
                self.assertFalse(upload.exists())
                self.assertTrue(status["events"])
                manifest = json.loads(
                    (record.run_dir / "manifest.json").read_text(encoding="utf-8")
                )
                summary = (record.run_dir / "summary.md").read_text(encoding="utf-8")
                self.assertEqual(manifest["event_count"], len(status["events"]))
                self.assertIn("analyze-pdf job succeeded", summary)
            finally:
                jobs._integration_root = original_root
                jobs._JOBS.pop("job-test", None)
                for key in (
                    "COMP8420_SESSION_LOG",
                    "COMP8420_RUN_ID",
                    "COMP8420_TURN_ID",
                    "COMP8420_REDACT_VALUES",
                ):
                    os.environ.pop(key, None)

    def test_timestamped_web_sessions_sort_chronologically(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = create_web_session(root=root)
            second = create_web_session(root=root)

            pattern = r"^\d{8}-\d{6}-\d{6}$"
            self.assertRegex(first["session_id"], pattern)
            self.assertRegex(second["session_id"], pattern)
            self.assertLess(first["session_id"], second["session_id"])

    def test_two_chat_turns_share_one_session_and_restore_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            answers = iter(
                (
                    {
                        "kind": "message",
                        "answer": "first answer",
                        "recommended_papers": [],
                    },
                    {
                        "kind": "message",
                        "answer": "second answer",
                        "recommended_papers": [],
                    },
                )
            )
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_direct_query_analysis(),
                ),
                patch("app.service._synthesizer_provider", return_value=Mock()),
                patch(
                    "app.service.pipeline.chat_response",
                    side_effect=lambda *_args, **_kwargs: next(answers),
                ),
            ):
                self.assertEqual(
                    service.chat_topic("first question", session_id=session_id)["answer"],
                    "first answer",
                )
                self.assertEqual(
                    service.chat_topic("second question", session_id=session_id)["answer"],
                    "second answer",
                )

            session_path = (
                root / "data" / "sessions" / session_id / "session.jsonl"
            )
            rows = [
                json.loads(line)
                for line in session_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(row.get("phase") == "session_start" for row in rows),
                1,
            )
            self.assertEqual(
                sum(row.get("phase") == "session_complete" for row in rows),
                0,
            )
            turn_ids = {
                row["turn_id"]
                for row in rows
                if row.get("turn_id")
            }
            self.assertEqual(len(turn_ids), 2)
            details = session_details(root=root, session_id=session_id)
            self.assertEqual(details["state"], "active")
            self.assertEqual(
                [
                    (item["role"], item["kind"], item["content"])
                    for item in details["transcript"]
                ],
                [
                    ("user", "text", "first question"),
                    ("assistant", "text", "first answer"),
                    ("user", "text", "second question"),
                    ("assistant", "text", "second answer"),
                ],
            )
            manifest = json.loads(
                (session_path.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["turn_count"], 2)
            self.assertEqual(manifest["message_count"], 4)

    def test_rag_chat_restores_bibliography_without_retrieved_snippets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            response = {
                "kind": "message",
                "answer": "Retrieval grounding reduces unsupported claims [p1].",
                "recommended_papers": [
                    {
                        "paper_id": "p1",
                        "title": "Grounded Generation",
                        "authors": ["Alice"],
                        "year": "2024",
                        "url": "https://arxiv.org/abs/p1",
                        "score": 0.8,
                        "apa_citation": "Alice (2024). Grounded Generation.",
                        "snippet": "retrieved abstract text must not be logged",
                        "why": "ranked by the retrieval model",
                    }
                ],
                "apa_citations": ["Alice (2024). Grounded Generation."],
            }
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_retrieval_query_analysis(),
                ),
                patch(
                    "app.service._resolve_corpus",
                    return_value=root / "corpus.jsonl",
                ),
                patch(
                    "app.service._topic_providers",
                    return_value=(Mock(), root / "corpus.jsonl"),
                ),
                patch(
                    "app.service.pipeline.chat_response",
                    return_value=response,
                ),
            ):
                result = service.chat_topic(
                    "How does retrieval grounding work?",
                    session_id=session_id,
                )

            self.assertEqual(result, response)
            details = session_details(root=root, session_id=session_id)
            self.assertEqual(
                [item["kind"] for item in details["transcript"]],
                ["text", "rag_message"],
            )
            restored = details["transcript"][1]["content"]
            self.assertEqual(restored["answer"], response["answer"])
            self.assertEqual(
                restored["apa_citations"],
                response["apa_citations"],
            )
            self.assertNotIn("snippet", restored["recommended_papers"][0])
            self.assertNotIn("why", restored["recommended_papers"][0])

            session_path = (
                root / "data" / "sessions" / session_id / "session.jsonl"
            )
            text = session_path.read_text(encoding="utf-8")
            self.assertNotIn("retrieved abstract text must not be logged", text)
            rows = [json.loads(line) for line in text.splitlines()]
            output = next(row for row in rows if row["event"] == "outputs_recorded")
            self.assertEqual(output["payload"]["recommended_count"], 1)
            self.assertEqual(output["payload"]["citation_count"], 1)

    def test_pdf_transcript_keeps_filename_and_redacts_temporary_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            secret = "/private/tmp/upload-secret.pdf"
            logger = SessionLogger.resume(
                root=root,
                session_id=session_id,
                redact_values=[secret],
                turn_id=new_timestamp_id(),
            )
            logger.log_pdf_attachment("original-paper.pdf")
            logger.log_user_input("analyze-pdf", {"pdf_path": secret})
            logger.log_assistant_analysis(
                {"summary": "Paper summary", "paper_analysis": {}}
            )
            logger.checkpoint()

            details = session_details(root=root, session_id=session_id)
            self.assertEqual(
                [item["kind"] for item in details["transcript"]],
                ["pdf_attachment", "analysis_result"],
            )
            self.assertEqual(
                details["transcript"][0]["content"],
                "original-paper.pdf",
            )
            text = (
                root
                / "data"
                / "sessions"
                / session_id
                / "session.jsonl"
            ).read_text(encoding="utf-8")
            self.assertNotIn(secret, text)
            self.assertIn("[redacted-upload]", text)

    def test_analysis_event_omits_reconstructable_paper_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret_body = "private extracted sentence from the uploaded paper"
            logger = SessionLogger.create(root=root, run_id="analysis-log-test")
            logger.log_assistant_analysis(
                {
                    "summary": "User-visible generated summary",
                    "paper_analysis": {
                        "pos": {
                            "token_count": 42,
                            "noun_chunks": [{"text": secret_body, "root": "chunk"}],
                            "tokens": [{"text": secret_body, "tag": "NN"}],
                        },
                        "entity_mentions": [
                            {"text": "retrieval augmentation", "type": "Method"}
                        ],
                        "extractive_summary": {
                            "text": "User-visible extractive summary",
                            "sentences": [{"text": secret_body}],
                            "candidate_sentence_count": 10,
                            "source_traceable": True,
                        },
                        "keyphrases": [{"text": "retrieval augmentation"}],
                        "structural_checks": [{"code": "no_limitations"}],
                        "timings_seconds": {"pos": 1.0},
                        "provenance": {"pos": "local model"},
                    },
                }
            )
            logger.close()

            session_path = (
                root
                / "data"
                / "sessions"
                / "analysis-log-test"
                / "session.jsonl"
            )
            text = session_path.read_text(encoding="utf-8")
            self.assertNotIn(secret_body, text)
            rows = [json.loads(line) for line in text.splitlines()]
            assistant = next(row for row in rows if row["event"] == "assistant")
            analysis = assistant["payload"]["content"]["paper_analysis"]
            self.assertEqual(analysis["pos"]["token_count"], 42)
            self.assertEqual(len(analysis["pos"]["noun_chunks"]), 1)
            self.assertNotIn("text", analysis["pos"]["noun_chunks"][0])
            self.assertNotIn("tokens", analysis["pos"])
            self.assertEqual(
                analysis["entity_mentions"],
                [{"text": "retrieval augmentation", "type": "Method"}],
            )
            self.assertEqual(
                analysis["extractive_summary"]["text"],
                "User-visible extractive summary",
            )
            self.assertEqual(analysis["extractive_summary"]["sentence_count"], 1)
            self.assertNotIn("sentences", analysis["extractive_summary"])
            self.assertEqual(
                analysis["keyphrases"],
                [{"text": "retrieval augmentation"}],
            )

    def test_summary_omits_module_cli_bookkeeping_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = SessionLogger.create(root=Path(temp_dir), run_id="summary-filter")
            logger.log_turn_progress("parse", "source=live")
            logger.log_event(
                event="user_input",
                component="llm",
                phase="synthesis",
                status="started",
                source="live",
                message="user input",
                payload={"command": "summarize"},
            )
            logger.log_event(
                event="pos_completed",
                component="pdf_nlp",
                phase="pos",
                status="completed",
                source="live",
                message="pos completed",
                duration_ms=12.0,
            )
            logger.close()
            summary = (
                Path(temp_dir)
                / "data"
                / "sessions"
                / "summary-filter"
                / "summary.md"
            ).read_text(encoding="utf-8")
            self.assertIn("| integration | parse |", summary)
            self.assertIn("pos completed", summary)
            self.assertNotIn("user input", summary)

    def test_completion_is_idempotent_and_rejects_later_turns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            complete_web_session(root=root, session_id=session_id)
            complete_web_session(root=root, session_id=session_id)

            session_path = (
                root / "data" / "sessions" / session_id / "session.jsonl"
            )
            rows = [
                json.loads(line)
                for line in session_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sum(row.get("phase") == "session_complete" for row in rows),
                1,
            )
            with self.assertRaises(SessionCompleted):
                SessionLogger.resume(
                    root=root,
                    session_id=session_id,
                    turn_id=new_timestamp_id(),
                )

    def test_failed_turn_keeps_web_session_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_direct_query_analysis(),
                ),
                patch("app.service._synthesizer_provider", return_value=Mock()),
                patch(
                    "app.service.pipeline.chat_response",
                    side_effect=(
                        RuntimeError("temporary failure"),
                        {
                            "kind": "message",
                            "answer": "recovered",
                            "recommended_papers": [],
                        },
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "temporary failure"):
                    service.chat_topic("first", session_id=session_id)
                self.assertEqual(
                    service.chat_topic("second", session_id=session_id)["answer"],
                    "recovered",
                )

            details = session_details(root=root, session_id=session_id)
            self.assertEqual(details["state"], "active")
            self.assertEqual(
                [item["content"] for item in details["transcript"]],
                ["first", "second", "recovered"],
            )
            manifest = json.loads(
                (
                    root
                    / "data"
                    / "sessions"
                    / session_id
                    / "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["failure_count"], 1)

    def test_session_identifier_validation_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(InvalidSessionId):
                session_details(
                    root=Path(temp_dir),
                    session_id="../outside",
                )

    def test_session_api_create_get_and_complete(self) -> None:
        from app.api import complete_session, create_session, get_session

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch("app.api.INTEGRATION_ROOT", root):
                created = create_session()
                fetched = get_session(created["session_id"])
                completed = complete_session(created["session_id"])

            self.assertEqual(fetched["state"], "active")
            self.assertEqual(created["created_at"], fetched["created_at"])
            self.assertEqual(completed["state"], "completed")
            self.assertEqual(fetched["transcript"], [])

    def test_concurrent_turns_do_not_duplicate_integration_event_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session_id = create_web_session(root=root)["session_id"]
            errors: list[Exception] = []

            def append_turn(index: int) -> None:
                try:
                    logger = SessionLogger.resume(
                        root=root,
                        session_id=session_id,
                        turn_id=f"turn-{index}",
                    )
                    logger.log_user(f"question {index}")
                    logger.log_assistant_text(f"answer {index}")
                    logger.checkpoint()
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [
                threading.Thread(target=append_turn, args=(index,))
                for index in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            rows = [
                json.loads(line)
                for line in (
                    root
                    / "data"
                    / "sessions"
                    / session_id
                    / "session.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            event_ids = [row["event_id"] for row in rows]
            self.assertEqual(len(event_ids), len(set(event_ids)))
            self.assertEqual(
                len(session_details(root=root, session_id=session_id)["transcript"]),
                8,
            )

    def test_direct_text_chat_logs_input_analysis_route_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_direct_query_analysis(),
                ),
                patch("app.service._synthesizer_provider", return_value=Mock()),
                patch(
                    "app.service.pipeline.chat_response",
                    return_value={
                        "kind": "message",
                        "answer": "generated answer",
                        "recommended_papers": [],
                    },
                ),
            ):
                answer = service.chat_topic("how areyou?", style="concise")

            self.assertEqual(answer["answer"], "generated answer")
            sessions = list((root / "data" / "sessions").glob("*/session.jsonl"))
            self.assertEqual(len(sessions), 1)
            rows = [
                json.loads(line)
                for line in sessions[0].read_text(encoding="utf-8").splitlines()
            ]
            events = [row["event"] for row in rows]
            self.assertIn("user_input", events)
            self.assertIn("user", events)
            self.assertIn("query_analysis", events)
            self.assertIn("route_selected", events)
            self.assertIn("assistant", events)
            route = next(row for row in rows if row["event"] == "route_selected")
            self.assertEqual(route["payload"]["input_type"], "text")
            self.assertEqual(route["payload"]["route"], "direct_llm_chat")
            self.assertFalse(route["payload"]["retrieval_used"])
            self.assertTrue((sessions[0].parent / "manifest.json").is_file())
            self.assertTrue((sessions[0].parent / "summary.md").is_file())

    def test_text_chat_failure_is_logged_before_session_closes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_direct_query_analysis(),
                ),
                patch("app.service._synthesizer_provider", return_value=Mock()),
                patch(
                    "app.service.pipeline.chat_response",
                    side_effect=RuntimeError("local model unavailable"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "local model unavailable"):
                    service.chat_topic("hi", style="concise")

            session_path = next(
                (root / "data" / "sessions").glob("*/session.jsonl")
            )
            rows = [
                json.loads(line)
                for line in session_path.read_text(encoding="utf-8").splitlines()
            ]
            failure = next(row for row in rows if row["event"] == "request_failed")
            self.assertEqual(failure["status"], "failed")
            self.assertEqual(failure["error"], "local model unavailable")
            manifest = json.loads(
                (session_path.parent / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["failure_count"], 1)

    def test_pdf_grounded_chat_logs_route_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paper_json = root / "parsed_paper.json"
            paper_json.write_text(
                json.dumps(
                    _parsed_paper(
                        "paper-1",
                        "Paper title",
                        "Paper abstract",
                    ).to_dict()
                ),
                encoding="utf-8",
            )
            with (
                patch("app.service._integration_root", return_value=root),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_direct_query_analysis(),
                ),
                patch(
                    "app.service._synthesizer_provider",
                    return_value=Mock(),
                ) as synthesizer_provider,
                patch("app.service.pipeline.chat", return_value="paper answer"),
            ):
                answer = service.run_pdf_chat(
                    "What is the contribution?",
                    str(paper_json),
                )

            self.assertEqual(answer, "paper answer")
            session_path = next(
                (root / "data" / "sessions").glob("*/session.jsonl")
            )
            rows = [
                json.loads(line)
                for line in session_path.read_text(encoding="utf-8").splitlines()
            ]
            route = next(row for row in rows if row["event"] == "route_selected")
            self.assertEqual(route["payload"]["input_type"], "pdf")
            self.assertEqual(route["payload"]["route"], "pdf_grounded_chat")
            self.assertFalse(route["payload"]["retrieval_used"])
            self.assertTrue(any(row["event"] == "assistant" for row in rows))
            query_analysis = synthesizer_provider.call_args.kwargs["query_analysis"]
            self.assertEqual(query_analysis["intent"], "question")
            self.assertEqual(
                query_analysis["field_sources"]["intent"],
                "pdf_grounded_route",
            )


if __name__ == "__main__":
    unittest.main()
