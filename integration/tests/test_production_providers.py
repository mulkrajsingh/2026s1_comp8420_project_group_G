"""Production-provider wiring tests for the integration CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from app import pipeline, service
from app.contracts import AnalysisResult, PaperRecord, ParsedPaper, RagEvidencePack, Recommendation
from app.providers.container import Providers
from app.providers.live_providers import (
    SubprocessPdfParser,
    SubprocessRecommender,
    SubprocessSynthesizer,
    write_corpus_slice,
)


class StaticSource:
    def get_corpus(self):
        return [PaperRecord("p1", "Title", "Abstract")]

    def search_topic(self, query: str, k: int = 5):
        raise AssertionError("integration must not rank topics")


class CorpusSliceTests(unittest.TestCase):
    def test_zero_limit_means_full_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "corpus.jsonl"
            output = root / "slice.jsonl"
            corpus.write_text(
                "".join(
                    json.dumps({"paper_id": f"p{index}"}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )

            write_corpus_slice(corpus, output, 0)

            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 3)


def _query_analysis(*, retrieval: bool) -> Mock:
    analysis = Mock(
        intent="question" if retrieval else "chit_chat",
        should_use_retrieval=retrieval,
        is_paper_recommendation=False,
        style="concise",
    )
    analysis.as_dict.return_value = {
        "intent": analysis.intent,
        "emotion": "neutral",
        "topic_expertise": "intermediate",
        "verbosity": "normal",
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


class LiveRecommender:
    def retrieve_evidence(self, query: str, mode: str = "offline"):
        return RagEvidencePack(query, [{"paper_id": "p1", "text": "evidence", "score": 1.0}])

    def recommend(self, query, candidates, k: int = 5):
        return Recommendation(
            query,
            [{
                "paper_id": "p1",
                "title": "Title",
                "score": 1.0,
                "apa_citation": "Bank APA",
                "why": "ranked by Bank",
            }],
        )


class StubPdfParser:
    def __init__(self, parsed: ParsedPaper):
        self._parsed = parsed

    def parse(self, pdf_path: str) -> ParsedPaper:
        return self._parsed


class LiveSynthesizer:
    def synthesize(self, parsed, query, evidence, recommendation, style="concise", model_mode="base"):
        return {
            "summary": "live summary",
            "key_findings": [],
            "research_gaps": [],
            "apa_citations": ["must not be used"],
        }

    def peer_review(self, parsed, model_mode="base"):
        raise AssertionError

    def answer(self, parsed, question, evidence):
        return "live answer"

    def answer_direct(self, question):
        return "direct live answer"

    def recommend_papers(self, query, recommendation):
        return list(recommendation.items)


class ProductionProviderTests(unittest.TestCase):
    def test_provider_metadata_distinguishes_registered_sources(self) -> None:
        providers = Providers()
        providers.add("paper_source", StaticSource(), "file-backed")
        providers.add("recommender", LiveRecommender(), "live")
        providers.add(
            "parser",
            StubPdfParser(_parsed_paper("p", "t", "a")),
            "live",
        )

        self.assertEqual(providers.source("paper_source"), "file-backed")
        self.assertEqual(providers.source("recommender"), "live")
        self.assertEqual(providers.source("parser"), "live")
        with self.assertRaises(RuntimeError):
            providers.require("synthesizer")

    def test_topic_pipeline_uses_retrieval_ranking_and_apa(self) -> None:
        providers = Providers()
        providers.add("paper_source", StaticSource(), "file-backed")
        providers.add("recommender", LiveRecommender(), "live")
        providers.add("synthesizer", LiveSynthesizer(), "live")

        result = pipeline.search_topic(providers, "topic")

        self.assertEqual(result.summary, "live summary")
        self.assertEqual(result.apa_citations, ["Bank APA"])
        self.assertEqual(
            result.flags["provider_sources"],
            {
                "paper_source": "file-backed",
                "recommender": "live",
                "synthesizer": "live",
            },
        )

    def test_missing_production_corpus_fails_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "dev_5k.jsonl"
            with self.assertRaisesRegex(FileNotFoundError, "Dataset production corpus missing"):
                service._resolve_corpus(str(missing))

    def test_retrieval_cli_is_invoked_and_current_artifacts_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "dev_5k.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            outputs = root / "outputs"

            def fake_run(command, cwd, capture_output, text):
                rec_out = Path(command[command.index("--out") + 1])
                rec_out.parent.mkdir(parents=True, exist_ok=True)
                rec_out.write_text(
                    json.dumps([{
                        "paper": {"paper_id": "p1", "title": "Ranked"},
                        "score": 0.9,
                        "apa_citation": "Bank APA",
                        "reason": "relevant",
                    }]),
                    encoding="utf-8",
                )
                (rec_out.parent / "rag_evidence_pack.json").write_text(
                    json.dumps({
                        "query": "topic",
                        "retrieval_mode": "offline",
                        "evidence_snippets": [{
                            "source_id": "p1",
                            "snippet": "grounding",
                        }],
                    }),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            provider = SubprocessRecommender(
                query="topic",
                corpus_jsonl=corpus,
                embedding_model="all-MiniLM-L6-v2",
                top_k=7,
                integration_outputs=outputs,
            )
            with patch("app.providers.live_providers.subprocess.run", side_effect=fake_run) as run:
                recommendation = provider.recommend("topic", [])

            command = run.call_args.args[0]
            self.assertIn("recommend-topic", command)
            self.assertEqual(command[command.index("--top-k") + 1], "7")
            self.assertEqual(
                command[command.index("--embedding-model") + 1],
                "all-MiniLM-L6-v2",
            )
            self.assertEqual(
                command[command.index("--retrieval-strategy") + 1],
                "hybrid_rrf",
            )
            self.assertNotIn("--tfidf-only", command)
            self.assertEqual(recommendation.items[0]["apa_citation"], "Bank APA")

    def test_retrieval_cli_passes_tfidf_retrieval_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "dev_5k.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            outputs = root / "outputs"

            def fake_run(command, cwd, capture_output, text):
                rec_out = Path(command[command.index("--out") + 1])
                rec_out.parent.mkdir(parents=True, exist_ok=True)
                rec_out.write_text("[]", encoding="utf-8")
                (rec_out.parent / "rag_evidence_pack.json").write_text(
                    json.dumps({"query": "topic", "retrieval_mode": "offline", "evidence_snippets": []}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            provider = SubprocessRecommender(
                query="topic",
                corpus_jsonl=corpus,
                retrieval_strategy="tfidf",
                top_k=5,
                integration_outputs=outputs,
            )
            with patch("app.providers.live_providers.subprocess.run", side_effect=fake_run) as run:
                provider.recommend("topic", [])

            command = run.call_args.args[0]
            self.assertEqual(
                command[command.index("--retrieval-strategy") + 1],
                "tfidf",
            )
            self.assertNotIn("--embedding-model", command)

    def test_llm_cli_is_invoked_for_synthesis_and_topic_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir)
            evidence = outputs / "rag_evidence_pack.json"
            evidence.write_text("{}", encoding="utf-8")

            def fake_run(command, cwd, capture_output, text):
                if "chat" in command:
                    Path(command[command.index("--out") + 1]).write_text(
                        "generated direct answer", encoding="utf-8"
                    )
                    Path(command[command.index("--metadata-out") + 1]).write_text(
                        json.dumps(
                            {
                                "backend": "ollama",
                                "model": "test",
                                "latency_seconds": 0.1,
                                "error": None,
                            }
                        ),
                        encoding="utf-8",
                    )
                    return subprocess.CompletedProcess(command, 0, "", "")
                Path(command[command.index("--out") + 1]).write_text(
                    "# analysis", encoding="utf-8"
                )
                Path(command[command.index("--json-out") + 1]).write_text(
                    json.dumps({
                        "summary": "generated answer",
                        "key_findings": [],
                        "research_gaps": [],
                    }),
                    encoding="utf-8",
                )
                (outputs / "llm_generation.json").write_text(
                    json.dumps({"backend": "ollama", "model": "test"}),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0, "", "")

            query_analysis = _query_analysis(retrieval=False).as_dict()
            provider = SubprocessSynthesizer(
                integration_outputs=outputs,
                query_analysis=query_analysis,
            )
            with patch("app.providers.live_providers.subprocess.run", side_effect=fake_run) as run:
                synthesized = provider.synthesize(None, "topic", Mock(), Mock())
                answer = provider.answer(None, "topic", Mock())
                direct_answer = provider.answer_direct("hi")

            self.assertEqual(synthesized["summary"], "generated answer")
            self.assertEqual(answer, "generated answer")
            self.assertEqual(direct_answer, "generated direct answer")
            self.assertEqual(run.call_count, 3)
            self.assertTrue((outputs / "llm_query_analysis.json").is_file())
            self.assertTrue(
                all(
                    any(command in call.args[0] for command in ("synthesize", "chat"))
                    for call in run.call_args_list
                )
            )
            self.assertTrue(
                all(
                    "--query-analysis" in call.args[0]
                    for call in run.call_args_list
                )
            )

    def test_pdf_summary_and_review_use_distinct_structured_llm_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir)
            parsed = _parsed_paper("p1", "Paper title", "Paper abstract")

            def fake_run(command, cwd, capture_output, text):
                md_out = Path(command[command.index("--out") + 1])
                json_out = Path(command[command.index("--json-out") + 1])
                metadata_out = Path(command[command.index("--metadata-out") + 1])
                md_out.write_text("# audit artifact", encoding="utf-8")
                metadata_out.write_text(
                    json.dumps({"backend": "ollama", "error": None}),
                    encoding="utf-8",
                )
                if "review" in command:
                    json_out.write_text(
                        json.dumps({"peer_review": "grounded review"}),
                        encoding="utf-8",
                    )
                else:
                    json_out.write_text(
                        json.dumps(
                            {
                                "summary": "clean summary",
                                "key_findings": ["finding"],
                                "research_gaps": ["gap"],
                            }
                        ),
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(command, 0, "", "")

            provider = SubprocessSynthesizer(integration_outputs=outputs)
            with patch(
                "app.providers.live_providers.subprocess.run",
                side_effect=fake_run,
            ) as run:
                summary = provider.synthesize(parsed, "query", Mock(), Mock())
                review = provider.peer_review(parsed)

            self.assertEqual(summary["summary"], "clean summary")
            self.assertEqual(summary["key_findings"], ["finding"])
            self.assertEqual(summary["research_gaps"], ["gap"])
            self.assertEqual(review, "grounded review")
            self.assertIn("summarize", run.call_args_list[0].args[0])
            self.assertIn("review", run.call_args_list[1].args[0])

    def test_synthesizer_forwards_prompt_strategy_to_llm_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outputs = Path(temp_dir)
            evidence = outputs / "rag_evidence_pack.json"
            evidence.write_text(
                json.dumps(
                    {
                        "query": "topic",
                        "retrieval_mode": "offline",
                        "evidence_snippets": [],
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(command, cwd, capture_output, text):
                Path(command[command.index("--out") + 1]).write_text(
                    "# analysis", encoding="utf-8"
                )
                Path(command[command.index("--json-out") + 1]).write_text(
                    json.dumps(
                        {
                            "summary": "answer",
                            "key_findings": [],
                            "research_gaps": [],
                        }
                    ),
                    encoding="utf-8",
                )
                (outputs / "llm_generation.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            provider = SubprocessSynthesizer(
                integration_outputs=outputs,
                prompt_strategy="few_shot",
            )
            with patch(
                "app.providers.live_providers.subprocess.run",
                side_effect=fake_run,
            ) as run:
                provider.synthesize(None, "topic", Mock(), Mock())

            command = run.call_args.args[0]
            self.assertIn("--prompt-strategy", command)
            self.assertEqual(
                command[command.index("--prompt-strategy") + 1],
                "few_shot",
            )

    def test_pdf_commands_fail_until_live_parser_is_available(self) -> None:
        providers = Providers()
        for operation in (
            lambda: pipeline.analyze_pdf(providers, "paper.pdf"),
            lambda: pipeline.peer_review(providers, "paper.pdf"),
        ):
            with self.assertRaisesRegex(RuntimeError, "live PDF parser unavailable"):
                operation()

    def test_pdf_nlp_cli_is_invoked_for_live_pdf_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf = root / "sample.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% mock\n")
            outputs = root / "outputs"

            canonical = {
                "metadata": {
                    "paper_id": "sample_abc",
                    "title": "Parsed Title",
                    "abstract": "Abstract text",
                    "authors": ["A. Author"],
                    "categories": [],
                    "published_date": None,
                    "venue": None,
                    "doi": None,
                    "arxiv_id": None,
                    "url": None,
                    "source": "uploaded_pdf",
                },
                "sections": {
                    "abstract": "Abstract text",
                    "introduction": "",
                    "method": "",
                    "results": "",
                    "conclusion": "",
                },
                "references": [],
                "keywords": [],
                "entities": {
                    "methods": [],
                    "datasets": [],
                    "tasks": [],
                    "metrics": [],
                    "institutions": [],
                },
            }

            def fake_run(command, cwd, capture_output, text):
                out = Path(command[command.index("--out") + 1])
                debug = Path(command[command.index("--debug-out") + 1])
                analysis = Path(command[command.index("--analysis-out") + 1])
                review = Path(command[command.index("--review-out") + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(canonical), encoding="utf-8")
                debug.write_text(json.dumps({"page_count": 1, "headings": []}), encoding="utf-8")
                analysis.write_text(json.dumps({}), encoding="utf-8")
                review.write_text(json.dumps({"checks": []}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "", "")

            provider = SubprocessPdfParser(integration_outputs=outputs)
            with patch("app.providers.live_providers.subprocess.run", side_effect=fake_run) as run:
                parsed = provider.parse(str(pdf))

            command = run.call_args.args[0]
            self.assertIn("analyze-paper", command)
            self.assertEqual(parsed.title, "Parsed Title")

    def test_pdf_chat_requires_live_synthesizer(self) -> None:
        parsed = _parsed_paper("p", "title", "abstract")
        providers = Providers().add(
            "synthesizer",
            LiveSynthesizer(),
            "file-backed",
        )
        with self.assertRaisesRegex(RuntimeError, "PDF-grounded chat requires live synthesizer"):
            pipeline.chat(providers, "question", parsed=parsed)

    def test_recommend_orchestration_uses_no_mock_provider(self) -> None:
        parsed = _parsed_paper("p", "topic", "abstract")
        session = Mock()

        providers = Providers()
        providers.add("paper_source", StaticSource(), "file-backed")
        providers.add("recommender", LiveRecommender(), "live")

        def recommend(active_providers, parsed, retrieval_mode):
            self.assertEqual(
                active_providers.source("paper_source"),
                "file-backed",
            )
            self.assertEqual(active_providers.source("recommender"), "live")
            return Recommendation(parsed.title, [])

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "dev_5k.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            with (
                patch("app.service.SessionLogger.create", return_value=session),
                patch(
                    "app.service._retrieval_providers",
                    return_value=(providers, Path(temp_dir) / "slice.jsonl"),
                ),
                patch("app.service.pipeline.recommend_for_parsed", side_effect=recommend),
            ):
                result = service.recommend_for_parsed(parsed, corpus=str(corpus))

        self.assertEqual(result.query, "topic")

    def test_topic_chat_orchestration_uses_live_bank_and_llm(self) -> None:
        session = Mock()
        registered: dict[str, str] = {}

        def configure(**kwargs):
            registered["style"] = kwargs["style"]
            providers = Providers()
            providers.add("paper_source", StaticSource(), "file-backed")
            providers.add("recommender", LiveRecommender(), "live")
            providers.add("synthesizer", LiveSynthesizer(), "live")
            return providers, Path("/tmp/poc_corpus.jsonl")

        def chat_response(providers, question, retrieval_mode, query_analysis, parsed=None):
            active = {
                role: providers.source(role)
                for role in ("paper_source", "recommender", "synthesizer")
            }
            self.assertEqual(active["recommender"], "live")
            self.assertTrue(query_analysis.should_use_retrieval)
            return {
                "kind": "message",
                "answer": "live answer",
                "recommended_papers": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "dev_5k.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            with (
                patch("app.service.SessionLogger.create", return_value=session),
                patch(
                    "app.service.pipeline.analyze_query",
                    return_value=_query_analysis(retrieval=True),
                ),
                patch("app.service._topic_providers", side_effect=configure),
                patch("app.service.pipeline.chat_response", side_effect=chat_response),
            ):
                answer = service.chat_topic("topic question", corpus=str(corpus))

        self.assertEqual(answer["answer"], "live answer")
        self.assertEqual(registered["style"], "concise")
        analysis = session.log_query_analysis.call_args.args[0]
        self.assertEqual(analysis["style"], "concise")
        self.assertEqual(analysis["style_source"], "cosine_similarity")

    def test_topic_chat_greeting_uses_direct_llm_without_retrieval(self) -> None:
        session = Mock()
        registered: dict[str, str] = {}

        def configure_synthesizer(**kwargs):
            registered["style"] = kwargs["style"]
            return LiveSynthesizer()

        with (
            patch("app.service.SessionLogger.create", return_value=session),
            patch(
                "app.service.pipeline.analyze_query",
                return_value=_query_analysis(retrieval=False),
            ),
            patch(
                "app.service._resolve_corpus",
                side_effect=AssertionError("direct chat must not resolve the corpus"),
            ),
            patch(
                "app.service._topic_providers",
                side_effect=AssertionError("direct chat must not register retrieval"),
            ),
            patch(
                "app.service._synthesizer_provider",
                side_effect=configure_synthesizer,
            ) as register_direct,
            patch(
                "app.service.pipeline.chat_response",
                return_value={
                    "kind": "message",
                    "answer": "direct live answer",
                    "recommended_papers": [],
                },
            ),
        ):
            answer = service.chat_topic("hi")

        self.assertEqual(answer["answer"], "direct live answer")
        self.assertEqual(registered["style"], "concise")
        register_direct.assert_called_once()
        session.log_route.assert_called_once_with(
            input_type="text",
            route="direct_llm_chat",
            retrieval_used=False,
        )

    def test_pdf_analysis_uses_parsed_title_for_retrieval_query(self) -> None:
        parsed = _parsed_paper(
            "p1",
            "Visual Continuous Control with DrQ-v2",
            "A model-free RL algorithm for visual control.",
        )
        self.assertEqual(
            pipeline.pdf_analysis_query(parsed),
            "Visual Continuous Control with DrQ-v2",
        )
        parsed_no_title = _parsed_paper("p2", "", "Abstract-only retrieval query text.")
        self.assertEqual(
            pipeline.pdf_analysis_query(parsed_no_title),
            "Abstract-only retrieval query text.",
        )

    def test_run_analyze_pdf_registers_retrieval_with_parsed_title(self) -> None:
        session = Mock()
        captured: dict[str, str] = {}

        canonical = {
            "metadata": {
                "paper_id": "sample_abc",
                "title": "Parsed Retrieval Title",
                "abstract": "Fallback abstract",
                "authors": [],
                "categories": [],
                "published_date": None,
                "venue": None,
                "doi": None,
                "arxiv_id": None,
                "url": None,
                "source": "uploaded_pdf",
            },
            "sections": {
                "abstract": "Fallback abstract",
                "introduction": "",
                "method": "",
                "results": "",
                "conclusion": "",
            },
            "references": [],
            "keywords": [],
            "entities": {
                "methods": [],
                "datasets": [],
                "tasks": [],
                "metrics": [],
                "institutions": [],
            },
        }

        parser = StubPdfParser(ParsedPaper.from_dict(canonical))

        def configure_pdf(**kwargs):
            captured["query"] = pipeline.pdf_analysis_query(kwargs["parsed"])
            providers = Providers().add("parser", parser, "live")
            providers.add("paper_source", StaticSource(), "file-backed")
            providers.add("recommender", LiveRecommender(), "live")
            providers.add("synthesizer", LiveSynthesizer(), "live")
            return providers, Path("/tmp/poc_corpus.jsonl")

        def fake_analyze_parsed(providers, parsed, pdf_path, **kwargs):
            self.assertEqual(parsed.title, "Parsed Retrieval Title")
            self.assertTrue(kwargs.get("with_related_papers"))
            return AnalysisResult("pdf", pdf_path, summary="ok")

        with (
            patch("app.service.SessionLogger.create", return_value=session),
            patch("app.service._parser_provider", return_value=parser),
            patch("app.service._pdf_providers", side_effect=configure_pdf),
            patch("app.service.pipeline.analyze_parsed", side_effect=fake_analyze_parsed),
            patch("app.service.write_json"),
            patch("app.service.write_text"),
            patch("app.service.render_markdown", return_value=""),
        ):
            out = service.run_analyze_pdf(
                "paper.pdf",
                with_related_papers=True,
            )

        self.assertEqual(captured["query"], "Parsed Retrieval Title")
        self.assertEqual(out["result"].summary, "ok")

    def test_run_analyze_pdf_no_related_papers_skips_retrieval(self) -> None:
        session = Mock()
        retrieval_called = {"value": False}

        parser = StubPdfParser(_parsed_paper("p", "Title", "Abstract"))

        def configure_pdf(**kwargs):
            retrieval_called["value"] = kwargs["with_related_papers"]
            providers = Providers().add("parser", parser, "live")
            providers.add("synthesizer", LiveSynthesizer(), "live")
            return providers, None

        def fake_analyze_parsed(providers, parsed, pdf_path, **kwargs):
            self.assertFalse(kwargs.get("with_related_papers"))
            return AnalysisResult("pdf", pdf_path, summary="paper-only")

        with (
            patch("app.service.SessionLogger.create", return_value=session),
            patch("app.service._parser_provider", return_value=parser),
            patch("app.service._pdf_providers", side_effect=configure_pdf),
            patch("app.service.pipeline.analyze_parsed", side_effect=fake_analyze_parsed),
            patch("app.service.write_json"),
            patch("app.service.write_text"),
            patch("app.service.render_markdown", return_value=""),
        ):
            service.run_analyze_pdf(
                "paper.pdf",
                with_related_papers=False,
                retrieval_mode="none",
            )

        self.assertFalse(retrieval_called["value"])

    def test_api_routes_use_production_orchestration_not_pipeline_mocks(self) -> None:
        from app.api import ChatReq, RecommendReq, TopicReq, chat, recommend, search_topic

        topic_result = AnalysisResult("topic", "q", summary="topic ok")
        with patch("app.service.run_topic", return_value={"result": topic_result}) as run_topic:
            payload = search_topic(TopicReq(query="topic"))
        self.assertEqual(payload["summary"], "topic ok")
        run_topic.assert_called_once()

        parsed = _parsed_paper("p", "topic", "abstract")
        with patch(
            "app.service.recommend_for_parsed",
            return_value=Recommendation("topic", []),
        ) as recommend_for_parsed:
            rec = recommend(RecommendReq(parsed=parsed.to_dict()))
        self.assertEqual(rec["query"], "topic")
        recommend_for_parsed.assert_called_once()

        with patch(
            "app.service.chat_topic",
            return_value={
                "kind": "message",
                "answer": "chat ok",
                "recommended_papers": [],
            },
        ) as chat_topic:
            request = ChatReq(question="hello")
            answer = chat(request)
        self.assertEqual(answer["answer"], "chat ok")
        self.assertEqual(request.style, "auto")
        chat_topic.assert_called_once_with(
            "hello",
            llm_model="qwen3:8b",
            style="auto",
            retrieval_mode="offline",
            retrieval_embedding_model="allenai/specter2_base",
            retrieval_top_k=5,
            retrieval_strategy="hybrid_rrf",
            prompt_strategy="zero_shot",
        )

if __name__ == "__main__":
    unittest.main()
