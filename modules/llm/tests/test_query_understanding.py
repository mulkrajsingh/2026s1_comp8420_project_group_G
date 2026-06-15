"""Tests for semantic query understanding and transformer fallback."""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path
from typing import Sequence

import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from app.prompt_library import paper_summary_prompt_record
from app.query_understanding import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    QueryAnalysis,
    QueryModelError,
    SemanticQueryAnalyzer,
    _PROTOTYPES,
)
from app.runtime import prompt_text_for_record

PARSED_ARTIFACT = (
    REPO_ROOT / "tests" / "papers" / "artifacts" / "drq_v2_parsed.json"
)


def _load_parsed_paper() -> dict:
    if PARSED_ARTIFACT.is_file():
        return json.loads(PARSED_ARTIFACT.read_text(encoding="utf-8"))
    return {
        "metadata": {
            "paper_id": "uploaded_test",
            "title": "Image Augmentation Is All You Need",
            "abstract": "Deep reinforcement learning from pixels.",
            "authors": ["Kostrikov", "Nair", "Zhou"],
            "categories": ["cs.LG"],
            "published_date": "2021-07-20",
            "venue": None,
            "doi": None,
            "arxiv_id": "2107.09645",
            "url": "https://arxiv.org/abs/2107.09645",
            "source": "uploaded_pdf",
        },
        "sections": {
            "abstract": "Deep reinforcement learning from pixels.",
            "introduction": "Sample introduction.",
            "method": "Sample method.",
            "results": "Sample results.",
            "conclusion": "Sample conclusion.",
        },
        "references": ["Reference one."],
        "keywords": ["reinforcement learning"],
        "entities": {
            "methods": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "institutions": [],
        },
    }


class ScriptedEncoder:
    model_name = "test/minilm"

    def __init__(
        self,
        targets: dict[str, str],
        confidences: dict[str, float] | None = None,
    ) -> None:
        self.targets = targets
        self.confidences = confidences or {}
        self.prototype_rows = [
            (field_name, label, text)
            for field_name, labels in _PROTOTYPES.items()
            for label, examples in labels.items()
            for text in examples
        ]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 1:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        vectors: list[list[float]] = []
        self.assert_prototype_order(texts)
        for field_name, label, _ in self.prototype_rows:
            if label != self.targets[field_name]:
                vectors.append([0.0, 1.0])
                continue
            confidence = self.confidences.get(field_name, 0.9)
            vectors.append(
                [confidence, math.sqrt(max(0.0, 1.0 - confidence**2))]
            )
        return np.asarray(vectors, dtype=np.float32)

    def assert_prototype_order(self, texts: Sequence[str]) -> None:
        expected = [text for _, _, text in self.prototype_rows]
        if list(texts) != expected:
            raise AssertionError("Unexpected prototype encoding order")


class ScriptedFallback:
    model_name = "test/tinybert-cross-encoder"

    def __init__(
        self,
        predictions: dict[str, str] | None = None,
        *,
        confidence: float = 0.88,
    ) -> None:
        self.predictions = predictions or {}
        self.confidence = confidence
        self.calls: list[str] = []

    def classify_many(
        self,
        text: str,
        candidates: dict[str, dict[str, tuple[str, ...]]],
    ) -> dict[str, tuple[str, float]]:
        del text
        self.calls.extend(candidates)
        return {
            field_name: (
                self.predictions.get(
                    field_name,
                    next(iter(_PROTOTYPES[field_name])),
                ),
                self.confidence,
            )
            for field_name in candidates
        }


def _targets(**overrides: str) -> dict[str, str]:
    values = {
        "intent": "request_explanation",
        "emotion": "confused",
        "topic_expertise": "beginner",
        "verbosity": "detailed",
        "style": "beginner",
    }
    values.update(overrides)
    return values


def _analyzer(
    targets: dict[str, str],
    *,
    confidences: dict[str, float] | None = None,
    fallback_predictions: dict[str, str] | None = None,
    fallback_confidence: float = 0.88,
) -> tuple[SemanticQueryAnalyzer, ScriptedFallback]:
    fallback = ScriptedFallback(
        fallback_predictions,
        confidence=fallback_confidence,
    )
    return (
        SemanticQueryAnalyzer(
            encoder=ScriptedEncoder(targets, confidences),
            fallback=fallback,
        ),
        fallback,
    )


class QueryUnderstandingTests(unittest.TestCase):
    def test_analysis_round_trip_supports_subprocess_reuse(self) -> None:
        analyzer, _ = _analyzer(_targets(intent="chit_chat"))
        original = analyzer.analyze("hi")

        restored = QueryAnalysis.from_dict(original.as_dict())

        self.assertEqual(restored, original)
        self.assertFalse(restored.should_use_retrieval)

    def test_high_cosine_confidence_does_not_load_fallback(self) -> None:
        analyzer, fallback = _analyzer(_targets())

        analysis = analyzer.analyze("I do not understand BM25")

        self.assertEqual(analysis.intent, "request_explanation")
        self.assertEqual(analysis.emotion, "confused")
        self.assertEqual(analysis.topic_expertise, "beginner")
        self.assertEqual(analysis.verbosity, "detailed")
        self.assertEqual(analysis.style, "beginner")
        self.assertFalse(analysis.fallback_used)
        self.assertEqual(analysis.style_source, "cosine_similarity")
        self.assertEqual(fallback.calls, [])
        self.assertGreaterEqual(
            analysis.confidence,
            DEFAULT_CONFIDENCE_THRESHOLD,
        )

    def test_low_cosine_confidence_loads_tinybert_for_that_field(self) -> None:
        analyzer, fallback = _analyzer(
            _targets(style="concise"),
            confidences={"style": 0.69},
            fallback_predictions={"style": "reviewer"},
            fallback_confidence=0.93,
        )

        analysis = analyzer.analyze("Assess the methodological rigor")

        self.assertEqual(analysis.style, "reviewer")
        self.assertEqual(analysis.style_source, "tinybert_fallback")
        self.assertEqual(analysis.field_confidences["style"], 0.93)
        self.assertEqual(fallback.calls, ["style"])
        self.assertTrue(analysis.fallback_used)
        self.assertEqual(analysis.fallback_fields, ("style",))
        self.assertLess(
            analysis.cosine_confidence,
            DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.assertEqual(analysis.fallback_model, fallback.model_name)

    def test_weak_auto_reviewer_style_respects_concise_verbosity(self) -> None:
        analyzer, _ = _analyzer(
            _targets(style="reviewer", verbosity="concise"),
            confidences={"style": 0.2, "verbosity": 0.2},
            fallback_predictions={
                "style": "reviewer",
                "verbosity": "concise",
            },
            fallback_confidence=0.2,
        )

        analysis = analyzer.analyze("How does an LLM get relevant data?")

        self.assertEqual(analysis.style, "concise")
        self.assertEqual(analysis.style_source, "verbosity_fallback")

    def test_threshold_of_seventy_percent_stays_on_cosine_path(self) -> None:
        analyzer, fallback = _analyzer(
            _targets(),
            confidences={"intent": DEFAULT_CONFIDENCE_THRESHOLD},
        )

        analysis = analyzer.analyze("Explain retrieval augmentation")

        self.assertEqual(analysis.field_sources["intent"], "cosine_similarity")
        self.assertNotIn("intent", fallback.calls)

    def test_multiple_low_confidence_fields_share_one_fallback_backend(self) -> None:
        analyzer, fallback = _analyzer(
            _targets(),
            confidences={"intent": 0.4, "emotion": 0.5},
            fallback_predictions={
                "intent": "debugging",
                "emotion": "frustrated",
            },
        )

        analysis = analyzer.analyze("The retrieval system keeps failing")

        self.assertEqual(analysis.intent, "debugging")
        self.assertEqual(analysis.emotion, "frustrated")
        self.assertEqual(fallback.calls, ["intent", "emotion"])
        self.assertEqual(
            analysis.field_sources["intent"],
            "tinybert_fallback",
        )

    def test_explicit_style_override_skips_style_fallback(self) -> None:
        analyzer, fallback = _analyzer(
            _targets(style="beginner"),
            confidences={"style": 0.2},
        )

        analysis = analyzer.analyze(
            "Explain BM25",
            style_override="reviewer",
        )

        self.assertEqual(analysis.style, "reviewer")
        self.assertEqual(analysis.style_source, "explicit_override")
        self.assertEqual(analysis.field_confidences["style"], 1.0)
        self.assertNotIn("style", fallback.calls)

    def test_chit_chat_intent_skips_retrieval(self) -> None:
        analyzer, _ = _analyzer(
            _targets(
                intent="chit_chat",
                emotion="positive",
                verbosity="concise",
                style="concise",
            )
        )

        analysis = analyzer.analyze("Hello, how are you?")

        self.assertEqual(analysis.intent, "chit_chat")
        self.assertFalse(analysis.should_use_retrieval)

    def test_similarity_scores_are_preserved_for_observability(self) -> None:
        analyzer, _ = _analyzer(_targets(style="technical"))

        analysis = analyzer.analyze("Explain the implementation architecture")

        self.assertEqual(
            analysis.style_scores,
            analysis.similarity_scores["style"],
        )
        self.assertGreater(
            analysis.style_scores["technical"],
            analysis.style_scores["concise"],
        )
        self.assertEqual(analysis.embedding_model, "test/minilm")

    def test_unknown_fallback_label_fails_closed(self) -> None:
        class InvalidFallback(ScriptedFallback):
            def classify_many(self, text, candidates):
                del text, candidates
                return {"intent": ("not a configured label", 0.99)}

        analyzer = SemanticQueryAnalyzer(
            encoder=ScriptedEncoder(
                _targets(),
                confidences={"intent": 0.1},
            ),
            fallback=InvalidFallback(),
        )

        with self.assertRaisesRegex(QueryModelError, "unknown intent label"):
            analyzer.analyze("Ambiguous query")

    def test_empty_query_is_rejected(self) -> None:
        analyzer, _ = _analyzer(_targets())

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            analyzer.analyze("   ")

    def test_analysis_becomes_backend_prompt_instruction(self) -> None:
        analyzer, _ = _analyzer(_targets())
        analysis = analyzer.analyze("I do not understand BM25")
        record = paper_summary_prompt_record(
            _load_parsed_paper(),
            analysis.style,
            user_query="I do not understand BM25",
            query_analysis=analysis.as_dict(),
        )

        prompt = prompt_text_for_record(record, "zero_shot")
        self.assertIn("supportive beginner-friendly language", prompt)
        self.assertIn('"emotion": "confused"', prompt)
        self.assertNotIn("similarity_scores", prompt)

    def test_specific_paper_question_uses_question_answer_prompt(self) -> None:
        record = paper_summary_prompt_record(
            _load_parsed_paper(),
            "concise",
            user_query="What baseline methods does this paper compare against?",
            query_analysis={"intent": "question", "style": "concise"},
        )

        self.assertEqual(record["task"], "paper_question_answer")
        self.assertIn(
            "What baseline methods does this paper compare against?",
            record["zero_shot_prompt"],
        )
        self.assertTrue(
            record["expected_output_contract"]["must_answer_user_request_directly"]
        )

    def test_summary_prompt_forbids_invented_limitations(self) -> None:
        paper = _load_parsed_paper()
        paper.setdefault("analysis", {})["structural_checks"] = [
            {"code": "no_explicit_limitations"}
        ]
        record = paper_summary_prompt_record(
            paper,
            "technical",
            user_query="Summarize the paper.",
            query_analysis={"intent": "request_explanation", "style": "technical"},
        )

        self.assertEqual(record["task"], "uploaded_paper_summary")
        self.assertIn("Do not invent limitations", record["zero_shot_prompt"])
        self.assertFalse(
            record["input"]["summary_presence_signals"][
                "explicit_limitations_section"
            ]
        )


    def test_paper_recommendation_request_is_detected_before_semantic_classifier(self) -> None:
        analysis = SemanticQueryAnalyzer(
            encoder=ScriptedEncoder(_targets(intent="critique", style="reviewer")),
            fallback=ScriptedFallback(),
        ).analyze("Suggest papers on artificial intelligence")
        self.assertEqual(analysis.intent, "paper_recommendation")
        self.assertEqual(analysis.style, "concise")
        self.assertEqual(analysis.field_sources["intent"], "pattern_match")

    def test_capability_greeting_is_detected_before_semantic_classifier(self) -> None:
        analysis = SemanticQueryAnalyzer(
            encoder=ScriptedEncoder(
                _targets(intent="request_explanation", style="technical")
            ),
            fallback=ScriptedFallback(),
        ).analyze("Hi, I am new to this tool. What can you help me with?")

        self.assertEqual(analysis.intent, "chit_chat")
        self.assertEqual(analysis.style, "concise")
        self.assertEqual(analysis.field_sources["intent"], "pattern_match")
        self.assertFalse(analysis.should_use_retrieval)

    def test_greeting_with_research_request_uses_semantic_classifier(self) -> None:
        analysis = SemanticQueryAnalyzer(
            encoder=ScriptedEncoder(
                _targets(intent="request_explanation", style="technical")
            ),
            fallback=ScriptedFallback(),
        ).analyze("Hi, explain retrieval augmented generation.")

        self.assertEqual(analysis.intent, "request_explanation")
        self.assertTrue(analysis.should_use_retrieval)

    def test_extract_recommendation_topic(self) -> None:
        from app.query_understanding import extract_recommendation_topic

        self.assertEqual(
            extract_recommendation_topic("Recommend papers about artificial intelligence"),
            "artificial intelligence",
        )
        self.assertEqual(
            extract_recommendation_topic("Find papers on retrieval augmented generation."),
            "retrieval augmented generation",
        )
        self.assertEqual(
            extract_recommendation_topic(
                "Recommend five research papers on scientific question answering."
            ),
            "scientific question answering",
        )


if __name__ == "__main__":
    unittest.main()
