"""Semantic user-query analysis with a transformer fallback."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Any, Sequence

import numpy as np


VALID_STYLES = ("auto", "concise", "technical", "beginner", "reviewer")
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MIN_AUTO_STYLE_FALLBACK_CONFIDENCE = 0.50

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DEFAULT_FALLBACK_MODEL = "cross-encoder/stsb-TinyBERT-L4"
DEFAULT_FALLBACK_REVISION = "1f478482145797a0cac02d6b9ab3c65c9d25fb1e"

_PAPER_RECOMMENDATION_PATTERNS = (
    re.compile(
        r"(?:suggest|recommend|find|show|list|give\s+me)\s+"
        r"(?:me\s+)?(?:(?:some|\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
        r"(?:research\s+)?papers?\s+"
        r"(?:on|about|for|regarding)\s+(?P<topic>.+)",
        re.I,
    ),
    re.compile(
        r"(?:what|which)\s+(?:research\s+)?papers?\s+"
        r"(?:should\s+i\s+read|do\s+you\s+recommend|are\s+(?:there|recommended))\s+"
        r"(?:on|about|for|regarding)\s+(?P<topic>.+)",
        re.I,
    ),
)

_DIRECT_CHAT_PATTERNS = (
    re.compile(
        r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening))[!,. ]*$",
        re.I,
    ),
    re.compile(r"^(?:thanks|thank\s+you|cheers)(?:\s+.*)?[!,. ]*$", re.I),
    re.compile(
        r"^(?:(?:hi|hello|hey)[!,. ]+)?"
        r"(?:i(?:'m|\s+am)\s+new(?:\s+to\s+this\s+tool)?[!,. ]*)?"
        r"(?:what\s+can\s+you\s+(?:do|help\s+me\s+with)|"
        r"how\s+can\s+you\s+help(?:\s+me)?|"
        r"what\s+do\s+you\s+do|"
        r"tell\s+me\s+about\s+(?:yourself|this\s+tool))"
        r"[?!. ]*$",
        re.I,
    ),
    re.compile(
        r"^(?:hi|hello|hey)[!,. ]+(?:how\s+are\s+you|"
        r"what\s+can\s+you\s+(?:do|help\s+me\s+with))"
        r"[?!. ]*$",
        re.I,
    ),
)


def _normalize_topic(topic: str) -> str:
    cleaned = topic.strip().rstrip("?.!")
    return re.sub(r"\s+", " ", cleaned)


def extract_recommendation_topic(query: str) -> str | None:
    """Extract the retrieval topic from a high-confidence recommendation request."""
    normalized = re.sub(r"\s+", " ", query.strip())
    if not normalized:
        return None
    for pattern in _PAPER_RECOMMENDATION_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        topic = _normalize_topic(match.group("topic"))
        if topic:
            return topic
    return None


def is_paper_recommendation_request(query: str) -> bool:
    return extract_recommendation_topic(query) is not None


def is_direct_chit_chat_request(query: str) -> bool:
    """Recognize only unambiguous social or tool-capability messages."""
    normalized = re.sub(r"\s+", " ", query.strip())
    if not normalized:
        return False
    return any(pattern.fullmatch(normalized) for pattern in _DIRECT_CHAT_PATTERNS)


_PROTOTYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "intent": {
        "paper_recommendation": (
            "Suggest papers on machine learning.",
            "Recommend research papers about artificial intelligence.",
            "Find papers on retrieval augmented generation.",
            "Show me papers about natural language processing.",
            "What papers should I read about computer vision?",
        ),
        "chit_chat": (
            "Hi",
            "Hello",
            "Thanks",
            "How are you?",
            "how areyou?",
            "Hello, how are you?",
            "Hi there, thanks for your help.",
            "This is casual conversation rather than a research question.",
        ),
        "debugging": (
            "Help me debug this error because the system keeps failing.",
            "Why does this implementation throw an exception?",
            "The code is broken and I need troubleshooting help.",
        ),
        "critique": (
            "Critique this research paper and assess its limitations.",
            "Review the methodology, strengths, weaknesses, and evidence.",
            "Evaluate the rigor and validity of these research findings.",
        ),
        "request_explanation": (
            "Explain this concept so I can understand it.",
            "Teach me how this method works step by step.",
            "Describe and summarize the idea for me.",
        ),
        "question": (
            "What is the answer to this research question?",
            "Which approach should be used for this topic?",
            "I have a factual question about research.",
        ),
    },
    "emotion": {
        "frustrated": (
            "I am frustrated and annoyed because this still does not work.",
            "This keeps failing and I am upset with the broken system.",
        ),
        "confused": (
            "I am confused, lost, and do not understand this.",
            "This is unclear to me and I need help understanding it.",
        ),
        "positive": (
            "Hi",
            "Hello, how are you?",
            "Thanks",
            "I am happy, grateful, and pleased with the help.",
            "Thanks, this is great and helpful.",
        ),
        "neutral": (
            "What is retrieval augmented generation?",
            "Assess the methodological rigor and evidence quality.",
            "I am asking calmly without expressing a strong emotion.",
            "This message has a neutral and objective tone.",
        ),
    },
    "topic_expertise": {
        "advanced": (
            "Give a formal mathematical derivation with equations and proof.",
            "Discuss algorithmic complexity, gradients, and implementation details.",
        ),
        "beginner": (
            "I am a beginner, so explain this in simple plain English.",
            "Give an intuitive introduction because I am new to the topic.",
        ),
        "intermediate": (
            "Hi",
            "This retrieval system keeps failing.",
            "What is retrieval augmented generation?",
            "Explain the research topic for someone with general technical knowledge.",
            "Give a standard academic explanation without assuming expert specialization.",
        ),
    },
    "verbosity": {
        "concise": (
            "Hi",
            "Hello, how are you?",
            "how areyou?",
            "This retrieval system keeps failing.",
            "Answer briefly, directly, and in one short summary.",
            "Give only the key points in a concise response.",
        ),
        "detailed": (
            "Give a thorough step-by-step answer with substantial detail.",
            "Provide a comprehensive explanation and show the full reasoning.",
        ),
        "normal": (
            "What is retrieval augmented generation?",
            "Use a normal amount of detail in the response.",
            "Give a balanced answer that is neither very short nor exhaustive.",
        ),
    },
    "style": {
        "concise": (
            "Hi",
            "Hello, how are you?",
            "how areyou?",
            "This retrieval system keeps failing.",
            "What is retrieval augmented generation?",
            "Use a concise, direct response style focused on key points.",
            "Write a brief compact answer without unnecessary detail.",
        ),
        "beginner": (
            "Use accessible beginner-friendly language and intuitive explanations.",
            "Explain gently in simple plain English for a new learner.",
        ),
        "reviewer": (
            "Use a critical peer-review style assessing rigor and limitations.",
            "Evaluate strengths, weaknesses, methodology, evidence, and validity.",
        ),
        "technical": (
            "Use a precise technical style with formal details and equations.",
            "Discuss algorithms, architecture, implementation, and complexity.",
        ),
    },
}


class QueryModelError(RuntimeError):
    """Raised when a configured local query-understanding model cannot load."""


@dataclass(frozen=True)
class QueryAnalysis:
    intent: str
    emotion: str
    topic_expertise: str
    verbosity: str
    style: str
    confidence: float
    cosine_confidence: float = 0.0
    style_source: str = "cosine_similarity"
    style_scores: dict[str, float] = field(default_factory=dict)
    field_confidences: dict[str, float] = field(default_factory=dict)
    field_sources: dict[str, str] = field(default_factory=dict)
    similarity_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_fields: tuple[str, ...] = ()
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    fallback_model: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> QueryAnalysis:
        """Restore a previously computed analysis without loading classifiers."""
        required = (
            "intent",
            "emotion",
            "topic_expertise",
            "verbosity",
            "style",
            "confidence",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(
                "QueryAnalysis is missing required fields: " + ", ".join(missing)
            )
        return cls(
            intent=str(payload["intent"]),
            emotion=str(payload["emotion"]),
            topic_expertise=str(payload["topic_expertise"]),
            verbosity=str(payload["verbosity"]),
            style=str(payload["style"]),
            confidence=float(payload["confidence"]),
            cosine_confidence=float(payload.get("cosine_confidence", 0.0)),
            style_source=str(payload.get("style_source", "cosine_similarity")),
            style_scores=dict(payload.get("style_scores") or {}),
            field_confidences=dict(payload.get("field_confidences") or {}),
            field_sources=dict(payload.get("field_sources") or {}),
            similarity_scores=dict(payload.get("similarity_scores") or {}),
            fallback_used=bool(payload.get("fallback_used", False)),
            fallback_fields=tuple(payload.get("fallback_fields") or ()),
            embedding_model=str(
                payload.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            ),
            fallback_model=(
                str(payload["fallback_model"])
                if payload.get("fallback_model") is not None
                else None
            ),
        )

    @property
    def should_use_retrieval(self) -> bool:
        return self.intent != "chit_chat"

    @property
    def is_paper_recommendation(self) -> bool:
        return self.intent == "paper_recommendation"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class SentenceTransformerEncoder:
    """Lazily load the pinned MiniLM sentence-embedding model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        revision: str | None = DEFAULT_EMBEDDING_REVISION,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.local_files_only = (
            _env_flag("QUERY_ANALYZER_LOCAL_FILES_ONLY", False)
            if local_files_only is None
            else local_files_only
        )

    @cached_property
    def _model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise QueryModelError(
                "Unable to load the query embedding model "
                f"{self.model_name!r}. Install/cache the pinned model or set "
                "QUERY_EMBEDDING_MODEL and QUERY_EMBEDDING_REVISION."
            ) from exc

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


class TinyBertCrossEncoderClassifier:
    """Lazily rerank low-confidence prototypes with a small TinyBERT model."""

    def __init__(
        self,
        model_name: str = DEFAULT_FALLBACK_MODEL,
        *,
        revision: str | None = DEFAULT_FALLBACK_REVISION,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.local_files_only = (
            _env_flag("QUERY_ANALYZER_LOCAL_FILES_ONLY", False)
            if local_files_only is None
            else local_files_only
        )

    @cached_property
    def _model(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder

            return CrossEncoder(
                self.model_name,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise QueryModelError(
                "Cosine confidence was below 0.70, but the TinyBERT fallback "
                f"{self.model_name!r} could not be loaded. Install/cache the "
                "pinned model or set QUERY_FALLBACK_MODEL and "
                "QUERY_FALLBACK_REVISION."
            ) from exc

    def classify_many(
        self,
        text: str,
        candidates: dict[str, dict[str, tuple[str, ...]]],
    ) -> dict[str, tuple[str, float]]:
        pairs: list[tuple[str, str]] = []
        owners: list[tuple[str, str]] = []
        for field_name, labels in candidates.items():
            for label, examples in labels.items():
                pairs.extend((text, example) for example in examples)
                owners.extend([(field_name, label)] * len(examples))

        scores = np.asarray(
            self._model.predict(pairs, show_progress_bar=False),
            dtype=np.float32,
        )
        field_scores = {
            field_name: {label: 0.0 for label in labels}
            for field_name, labels in candidates.items()
        }
        for (field_name, label), score in zip(owners, scores, strict=True):
            field_scores[field_name][label] = max(
                field_scores[field_name][label],
                float(score),
            )
        return {
            field_name: max(scores_by_label.items(), key=lambda item: item[1])
            for field_name, scores_by_label in field_scores.items()
        }


class SemanticQueryAnalyzer:
    """Use embedding cosine similarity first, then TinyBERT below threshold."""

    def __init__(
        self,
        *,
        encoder=None,
        fallback=None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.encoder = encoder or SentenceTransformerEncoder(
            model_name=os.getenv("QUERY_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            revision=os.getenv(
                "QUERY_EMBEDDING_REVISION",
                DEFAULT_EMBEDDING_REVISION,
            ),
        )
        self.fallback = fallback or TinyBertCrossEncoderClassifier(
            model_name=os.getenv("QUERY_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
            revision=os.getenv(
                "QUERY_FALLBACK_REVISION",
                DEFAULT_FALLBACK_REVISION,
            ),
        )
        self.confidence_threshold = confidence_threshold
        self._prototype_vectors: dict[str, dict[str, np.ndarray]] | None = None

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().replace("\u2019", "'"))

    @staticmethod
    def _unit_rows(matrix: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float32))
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms != 0,
        )

    def _load_prototype_vectors(self) -> dict[str, dict[str, np.ndarray]]:
        if self._prototype_vectors is not None:
            return self._prototype_vectors

        texts: list[str] = []
        positions: list[tuple[str, str, int]] = []
        for field_name, labels in _PROTOTYPES.items():
            for label, examples in labels.items():
                start = len(texts)
                texts.extend(examples)
                positions.append((field_name, label, start))

        encoded = self._unit_rows(self.encoder.encode(texts))
        vectors: dict[str, dict[str, np.ndarray]] = {
            field_name: {} for field_name in _PROTOTYPES
        }
        for index, (field_name, label, start) in enumerate(positions):
            next_start = (
                positions[index + 1][2]
                if index + 1 < len(positions)
                else len(texts)
            )
            vectors[field_name][label] = encoded[start:next_start]
        self._prototype_vectors = vectors
        return vectors

    def _cosine_predictions(
        self,
        query: str,
    ) -> tuple[dict[str, str], dict[str, float], dict[str, dict[str, float]]]:
        query_vector = self._unit_rows(self.encoder.encode([query]))[0]
        predictions: dict[str, str] = {}
        confidences: dict[str, float] = {}
        all_scores: dict[str, dict[str, float]] = {}

        for field_name, label_vectors in self._load_prototype_vectors().items():
            scores = {
                label: max(0.0, float(np.max(vectors @ query_vector)))
                for label, vectors in label_vectors.items()
            }
            label, confidence = max(scores.items(), key=lambda item: item[1])
            predictions[field_name] = label
            confidences[field_name] = round(confidence, 4)
            all_scores[field_name] = {
                candidate: round(score, 4)
                for candidate, score in scores.items()
            }
        return predictions, confidences, all_scores

    def analyze(self, query: str, *, style_override: str = "auto") -> QueryAnalysis:
        if style_override not in VALID_STYLES:
            raise ValueError(f"Unsupported style: {style_override}")

        normalized = self._normalize_query(query)
        if not normalized:
            raise ValueError("Query must not be empty")

        recommendation_topic = extract_recommendation_topic(normalized)
        if recommendation_topic:
            resolved_style = (
                style_override if style_override != "auto" else "concise"
            )
            return QueryAnalysis(
                intent="paper_recommendation",
                emotion="neutral",
                topic_expertise="intermediate",
                verbosity="concise",
                style=resolved_style,
                confidence=1.0,
                cosine_confidence=1.0,
                style_source=(
                    "explicit_override"
                    if style_override != "auto"
                    else "pattern_match"
                ),
                field_confidences={
                    "intent": 1.0,
                    "emotion": 1.0,
                    "topic_expertise": 1.0,
                    "verbosity": 1.0,
                "style": 1.0,
                },
                field_sources={
                    "intent": "pattern_match",
                    "emotion": "pattern_match",
                    "topic_expertise": "pattern_match",
                    "verbosity": "pattern_match",
                    "style": (
                        "explicit_override"
                        if style_override != "auto"
                        else "pattern_match"
                    ),
                },
                embedding_model=self.encoder.model_name,
            )

        if is_direct_chit_chat_request(normalized):
            resolved_style = (
                style_override if style_override != "auto" else "concise"
            )
            return QueryAnalysis(
                intent="chit_chat",
                emotion="positive",
                topic_expertise="intermediate",
                verbosity="concise",
                style=resolved_style,
                confidence=1.0,
                cosine_confidence=1.0,
                style_source=(
                    "explicit_override"
                    if style_override != "auto"
                    else "pattern_match"
                ),
                field_confidences={
                    "intent": 1.0,
                    "emotion": 1.0,
                    "topic_expertise": 1.0,
                    "verbosity": 1.0,
                    "style": 1.0,
                },
                field_sources={
                    "intent": "pattern_match",
                    "emotion": "pattern_match",
                    "topic_expertise": "pattern_match",
                    "verbosity": "pattern_match",
                    "style": (
                        "explicit_override"
                        if style_override != "auto"
                        else "pattern_match"
                    ),
                },
                embedding_model=self.encoder.model_name,
            )

        predictions, confidences, similarity_scores = self._cosine_predictions(
            normalized
        )
        cosine_confidence = min(confidences.values())
        sources = {field_name: "cosine_similarity" for field_name in predictions}
        fallback_fields = [
            field_name
            for field_name in predictions
            if not (field_name == "style" and style_override != "auto")
            and confidences[field_name] < self.confidence_threshold
        ]
        fallback_predictions = (
            self.fallback.classify_many(
                normalized,
                {
                    field_name: _PROTOTYPES[field_name]
                    for field_name in fallback_fields
                },
            )
            if fallback_fields
            else {}
        )
        for field_name in fallback_fields:
            if field_name not in fallback_predictions:
                raise QueryModelError(
                    f"Fallback did not return a result for {field_name}"
                )
            label, confidence = fallback_predictions[field_name]
            if label not in _PROTOTYPES[field_name]:
                raise QueryModelError(
                    f"Fallback returned an unknown {field_name} label: {label!r}"
                )
            predictions[field_name] = label
            confidences[field_name] = round(confidence, 4)
            sources[field_name] = "tinybert_fallback"

        if style_override != "auto":
            predictions["style"] = style_override
            confidences["style"] = 1.0
            sources["style"] = "explicit_override"
        elif (
            confidences["style"] < MIN_AUTO_STYLE_FALLBACK_CONFIDENCE
            and predictions["verbosity"] == "concise"
        ):
            predictions["style"] = "concise"
            sources["style"] = "verbosity_fallback"

        overall_confidence = min(confidences.values())
        fallback_used = bool(fallback_fields)
        fallback_model = self.fallback.model_name if fallback_used else None
        return QueryAnalysis(
            intent=predictions["intent"],
            emotion=predictions["emotion"],
            topic_expertise=predictions["topic_expertise"],
            verbosity=predictions["verbosity"],
            style=predictions["style"],
            confidence=round(overall_confidence, 4),
            cosine_confidence=round(cosine_confidence, 4),
            style_source=sources["style"],
            style_scores=similarity_scores["style"],
            field_confidences=confidences,
            field_sources=sources,
            similarity_scores=similarity_scores,
            fallback_used=fallback_used,
            fallback_fields=tuple(fallback_fields),
            embedding_model=self.encoder.model_name,
            fallback_model=fallback_model,
        )


DEFAULT_QUERY_ANALYZER = SemanticQueryAnalyzer()


def analyze_query(query: str, *, style_override: str = "auto") -> QueryAnalysis:
    return DEFAULT_QUERY_ANALYZER.analyze(query, style_override=style_override)
