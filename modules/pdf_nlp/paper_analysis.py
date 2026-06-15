"""Local NLP enrichment for canonical ``ParsedPaper`` records.

Runs spaCy POS tagging, fine-tuned SciER NER, KeyBERT keyphrases, TextRank
summaries, and rule-based structural checks. Models load lazily from the
checksum-validated runtime directory under ``modules/pdf_nlp/models``.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from functools import lru_cache
from typing import Any, Callable

from model_assets import MODEL_ROOT, require_model_assets

CORE_SECTION_ORDER = (
    "abstract",
    "introduction",
    "method",
    "results",
    "conclusion",
)
OPTIONAL_SECTION_ORDER = ("related_work", "limitations", "discussion")
SCIER_THRESHOLD = 0.7
METRIC_PATTERN = re.compile(
    r"\b(?:accuracy|precision|recall|f1(?:-score)?|bleu|rouge(?:-[12l])?|"
    r"perplexity|mrr|map|ndcg(?:@\d+)?|auc|exact match|treesim|"
    r"episode return|sample efficiency|wall-clock time)\b",
    flags=re.IGNORECASE,
)
GAZETTEER: dict[str, tuple[str, ...]] = {
    "Method": (
        "Transformer",
        "BERT",
        "RAG",
        "retrieval-augmented generation",
        "reinforcement learning",
        "data augmentation",
        "actor-critic",
        "seq2seq",
        "neural retriever",
        "fine-tuning",
        "self-attention",
    ),
    "Dataset": (
        "SQuAD",
        "GLUE",
        "MultiNLI",
        "WMT 2014",
        "ImageNet",
        "Wikipedia",
        "DeepMind Control Suite",
        "OpenFOAM",
        "LAMMPS",
        "GEOS",
    ),
    "Task": (
        "machine translation",
        "question answering",
        "visual continuous control",
        "language generation",
        "scientific simulation",
        "open domain QA",
        "constituency parsing",
    ),
}


def _import_dependency(module: str, setup_hint: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise RuntimeError(
            f"Missing PDF-NLP dependency {module!r}. {setup_hint}"
        ) from exc


ROOT_REQUIREMENTS_HINT = (
    "From the repository root run: pip install -r requirements.txt."
)


@lru_cache(maxsize=1)
def spacy_model():
    """Load the archive-provided spaCy English model."""
    require_model_assets()
    spacy = _import_dependency(
        "spacy",
        ROOT_REQUIREMENTS_HINT,
    )
    model_path = MODEL_ROOT / "spacy" / "en_core_web_sm"
    nlp = spacy.load(model_path)
    nlp.max_length = 3_000_000
    return nlp


@lru_cache(maxsize=1)
def scier_pipeline():
    """Load the local fine-tuned SciER DistilBERT pipeline."""
    require_model_assets()
    transformers = _import_dependency(
        "transformers",
        ROOT_REQUIREMENTS_HINT,
    )
    model_path = MODEL_ROOT / "scier-distilbert-final"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )
    # The retained tokenizer advertises token_type_ids, which DistilBERT rejects.
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    return transformers.pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=-1,
    )


@lru_cache(maxsize=1)
def keybert_model():
    """Load KeyBERT with the archive-provided local MiniLM encoder."""
    require_model_assets()
    keybert = _import_dependency(
        "keybert",
        ROOT_REQUIREMENTS_HINT,
    )
    sentence_transformers = _import_dependency(
        "sentence_transformers",
        ROOT_REQUIREMENTS_HINT,
    )
    encoder = sentence_transformers.SentenceTransformer(
        str(MODEL_ROOT / "all-MiniLM-L6-v2"),
        local_files_only=True,
    )
    return keybert.KeyBERT(model=encoder)


def section_texts(paper: dict[str, Any]) -> list[tuple[str, str]]:
    """Return non-empty core and optional sections in stable order."""
    sections = paper.get("sections") or {}
    return [
        (name, str(sections.get(name) or "").strip())
        for name in (*CORE_SECTION_ORDER, *OPTIONAL_SECTION_ORDER)
        if str(sections.get(name) or "").strip()
    ]


def _measure(
    timings: dict[str, float],
    name: str,
    function: Callable[..., Any],
    *args,
    event_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
    **kwargs,
):
    if event_callback:
        event_callback(name, "started", {})
    started = time.perf_counter()
    try:
        value = function(*args, **kwargs)
    except Exception as exc:
        elapsed = round(time.perf_counter() - started, 6)
        timings[name] = elapsed
        if event_callback:
            event_callback(name, "failed", {"duration_ms": elapsed * 1000, "error": str(exc)})
        raise
    timings[name] = round(time.perf_counter() - started, 6)
    if event_callback:
        event_callback(
            name,
            "completed",
            {"duration_ms": timings[name] * 1000},
        )
    return value


def pos_analysis(paper: dict[str, Any], *, token_limit: int = 250) -> dict[str, Any]:
    """Create compact POS, dependency, lemma, and noun-chunk evidence."""
    nlp = spacy_model()
    tokens: list[dict[str, Any]] = []
    noun_chunks: list[dict[str, Any]] = []
    pos_counts: Counter[str] = Counter()
    total_tokens = 0
    for section, text in section_texts(paper):
        doc = nlp(text)
        for token in doc:
            if token.is_space:
                continue
            total_tokens += 1
            pos_counts[token.pos_] += 1
            if len(tokens) < token_limit:
                tokens.append(
                    {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "dependency": token.dep_,
                        "section": section,
                        "start": token.idx,
                        "end": token.idx + len(token.text),
                    }
                )
        for chunk in doc.noun_chunks:
            if len(noun_chunks) >= 150:
                break
            noun_chunks.append(
                {
                    "text": chunk.text,
                    "section": section,
                    "start": chunk.start_char,
                    "end": chunk.end_char,
                }
            )
    return {
        "token_count": total_tokens,
        "displayed_token_count": len(tokens),
        "pos_counts": dict(sorted(pos_counts.items())),
        "tokens": tokens,
        "noun_chunks": noun_chunks,
    }


def _sentence_rows(paper: dict[str, Any]) -> list[dict[str, Any]]:
    nlp = spacy_model()
    rows: list[dict[str, Any]] = []
    for section, text in section_texts(paper):
        doc = nlp(text)
        for sentence in doc.sents:
            value = sentence.text.strip()
            if value:
                rows.append(
                    {
                        "section": section,
                        "text": value,
                        "start": sentence.start_char,
                    }
                )
    return rows


def entity_analysis(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Run sentence-batched SciER NER plus metric and institution completion."""
    rows = _sentence_rows(paper)
    entities: list[dict[str, Any]] = []
    pipeline = scier_pipeline()
    sentence_texts = [row["text"] for row in rows]
    if sentence_texts:
        predictions = pipeline(
            sentence_texts,
            batch_size=8,
        )
        for sentence, sentence_predictions in zip(rows, predictions, strict=True):
            for prediction in sentence_predictions:
                label = str(prediction.get("entity_group") or "")
                score = float(prediction.get("score") or 0.0)
                if label in {"O", "LABEL_0"} or score < SCIER_THRESHOLD:
                    continue
                entities.append(
                    {
                        "text": str(prediction.get("word") or "").strip(),
                        "type": label,
                        "score": round(score, 6),
                        "source": "scier_distilbert",
                        "section": sentence["section"],
                        "start": sentence["start"] + int(prediction.get("start") or 0),
                        "end": sentence["start"] + int(prediction.get("end") or 0),
                    }
                )

    nlp = spacy_model()
    for section, text in section_texts(paper):
        for entity_type, terms in GAZETTEER.items():
            for term in sorted(terms, key=len, reverse=True):
                for match in re.finditer(
                    rf"(?<!\w){re.escape(term)}(?!\w)",
                    text,
                    flags=re.IGNORECASE,
                ):
                    entities.append(
                        {
                            "text": match.group(0),
                            "type": entity_type,
                            "score": 1.0,
                            "source": "baseline_gazetteer",
                            "section": section,
                            "start": match.start(),
                            "end": match.end(),
                        }
                    )
        for match in METRIC_PATTERN.finditer(text):
            entities.append(
                {
                    "text": match.group(0),
                    "type": "Metric",
                    "score": 1.0,
                    "source": "metric_regex",
                    "section": section,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        for entity in nlp(text).ents:
            if entity.label_ != "ORG":
                continue
            entities.append(
                {
                    "text": entity.text,
                    "type": "Institution",
                    "score": 1.0,
                    "source": "spacy_org",
                    "section": section,
                    "start": entity.start_char,
                    "end": entity.end_char,
                }
            )

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entity in entities:
        key = (
            entity["type"].lower(),
            entity["section"],
            re.sub(r"\s+", " ", entity["text"].lower()).strip(),
        )
        current = unique.get(key)
        if current is None or entity["score"] > current["score"]:
            unique[key] = entity
    return sorted(
        unique.values(),
        key=lambda row: (row["section"], row["start"], row["type"], row["text"]),
    )


def keyphrase_analysis(
    paper: dict[str, Any],
    *,
    per_section: int = 5,
    total_limit: int = 20,
) -> list[dict[str, Any]]:
    """Extract KeyBERT phrases independently per section and deduplicate them."""
    model = keybert_model()
    candidates: list[dict[str, Any]] = []
    for section, text in section_texts(paper):
        rows = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=per_section,
        )
        candidates.extend(
            {
                "text": phrase,
                "score": round(float(score), 6),
                "section": section,
                "source": "keybert_minilm",
            }
            for phrase, score in rows
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in candidates:
        key = re.sub(r"[\W_]+", " ", row["text"].lower()).strip()
        current = deduped.get(key)
        if current is None or row["score"] > current["score"]:
            deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda row: (-row["score"], row["text"].lower()),
    )[:total_limit]


def extractive_summary(
    paper: dict[str, Any],
    *,
    sentence_count: int = 5,
) -> dict[str, Any]:
    """Build a deterministic TextRank summary from source-identical sentences."""
    numpy = _import_dependency("numpy", ROOT_REQUIREMENTS_HINT)
    networkx = _import_dependency("networkx", ROOT_REQUIREMENTS_HINT)
    sklearn = _import_dependency("sklearn", ROOT_REQUIREMENTS_HINT)
    del sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = [
        row
        for row in _sentence_rows(paper)
        if 35 <= len(row["text"]) <= 800
    ]
    # Bound the dense similarity matrix while retaining coverage from every section.
    candidates = candidates[:300]
    if len(candidates) <= sentence_count:
        selected = candidates
    else:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(
            [row["text"] for row in candidates]
        )
        similarity = cosine_similarity(matrix)
        numpy.fill_diagonal(similarity, 0.0)
        graph = networkx.from_numpy_array(similarity)
        scores = networkx.pagerank(graph, weight="weight")
        indices = sorted(
            sorted(scores, key=scores.get, reverse=True)[:sentence_count]
        )
        selected = [candidates[index] for index in indices]
    sentences = [
        {
            "text": row["text"],
            "section": row["section"],
            "source": "extractive_textrank",
        }
        for row in selected
    ]
    return {
        "text": " ".join(row["text"] for row in selected),
        "sentences": sentences,
        "candidate_sentence_count": len(candidates),
        "source_traceable": True,
    }


def structural_checks(paper: dict[str, Any]) -> list[dict[str, Any]]:
    """Return deterministic paper-structure findings with evidence snippets."""
    sections = paper.get("sections") or {}
    references = paper.get("references") or []
    joined = "\n".join(text for _, text in section_texts(paper))
    checks: list[dict[str, Any]] = []

    def add(code: str, severity: str, message: str, evidence: str = "") -> None:
        checks.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "evidence": re.sub(r"\s+", " ", evidence).strip()[:300],
            }
        )

    if len(str(sections.get("abstract") or "")) < 200:
        add("missing_or_weak_abstract", "high", "Abstract is missing or very short.")
    if len(str(sections.get("method") or "")) < 400:
        add("missing_or_weak_method", "high", "Method section is missing or very short.")
    if len(str(sections.get("results") or "")) < 300:
        add(
            "missing_or_weak_results",
            "high",
            "Experiment or results evidence is missing or very short.",
        )
    if len(str(sections.get("conclusion") or "")) < 150:
        add("missing_conclusion", "medium", "No substantial conclusion was detected.")
    if len(references) < 10:
        add(
            "short_reference_list",
            "medium",
            "Fewer than ten parsed references were detected.",
            f"parsed references: {len(references)}",
        )
    if not METRIC_PATTERN.search(joined):
        add("no_clear_metric", "medium", "No common evaluation metric was detected.")
    if not str(sections.get("limitations") or "").strip():
        add(
            "no_explicit_limitations",
            "low",
            "No explicit limitations section was detected.",
        )
    if not str(sections.get("related_work") or "").strip():
        add(
            "weak_related_work_signal",
            "low",
            "No explicit related-work section was detected.",
        )
    return checks


def _canonical_entities(
    mentions: list[dict[str, Any]],
) -> dict[str, list[str]]:
    mapping = {
        "method": "methods",
        "dataset": "datasets",
        "task": "tasks",
        "metric": "metrics",
        "institution": "institutions",
    }
    output = {value: [] for value in mapping.values()}
    for mention in mentions:
        target = mapping.get(str(mention["type"]).lower())
        text = str(mention["text"]).strip()
        if target and text and text not in output[target]:
            output[target].append(text)
    return output


def analyze_parsed_paper(
    paper: dict[str, Any],
    *,
    event_callback: Callable[[str, str, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Enrich one canonical ParsedPaper and return its sidecar analysis."""
    require_model_assets()
    timings: dict[str, float] = {}
    pos = _measure(
        timings,
        "pos",
        pos_analysis,
        paper,
        event_callback=event_callback,
    )
    mentions = _measure(
        timings,
        "ner",
        entity_analysis,
        paper,
        event_callback=event_callback,
    )
    keyphrases = _measure(
        timings,
        "keyphrases",
        keyphrase_analysis,
        paper,
        event_callback=event_callback,
    )
    summary = _measure(
        timings,
        "extractive_summary",
        extractive_summary,
        paper,
        event_callback=event_callback,
    )
    checks = _measure(
        timings,
        "structural_checks",
        structural_checks,
        paper,
        event_callback=event_callback,
    )
    analysis = {
        "pos": pos,
        "entity_mentions": mentions,
        "keyphrases": keyphrases,
        "extractive_summary": summary,
        "structural_checks": checks,
        "timings_seconds": timings,
        "provenance": {
            "pos": "spaCy en_core_web_sm from team runtime archive",
            "ner": (
                "Fine-tuned SciER DistilBERT at threshold "
                f"{SCIER_THRESHOLD}, plus regex/spaCy completion"
            ),
            "keyphrases": "KeyBERT with local all-MiniLM-L6-v2",
            "summary": "deterministic extractive TextRank",
            "structural_checks": "deterministic project rules",
            "bart_status": "historical comparison only; not production",
        },
    }
    enriched = dict(paper)
    enriched["keywords"] = [row["text"] for row in keyphrases]
    enriched["entities"] = _canonical_entities(mentions)
    enriched["analysis"] = analysis
    return enriched, analysis
