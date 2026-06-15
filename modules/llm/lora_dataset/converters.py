"""Convert Hugging Face and local exports to adapter instruction JSONL."""

from __future__ import annotations

import ast
import random
from typing import Any, Iterable

from lora_dataset.io import chat_record, truncate


def sample_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return list(records)
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[index] for index in indices[:limit]]


def shuffle_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _answer_text_from_qasper_answer(answer: dict[str, Any]) -> str:
    if answer.get("unanswerable"):
        return ""
    if answer.get("free_form_answer"):
        return str(answer["free_form_answer"]).strip()
    spans = answer.get("extractive_spans") or []
    if spans:
        return "; ".join(str(span) for span in spans[:3])
    if answer.get("yes_no") is not None:
        return "yes" if answer.get("yes_no") else "no"
    return ""


def convert_qasper_paper_row(row: dict[str, Any], paper_index: int) -> list[dict[str, Any]]:
    title = str(row.get("title") or row.get("id") or "Untitled")
    full_text = row.get("full_text") or {}
    paragraphs: list[str] = []
    if isinstance(full_text, dict):
        for section in full_text.values():
            if isinstance(section, dict):
                for paragraph in section.values():
                    if isinstance(paragraph, str) and paragraph.strip():
                        paragraphs.append(paragraph.strip())
    context = truncate(" ".join(paragraphs[:8]) or str(row.get("abstract") or ""), 3500)
    qas = row.get("qas") or {}
    records: list[dict[str, Any]] = []

    questions = qas.get("question") or []
    question_ids = qas.get("question_id") or [str(i) for i in range(len(questions))]
    answers_list = qas.get("answers") or []

    for q_index, question in enumerate(questions):
        if not question:
            continue
        qid = question_ids[q_index] if q_index < len(question_ids) else str(q_index)
        answer_text = ""
        if q_index < len(answers_list) and isinstance(answers_list[q_index], dict):
            for annot in answers_list[q_index].get("answer") or []:
                if isinstance(annot, dict):
                    answer_text = _answer_text_from_qasper_answer(annot)
                    if answer_text:
                        break
        if not answer_text:
            continue
        user_content = "\n\n".join(
            [
                "Task: evidence-grounded QA over a scientific paper.",
                f"Title: {title}",
                "Context:",
                context,
                f"Question: {question}",
                "Answer using only the context. Cite [S1] when referencing the paper body.",
            ]
        )
        records.append(
            chat_record(
                user_content=user_content,
                assistant_content=f"{answer_text} [S1]",
                source="qasper",
                task="evidence_to_answer",
                prompt_id=f"qasper_{paper_index}_{qid}",
                license_note="allenai/qasper HF export.",
            )
        )
    return records


def convert_qasper_rows(
    dataset: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for paper_index, row in enumerate(dataset):
        records.extend(convert_qasper_paper_row(row, paper_index))
    return sample_records(records, limit, seed)


def _parse_list_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        return _parse_list_field(value[0])
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list) and parsed:
                    return str(parsed[0]).strip()
            except (SyntaxError, ValueError):
                pass
        return text
    return str(value).strip()


def convert_scitldr_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    source = _parse_list_field(row.get("source")) or _parse_list_field(row.get("text"))
    summary = _parse_list_field(row.get("target")) or _parse_list_field(row.get("summary"))
    if not source or not summary:
        return None
    return chat_record(
        user_content="\n\n".join(
            [
                "Task: write a concise scientific summary.",
                "Paper text:",
                truncate(source, 3000),
            ]
        ),
        assistant_content=truncate(summary, 1200),
        source="scitldr",
        task="paper_to_summary",
        prompt_id=f"scitldr_{index}",
        license_note="allenai/scitldr Abstract config.",
    )


def convert_scitldr_rows(
    dataset: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        converted = convert_scitldr_row(row, index)
        if converted:
            records.append(converted)
    return sample_records(records, limit, seed)


def convert_scicite_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    citation = truncate(str(row.get("string") or row.get("citation") or ""), 1200)
    if not citation:
        return None
    label = str(row.get("label") or row.get("label2") or row.get("cite_end") or "background")
    return chat_record(
        user_content="\n\n".join(
            [
                "Task: describe citation intent for the citing sentence.",
                f"Citing context: {citation}",
            ]
        ),
        assistant_content=f"Citation intent: {label}.",
        source="scicite",
        task="citation_intent",
        prompt_id=f"scicite_{index}",
        license_note="allenai/scicite extended config.",
    )


def convert_scicite_rows(
    dataset: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        converted = convert_scicite_row(row, index)
        if converted:
            records.append(converted)
    return sample_records(records, limit, seed)


def _peer_review_from_abstract(title: str, abstract: str) -> str:
    topic = truncate(abstract, 200)
    return "\n".join(
        [
            "Strengths: The abstract states a clear problem and approach.",
            f"Evidence: {topic}",
            "Weaknesses: Full methodology and evaluation details are not available from the abstract alone.",
            "Missing evidence: quantitative results, baselines, and reproducibility notes.",
            "Suggested improvements: add explicit metrics and comparison to prior work in the main text.",
        ]
    )


def convert_peerread_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    if not abstract:
        return None
    reviews = row.get("reviews") or {}
    comment_parts: list[str] = []
    if isinstance(reviews, dict):
        for key in ("comments", "recommendation", "substance", "clarity", "originality"):
            for value in reviews.get(key) or []:
                text = str(value).strip()
                if len(text) > 20:
                    comment_parts.append(f"{key}: {text}")
    if comment_parts:
        assistant = "\n".join(comment_parts[:6])
        license_note = "allenai/peer_read reviews config with review text."
    else:
        assistant = _peer_review_from_abstract(title, abstract)
        license_note = (
            "allenai/peer_read HF row; review comments empty on Hub — "
            "abstract-only structured critique template."
        )
    return chat_record(
        user_content="\n\n".join(
            [
                "Task: write peer-review style feedback for the paper.",
                f"Title: {title}",
                "Paper abstract:",
                truncate(abstract, 2500),
            ]
        ),
        assistant_content=assistant,
        source="peerread",
        task="peer_review_critique",
        prompt_id=f"peerread_{index}",
        license_note=license_note,
    )


def convert_peerread_rows(
    dataset: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        converted = convert_peerread_row(row, index)
        if converted:
            records.append(converted)
    return sample_records(records, limit, seed)


def convert_researchqa_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    question_type = str(row.get("question_type") or "")
    if question_type == "adversarial":
        expected_refusal = row.get("expected_refusal")
        if expected_refusal is True:
            return chat_record(
                user_content="\n\n".join(
                    [
                        "Task: answer using only supplied paper evidence.",
                        f"Question: {row.get('question', '')}",
                        "Evidence: none sufficient for the false premise.",
                    ]
                ),
                assistant_content=(
                    "The question contains a false premise or is not supported by the supplied evidence. "
                    "I cannot provide a confident answer without inventing details."
                ),
                source="researchqa",
                task="refusal",
                prompt_id=f"RQ_refusal_{index}",
                license_note="khoj-ai/ResearchQA adversarial row.",
            )

    passages: list[tuple[str, str]] = []
    refs = row.get("expected_references") or []
    if isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            alternatives = ref.get("alternatives") or []
            if alternatives:
                passages.append((f"S{len(passages) + 1}", truncate(str(alternatives[0]), 900)))
    if not passages and row.get("metadata_source_text"):
        passages.append(("S1", truncate(str(row["metadata_source_text"]), 900)))
    if not passages:
        return None

    question = str(row.get("question") or "").strip()
    answer = str(row.get("expected_answer") or "").strip()
    if not question or not answer:
        return None

    evidence_block = "\n".join(f"[{sid}] {text}" for sid, text in passages)
    assistant = answer
    if not any(sid in answer for sid, _ in passages):
        assistant = f"{answer} [{passages[0][0]}]"

    return chat_record(
        user_content="\n\n".join(
            [
                "Task: evidence-grounded scientific QA over supplied paper passages.",
                f"Question: {question}",
                f"Paper id: {row.get('paper_id', '')}",
                "Evidence passages:",
                evidence_block,
                "Cite source IDs for substantive claims.",
            ]
        ),
        assistant_content=assistant,
        source="researchqa",
        task=question_type or "evidence_to_answer",
        prompt_id=f"RQ_{index}",
        license_note="khoj-ai/ResearchQA test split.",
    )


def convert_researchqa_rows(
    dataset: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        converted = convert_researchqa_row(row, index)
        if converted:
            records.append(converted)
    return sample_records(records, limit, seed)

