"""Build project-specific paper-only LoRA rows from cached open data."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from app.prompt_library import PAPER_ONLY_GUARDRAILS, paper_summary_prompt_record
from app.query_understanding import analyze_query

from lora_dataset.io import prompt_record_to_chat, read_jsonl, write_jsonl
from lora_dataset.paths import HYBRID_JSONL, PAPER_ONLY_JSONL, PAPER_ONLY_PER_TASK, SEED


def _extract_between(text: str, start: str, end: str | None = None) -> str:
    after = text.split(start, 1)[1] if start in text else ""
    if end and end in after:
        after = after.split(end, 1)[0]
    return after.strip()


def _paper_id(prompt_id: str) -> str:
    digest = hashlib.sha256(prompt_id.encode("utf-8")).hexdigest()[:12]
    return f"training_{digest}"


def _parsed_paper(prompt_id: str, *, title: str, abstract: str) -> dict[str, Any]:
    return {
        "metadata": {
            "paper_id": _paper_id(prompt_id),
            "title": title or "Untitled training paper",
            "abstract": abstract,
            "authors": [],
            "categories": [],
            "published_date": None,
            "venue": None,
            "doi": None,
            "arxiv_id": None,
            "url": None,
            "source": "open_academic_training",
        },
        "sections": {
            "abstract": abstract,
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


def _custom_record(
    *,
    prompt_id: str,
    task: str,
    style: str,
    paper: dict[str, Any],
    user_query: str,
    instruction: str,
) -> dict[str, Any]:
    analysis = analyze_query(user_query, style_override=style)
    prompt = "\n\n".join([PAPER_ONLY_GUARDRAILS, f"Style: {style}.", instruction])
    return {
        "prompt_id": prompt_id,
        "task": task,
        "style": style,
        "system_guardrails": PAPER_ONLY_GUARDRAILS,
        "zero_shot_prompt": prompt,
        "few_shot_prompt": prompt,
        "input": {
            "parsed_paper": paper,
            "rag_evidence_pack": None,
            "user_query": user_query,
        },
        "query_analysis": analysis.as_dict(),
        "expected_output_contract": {
            "input_contract": "parsed_paper_only",
            "must_include_source_ids": False,
            "must_not_use_external_evidence": True,
            "must_not_fabricate_metadata": True,
        },
    }


def _summary_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_rows = sorted(
        (row for row in rows if row.get("source") == "scitldr"),
        key=lambda row: str(row.get("prompt_id")),
    )
    for index, row in enumerate(source_rows[:limit]):
        source = _extract_between(row["messages"][1]["content"], "Paper text:")
        paper = _parsed_paper(row["prompt_id"], title="Scientific paper", abstract=source)
        query = "Summarize the supplied paper."
        analysis = analyze_query(query, style_override="technical")
        record = paper_summary_prompt_record(
            paper,
            "technical",
            user_query=query,
            query_analysis=analysis.as_dict(),
        )
        record["prompt_id"] = f"TRAIN_PAPER_SUMMARY_{index:03d}"
        target = "\n".join(
            [
                "## Scope",
                source,
                "",
                "## Core Contribution",
                row["messages"][2]["content"].strip(),
                "",
                "## Method",
                "Not stated in the supplied ParsedPaper.",
                "",
                "## Results",
                "Not stated in the supplied ParsedPaper.",
                "",
                "## Limitations",
                "The supplied ParsedPaper contains only brief source text.",
            ]
        )
        output.append(
            prompt_record_to_chat(
                record,
                target,
                source="project_paper_only",
                license_note=f"Derived from {row['license_note']}",
            )
        )
    return output


def _qa_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_rows = sorted(
        (row for row in rows if row.get("source") == "qasper"),
        key=lambda row: str(row.get("prompt_id")),
    )
    for index, row in enumerate(source_rows[:limit]):
        user = row["messages"][1]["content"]
        title = _extract_between(user, "Title:", "Context:")
        context = _extract_between(user, "Context:", "Question:")
        question = _extract_between(user, "Question:", "Answer using only")
        paper = _parsed_paper(row["prompt_id"], title=title, abstract=context)
        record = _custom_record(
            prompt_id=f"TRAIN_PAPER_QA_{index:03d}",
            task="paper_question_answering",
            style="technical",
            paper=paper,
            user_query=question,
            instruction=(
                "Task: answer the user's question using only the supplied ParsedPaper. "
                "State when the paper does not provide enough information."
            ),
        )
        target = re.sub(r"\s*\[S1\]\s*$", "", row["messages"][2]["content"]).strip()
        output.append(
            prompt_record_to_chat(
                record,
                target,
                source="project_paper_only",
                license_note=f"Derived from {row['license_note']}",
            )
        )
    return output


def _peer_review_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    source_rows = sorted(
        (row for row in rows if row.get("source") == "peerread"),
        key=lambda row: str(row.get("prompt_id")),
    )
    for index, row in enumerate(source_rows[:limit]):
        user = row["messages"][1]["content"]
        title = _extract_between(user, "Title:", "Paper abstract:")
        abstract = _extract_between(user, "Paper abstract:")
        query = (
            "Write peer-review style feedback with strengths, weaknesses, "
            "missing evidence, and suggested improvements."
        )
        paper = _parsed_paper(row["prompt_id"], title=title, abstract=abstract)
        record = _custom_record(
            prompt_id=f"TRAIN_PAPER_REVIEW_{index:03d}",
            task="peer_review_critique",
            style="reviewer",
            paper=paper,
            user_query=query,
            instruction=(
                "Task: write reviewer-style strengths, weaknesses, missing evidence, "
                "and suggested improvements using only the supplied ParsedPaper."
            ),
        )
        output.append(
            prompt_record_to_chat(
                record,
                row["messages"][2]["content"].strip(),
                source="project_paper_only",
                license_note=f"Derived from {row['license_note']}",
            )
        )
    return output


def build_paper_only_instructions(
    *,
    input_path=HYBRID_JSONL,
    output_path=PAPER_ONLY_JSONL,
    per_task: int = PAPER_ONLY_PER_TASK,
    seed: int = SEED,
) -> dict[str, int]:
    del seed  # Stable prompt-id ordering is deterministic without random sampling.
    rows = read_jsonl(input_path)
    examples = [
        *_summary_rows(rows, per_task),
        *_qa_rows(rows, per_task),
        *_peer_review_rows(rows, per_task),
    ]
    counts = {
        "uploaded_paper_summary": sum(row["task"] == "uploaded_paper_summary" for row in examples),
        "paper_question_answering": sum(row["task"] == "paper_question_answering" for row in examples),
        "peer_review_critique": sum(row["task"] == "peer_review_critique" for row in examples),
    }
    if any(count != per_task for count in counts.values()):
        raise ValueError(f"Insufficient cached rows for paper-only dataset: {counts}")
    write_jsonl(output_path, examples)
    print(f"Wrote {len(examples)} paper-only rows to {output_path}")
    return counts
