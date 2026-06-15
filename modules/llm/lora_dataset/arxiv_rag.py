"""Build project arXiv RAG LoRA instructions from the sampled corpus and BM25 packs."""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

from app.prompt_library import SYSTEM_GUARDRAILS, build_prompt
from app.schemas import validate_paper_record

from lora_dataset.arxiv_rag_labels import build_assistant_label
from lora_dataset.arxiv_rag_retrieval import build_rag_evidence_pack
from lora_dataset.io import prompt_record_to_chat, read_jsonl, write_jsonl
from lora_dataset.paths import (
    CORPUS_JSONL,
    RAG_JSONL,
    RAG_NUM_QUERIES,
    RAG_PACKS_DIR,
    RAG_QUERIES_JSONL,
    SEED,
    MAX_TRAINING_PROMPT_CHARS,
)
from app.runtime import prompt_text_for_record

RAG_TASKS = [
    ("topic_search_synthesis", "technical"),
    ("citation_recommendation", "technical"),
    ("research_gap_identification", "technical"),
    ("explain_recommendation", "technical"),
    ("follow_up_qa", "technical"),
    ("beginner_explanation", "beginner"),
]

EVAL_QUERY_BLOCKLIST = {"retrieval augmented generation for scientific papers"}
APS_INSTRUCTION = (
    "For external-paper claims, retain source grounding such as [S1] and add the "
    "matching APS-style numbered citation such as [1]. Include a numbered References "
    "section for Markdown answers. Do not invent missing bibliographic fields."
)


def _load_papers(path: Path) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        validate_paper_record(row)
        if str(row.get("paper_id", "")).startswith("sample_"):
            raise SystemExit(
                f"Refusing synthetic corpus at {path}. Rebuild from Kaggle — see lora_dataset/README.md"
            )
        papers.append(row)
    if not papers:
        raise SystemExit(f"No papers in {path}")
    return papers


def _sample_queries(papers: list[dict[str, Any]], *, count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    queries: list[str] = []
    for paper in papers:
        title = (paper.get("title") or "").strip()
        if title and title.lower() not in EVAL_QUERY_BLOCKLIST:
            queries.append(title)
        abstract = (paper.get("abstract") or "").strip()
        words = abstract.split()
        if len(words) >= 8:
            queries.append(" ".join(words[:12]))
    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        key = query.lower()
        if key in seen or key in EVAL_QUERY_BLOCKLIST:
            continue
        seen.add(key)
        unique.append(query)
    rng.shuffle(unique)
    return unique[:count]


def _make_prompt_record(*, prompt_id: str, task: str, style: str, pack: dict[str, Any]) -> dict[str, Any]:
    if task == "explain_recommendation":
        template = (
            "Task: explain why the top recommended paper in the RagEvidencePack is relevant.\n"
            "Use only Recommendation metadata and evidence snippets. Cite source IDs."
        )
        few_shot = False
    else:
        template = None
        few_shot = task in {"citation_recommendation", "peer_review_critique", "research_gap_identification"}

    base_prompt = build_prompt(task, style=style, few_shot=few_shot) if template is None else "\n\n".join(
        [SYSTEM_GUARDRAILS, f"Style: {style}.", template]
    )
    base_prompt = f"{base_prompt}\n\n{APS_INSTRUCTION}"
    zero_shot = base_prompt if not few_shot else build_prompt(task, style=style, few_shot=False)
    if few_shot:
        zero_shot = f"{zero_shot}\n\n{APS_INSTRUCTION}"
    few_shot_prompt = base_prompt if few_shot else zero_shot

    return {
        "prompt_id": prompt_id,
        "task": task,
        "style": style,
        "system_guardrails": SYSTEM_GUARDRAILS,
        "zero_shot_prompt": zero_shot,
        "few_shot_prompt": few_shot_prompt,
        "input": {"parsed_paper": None, "rag_evidence_pack": pack},
        "expected_output_contract": {
            "must_include_source_ids": True,
            "must_separate_findings_from_assumptions": True,
            "must_not_fabricate_metadata": True,
        },
    }


def _compact_pack(pack: dict[str, Any]) -> dict[str, Any]:
    compact = deepcopy(pack)
    for candidate in compact["candidates"]:
        paper = candidate["paper"]
        paper["abstract"] = ""
        paper["authors"] = paper.get("authors", [])[:3]
        paper["categories"] = paper.get("categories", [])[:3]
        candidate["reason"] = "BM25 title/abstract match for the supplied query."
    for snippet in compact["evidence_snippets"]:
        snippet["metadata"]["authors"] = snippet["metadata"].get("authors", [])[:3]
    return compact


def build_rag_instructions(
    *,
    corpus_path: Path = CORPUS_JSONL,
    output_path: Path = RAG_JSONL,
    num_queries: int = RAG_NUM_QUERIES,
    seed: int = SEED,
    save_packs: bool = True,
) -> int:
    """Sample queries, build packs, and write project arXiv RAG instruction rows."""
    if not corpus_path.is_file():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    papers = _load_papers(corpus_path)
    queries = _sample_queries(papers, count=num_queries, seed=seed)
    write_jsonl(RAG_QUERIES_JSONL, [{"query_id": f"Q{i:04d}", "query": q} for i, q in enumerate(queries)])

    if save_packs:
        RAG_PACKS_DIR.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        pack = build_rag_evidence_pack(query, papers, top_k=5)
        if save_packs:
            pack_path = RAG_PACKS_DIR / f"Q{query_index:04d}.json"
            pack_path.write_text(json.dumps(pack, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

        for task, style in RAG_TASKS:
            prompt_id = f"ARXIV_RAG_{query_index:04d}_{task}"
            assistant = build_assistant_label(pack, task)
            if not assistant:
                continue
            prompt_record = _make_prompt_record(
                prompt_id=prompt_id,
                task=task,
                style=style,
                pack=_compact_pack(pack),
            )
            prompt_chars = len(prompt_text_for_record(prompt_record, "few_shot"))
            if prompt_chars > MAX_TRAINING_PROMPT_CHARS:
                raise ValueError(
                    f"Training prompt {prompt_id} is {prompt_chars} chars; "
                    f"limit is {MAX_TRAINING_PROMPT_CHARS}"
                )
            examples.append(
                prompt_record_to_chat(
                    prompt_record,
                    assistant,
                    source="project_arxiv_rag",
                    prompt_id=prompt_id,
                    license_note="BM25 RAG over Kaggle PaperRecord corpus; template assistant label.",
                )
            )

    write_jsonl(output_path, examples)
    print(f"Wrote {len(queries)} queries and {len(examples)} RAG instruction rows")
    return len(examples)
