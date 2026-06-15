"""Prompt templates and fixed evaluation prompts for the LLM module."""

from __future__ import annotations

from typing import Any

import json
import re
from pathlib import Path


SYSTEM_GUARDRAILS = (
    "You are a local research-paper assistant. Answer only from ParsedPaper fields, "
    "RagEvidencePack evidence snippets, Recommendation metadata, or explicitly stated "
    "assumptions. Cite source IDs for evidence-backed claims. Do not invent authors, "
    "venues, citations, scores, or findings. If evidence is incomplete, say so."
)

PAPER_ONLY_GUARDRAILS = (
    "You are a local research-paper assistant. Answer only from the supplied ParsedPaper "
    "fields. Do not use model memory as external evidence, retrieve related papers, or "
    "invent authors, citations, results, methods, or limitations. Clearly distinguish "
    "missing information from information stated in the paper. When the optional "
    "ParsedPaper.analysis object is present, treat its extractive summary, entities, "
    "keyphrases, and structural checks as deterministic local evidence."
)

DIRECT_CHAT_GUARDRAILS = (
    "You are a local research-paper assistant responding to a conversational text "
    "message without document retrieval. Reply naturally and directly. Do not claim "
    "that papers, citations, or external evidence were retrieved. Do not invent "
    "paper-specific facts. If the user asks for evidence-dependent research analysis, "
    "state that retrieved evidence is required."
)


PROMPT_TEMPLATES: dict[str, str] = {
    "direct_text_chat": (
        "Task: respond to the user's conversational message. No PDF, ParsedPaper, "
        "RagEvidencePack, recommendation metadata, or external evidence is available."
    ),
    "uploaded_paper_summary": (
        "Task: summarize the uploaded paper for a researcher.\n"
        "Required sections: scope, core contribution, method, results, and limitations.\n"
        "Use optional deterministic PDF-NLP analysis fields to ground terminology and "
        "structural limitations, but do not present them as LLM findings.\n"
        "Input object: ParsedPaper only. No RagEvidencePack, retrieval, related-paper "
        "analysis, or external evidence is available."
    ),
    "paper_question_answer": (
        "Task: answer the user's specific question about the uploaded paper.\n"
        "Lead with the direct answer, then name the relevant supplied section. Do not "
        "replace the requested answer with a generic paper summary. If the ParsedPaper "
        "does not contain enough evidence, say exactly what is missing.\n"
        "Input object: ParsedPaper only. No RagEvidencePack, retrieval, related-paper "
        "analysis, or external evidence is available."
    ),
    "topic_search_synthesis": (
        "Task: synthesize retrieved papers for a topic query.\n"
        "Use only RagEvidencePack candidates and evidence_snippets. Cite source IDs in each claim.\n"
        "Respond with a single JSON object (no markdown fences) with keys:\n"
        '  "summary": string (2-4 sentences),\n'
        '  "key_findings": array of strings (evidence-backed, with source IDs),\n'
        '  "research_gaps": array of strings (gaps or limitations supported by evidence).\n'
        "Write the answer content directly: in \"summary\" describe what the papers say, "
        "not what the user asked. Never restate, paraphrase, or narrate the request (e.g. "
        "do not begin with phrases like 'The user asked'). If the query targets one specific "
        "paper, make that paper the focus of the summary and explain its problem, method, and "
        "findings from its evidence snippet.\n"
        "Do not invent metadata. If evidence is insufficient, state that in summary or gaps."
    ),
    "research_gap_identification": (
        "Task: identify research gaps from the paper and retrieved evidence.\n"
        "Separate evidence-supported gaps from assumptions. Do not claim a gap is proven "
        "unless a source ID supports it."
    ),
    "citation_recommendation": (
        "Task: recommend only citations that are directly relevant to the query, using "
        "RagEvidencePack candidates and evidence snippets.\n"
        "For each recommended paper, copy its supplied APA citation exactly, include its "
        "source ID in square brackets, and give one evidence-grounded sentence explaining "
        "relevance. Do not "
        "recommend weakly related or unrelated candidates merely to fill a list. Briefly "
        "identify rejected candidates when that distinction prevents a misleading "
        "recommendation. Broad field overlap alone is insufficient: the supplied title or "
        "snippet must match the query's core method, problem, or setting. A generic paper "
        "sharing only an umbrella term such as reinforcement learning may be labelled a "
        "nearby lead, but not a recommended citation. Never infer visual control, a method, "
        "or a result from a generic robotics/RL snippet. The deterministic eligibility "
        "groups in the input are mandatory: recommend only eligible_candidates; "
        "nearby_candidates may be listed only as nearby leads. If no candidate is eligible, "
        "say that no directly relevant citation was found.\n"
        "Never fabricate or alter DOI, URL, venue, year, author, title, or findings."
    ),
    "paper_recommendation": (
        "Task: write one or two grounded sentences per retrieved paper.\n"
        "Use only RagEvidencePack evidence_snippets. Key each summary by source_id.\n"
        "Respond with a single JSON object (no markdown fences) with key "
        '"paper_summaries": an array of {"source_id": string, "summary": string}.\n'
        "Do not invent titles, authors, years, URLs, scores, or APA citations."
    ),
    "peer_review_critique": (
        "Task: review the uploaded paper as a constructive academic peer reviewer.\n"
        "Required sections: strengths, weaknesses, missing evidence, suggested improvements, "
        "and evidence basis.\n"
        "Ground every point in supplied paper sections or deterministic structural checks. "
        "Name the relevant section when possible. Distinguish reviewer inference from facts "
        "stated by the paper. Do not invent experiments, citations, or external comparisons.\n"
        "Section excerpts may be truncated. Never claim that content is absent merely because "
        "it is not visible in an excerpt. Make an absence claim only when a deterministic "
        "structural check supports it. The review_presence_signals object records lexical "
        "presence in the full parsed sections: when a signal is true, do not call that item "
        "missing. You may question adequacy only when supplied text supports the concern.\n"
        "Use at most two concise bullets in each required section. Prefer one well-supported "
        "issue over a generic checklist. Do not wrap the answer in a Markdown code fence.\n"
        "Input object: ParsedPaper only. No retrieval or external evidence is available."
    ),
    "beginner_explanation": (
        "Task: explain the paper to a beginner without losing technical precision. Keep "
        "citations/source IDs for claims about external papers."
    ),
    "follow_up_qa": (
        "Task: answer a follow-up question using only current evidence. State uncertainty "
        "when the evidence pack does not contain enough information."
    ),
}


FEW_SHOT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "peer_review_critique": [
        {
            "input": (
                "Known present: episode return, wall-clock time, SAC/CURL/DrQ/"
                "DreamerV2 comparisons, hyperparameters, and ablations. Structural "
                "check: no explicit limitations section. Evaluation scope: DMC."
            ),
            "output": (
                "Strength: The evaluation reports episode return and wall-clock results "
                "against named model-free and model-based baselines. Weakness: The paper "
                "does not explicitly discuss limitations. Missing evidence: The supplied "
                "evaluation is scoped to DMC, so broader generalization is not established. "
                "Improvement: add a limitations discussion and clearly delimit claims to "
                "the evaluated benchmark."
            ),
        }
    ],
    "citation_recommendation": [
        {
            "input": (
                "Query: retrieval-grounded question answering. Candidate [A] directly "
                "studies retrieval-grounded generation and supplies APA string <APA A>. "
                "Candidate [B] studies unrelated image classification."
            ),
            "output": (
                "Recommended: [A] <APA A>. Its supplied evidence directly addresses "
                "retrieval-grounded generation. Not recommended: [B], because the supplied "
                "evidence is about an unrelated task. Do not replace placeholders or copy "
                "example entities into the real answer."
            ),
        }
    ],
    "research_gap_identification": [
        {
            "input": "Evidence says section-aware matching is useful; no user study is listed.",
            "output": (
                "Gap: The system should validate whether section-aware recommendations "
                "improve user satisfaction; this is an assumption unless a study source is added."
            ),
        }
    ],
}


def build_prompt(task: str, style: str = "technical", few_shot: bool = False) -> str:
    template = PROMPT_TEMPLATES[task]
    lines = [SYSTEM_GUARDRAILS, f"Style: {style}.", template]
    if few_shot and task in FEW_SHOT_EXAMPLES:
        lines.append("Few-shot examples:")
        for example in FEW_SHOT_EXAMPLES[task]:
            lines.append(f"Input: {example['input']}")
            lines.append(f"Output: {example['output']}")
    return "\n\n".join(lines)


def direct_chat_prompt_record(
    user_query: str,
    style: str,
    *,
    query_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a no-retrieval prompt for conversational text input."""
    prompt = "\n\n".join(
        [
            DIRECT_CHAT_GUARDRAILS,
            f"Style: {style}.",
            PROMPT_TEMPLATES["direct_text_chat"],
        ]
    )
    return {
        "prompt_id": "direct_text_chat",
        "task": "direct_text_chat",
        "style": style,
        "system_guardrails": DIRECT_CHAT_GUARDRAILS,
        "zero_shot_prompt": prompt,
        "few_shot_prompt": prompt,
        "input": {
            "parsed_paper": None,
            "rag_evidence_pack": None,
            "user_query": user_query,
        },
        "query_analysis": query_analysis,
        "expected_output_contract": {
            "input_contract": "text_only",
            "must_include_source_ids": False,
            "must_not_claim_retrieval": True,
            "must_not_fabricate_metadata": True,
        },
    }


def paper_summary_prompt_record(
    parsed_paper: dict[str, Any],
    style: str,
    *,
    user_query: str = "Summarize the supplied paper.",
    query_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    intent = str((query_analysis or {}).get("intent") or "")
    summary_cues = (
        "summarize",
        "summary",
        "overview",
        "core contribution",
        "main contribution",
    )
    is_specific_question = (
        intent == "question"
        and not any(cue in user_query.lower() for cue in summary_cues)
    )
    task = "paper_question_answer" if is_specific_question else "uploaded_paper_summary"
    sections = parsed_paper.get("sections") or {}
    analysis = parsed_paper.get("analysis") or {}
    structural_checks = analysis.get("structural_checks") or []
    structural_codes = {
        str(check.get("code") or "")
        for check in structural_checks
        if isinstance(check, dict)
    }
    explicit_limitations = (
        bool(str(sections.get("limitations") or "").strip())
        and "no_explicit_limitations" not in structural_codes
    )
    summary_presence_signals = {
        "explicit_limitations_section": explicit_limitations,
        "structural_check_codes": sorted(structural_codes),
    }
    evidence_instruction = (
        "Deterministic structural facts (do not contradict): "
        + json.dumps(summary_presence_signals, ensure_ascii=True)
    )
    if task == "uploaded_paper_summary" and not explicit_limitations:
        evidence_instruction += (
            " In the Limitations section, state that no explicit limitations section was "
            "detected. Do not invent limitations or say the paper notes them. A critique of "
            "evaluation scope is allowed only when supported by supplied sections and must be "
            "labeled as reviewer inference."
        )
    prompt = "\n\n".join(
        [
            PAPER_ONLY_GUARDRAILS,
            f"Style: {style}.",
            f"User request: {user_query}",
            PROMPT_TEMPLATES[task],
            evidence_instruction,
        ]
    )
    return {
        "prompt_id": "paper_only_summary",
        "task": task,
        "style": style,
        "system_guardrails": PAPER_ONLY_GUARDRAILS,
        "zero_shot_prompt": prompt,
        "few_shot_prompt": prompt,
        "input": {
            "parsed_paper": parsed_paper,
            "rag_evidence_pack": None,
            "user_query": user_query,
            "summary_presence_signals": summary_presence_signals,
        },
        "query_analysis": query_analysis,
        "expected_output_contract": {
            "input_contract": "parsed_paper_only",
            "must_include_source_ids": False,
            "must_answer_user_request_directly": is_specific_question,
            "must_not_use_external_evidence": True,
            "must_not_fabricate_metadata": True,
        },
    }


def paper_review_prompt_record(
    parsed_paper: dict[str, Any],
    style: str,
    *,
    user_query: str = "Write a constructive peer review of the supplied paper.",
    query_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a paper-only peer-review prompt with a task-specific example."""
    sections = parsed_paper.get("sections") or {}
    section_text = "\n".join(
        str(value)
        for value in sections.values()
        if isinstance(value, str)
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", section_text.lower())
    analysis = parsed_paper.get("analysis") or {}
    checks = analysis.get("structural_checks") or []
    check_codes = {
        str(check.get("code") or "")
        for check in checks
        if isinstance(check, dict)
    }
    review_presence_signals = {
        "explicit_limitations_section": bool(
            str(sections.get("limitations") or "").strip()
        )
        and "no_explicit_limitations" not in check_codes,
        "episode_return_or_reward_metrics": any(
            term in normalized
            for term in ("episode return", "reward curve", "convergence rate")
        ),
        "hyperparameter_details": bool(
            re.search(r"\bhyper parameter", normalized)
            or "learning rate" in normalized
            or "batch size" in normalized
        ),
        "exploration_schedule": "exploration schedule" in normalized,
        "ablation_study": "ablation" in normalized,
        "model_free_baseline_comparison": "model free" in normalized,
        "model_based_baseline_comparison": "model based" in normalized,
        "public_implementation": any(
            term in normalized
            for term in ("publicly release", "publicly available", "github com")
        ),
    }
    lower_text = section_text.lower()

    def present_terms(candidates: tuple[str, ...]) -> list[str]:
        return [
            term
            for term in candidates
            if term.lower() in lower_text
        ]

    review_evidence_inventory = {
        "reported_metrics": present_terms(
            (
                "episode return",
                "wall-clock time",
                "sample efficiency",
                "frames per second",
                "FPS",
            )
        ),
        "named_baselines_or_backbones": present_terms(
            (
                "SAC",
                "DDPG",
                "CURL",
                "DrQ",
                "DreamerV2",
                "Dreamer-v2",
                "RAD",
                "PlaNet",
            )
        ),
        "experiment_details": present_terms(
            (
                "ablation study",
                "learning rate",
                "batch size",
                "replay buffer",
                "exploration schedule",
                "n-step returns",
            )
        ),
        "benchmarks_or_tasks": present_terms(
            (
                "DeepMind Control Suite",
                "DMC",
                "Humanoid Stand",
                "Humanoid Walk",
                "Humanoid Run",
            )
        ),
    }
    inventory_instruction = (
        "Known-present evidence inventory (do not contradict this list): "
        + json.dumps(review_evidence_inventory, ensure_ascii=True)
    )
    zero_shot = "\n\n".join(
        [
            PAPER_ONLY_GUARDRAILS,
            f"Style: {style}.",
            PROMPT_TEMPLATES["peer_review_critique"],
            inventory_instruction,
        ]
    )
    example = FEW_SHOT_EXAMPLES["peer_review_critique"][0]
    few_shot = "\n\n".join(
        [
            zero_shot,
            "Few-shot example showing the required evidence-to-critique pattern:",
            f"Input: {example['input']}",
            f"Output: {example['output']}",
        ]
    )
    return {
        "prompt_id": "paper_only_peer_review",
        "task": "peer_review_critique",
        "style": style,
        "system_guardrails": PAPER_ONLY_GUARDRAILS,
        "zero_shot_prompt": zero_shot,
        "few_shot_prompt": few_shot,
        "input": {
            "parsed_paper": parsed_paper,
            "rag_evidence_pack": None,
            "user_query": user_query,
            "review_presence_signals": review_presence_signals,
            "review_evidence_inventory": review_evidence_inventory,
        },
        "query_analysis": query_analysis,
        "expected_output_contract": {
            "input_contract": "parsed_paper_only",
            "must_include_source_ids": False,
            "must_ground_critique_in_paper": True,
            "must_separate_inference_from_stated_evidence": True,
            "must_not_use_external_evidence": True,
            "must_not_fabricate_metadata": True,
        },
    }


def topic_synthesis_prompt_record(
    evidence_pack: dict[str, Any],
    style: str,
    *,
    query_analysis: dict[str, Any] | None = None,
    prompt_id: str = "topic_synthesis",
) -> dict[str, Any]:
    """Build the explicit RAG input contract used by topic synthesis."""
    return {
        "prompt_id": prompt_id,
        "task": "topic_search_synthesis",
        "style": style,
        "system_guardrails": SYSTEM_GUARDRAILS,
        "zero_shot_prompt": build_prompt(
            "topic_search_synthesis",
            style=style,
            few_shot=False,
        ),
        "few_shot_prompt": build_prompt(
            "topic_search_synthesis",
            style=style,
            few_shot=True,
        ),
        "input": {
            "parsed_paper": None,
            "rag_evidence_pack": evidence_pack,
        },
        "query_analysis": query_analysis,
        "expected_output_contract": {
            "input_contract": "rag_evidence_pack",
            "must_include_source_ids": True,
            "must_separate_findings_from_assumptions": True,
            "must_not_fabricate_metadata": True,
        },
    }


def paper_recommendation_prompt_record(
    evidence_pack: dict[str, Any],
    style: str,
    *,
    query_analysis: dict[str, Any] | None = None,
    prompt_id: str = "paper_recommendation",
) -> dict[str, Any]:
    """Build the per-paper summary contract for structured recommendations."""
    return {
        "prompt_id": prompt_id,
        "task": "paper_recommendation",
        "style": style,
        "system_guardrails": SYSTEM_GUARDRAILS,
        "zero_shot_prompt": build_prompt(
            "paper_recommendation",
            style=style,
            few_shot=False,
        ),
        "few_shot_prompt": build_prompt(
            "paper_recommendation",
            style=style,
            few_shot=False,
        ),
        "input": {
            "parsed_paper": None,
            "rag_evidence_pack": evidence_pack,
            "user_query": evidence_pack.get("query", ""),
        },
        "query_analysis": query_analysis,
        "expected_output_contract": {
            "input_contract": "rag_evidence_pack",
            "must_include_source_ids": True,
            "must_not_fabricate_metadata": True,
            "output_shape": "paper_summaries_by_source_id",
        },
    }


def _load_test_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load parsed paper and RAG pack generated from real test PDFs."""
    artifacts = (
        Path(__file__).resolve().parents[3] / "tests" / "papers" / "artifacts"
    )
    parsed_path = artifacts / "drq_v2_parsed.json"
    rag_path = artifacts / "drq_v2_rag_pack.json"
    if not parsed_path.is_file() or not rag_path.is_file():
        raise FileNotFoundError(
            "Test artifacts missing. Run: python tests/harness/bootstrap_artifacts.py"
        )
    with parsed_path.open(encoding="utf-8") as f:
        parsed_paper = json.load(f)
    with rag_path.open(encoding="utf-8") as f:
        evidence_pack = json.load(f)
    return parsed_paper, evidence_pack


def fixed_prompt_records() -> list[dict[str, Any]]:
    parsed_paper, evidence_pack = _load_test_artifacts()
    tasks = [
        ("P01", "uploaded_paper_summary", "technical"),
        ("P02", "topic_search_synthesis", "technical"),
        ("P03", "research_gap_identification", "technical"),
        ("P04", "citation_recommendation", "technical"),
        ("P05", "peer_review_critique", "reviewer"),
        ("P06", "beginner_explanation", "beginner"),
    ]
    records: list[dict[str, Any]] = []
    for prompt_id, task, style in tasks:
        if task == "uploaded_paper_summary":
            record = paper_summary_prompt_record(parsed_paper, style)
            record["prompt_id"] = prompt_id
            records.append(record)
            continue
        if task == "topic_search_synthesis":
            records.append(
                topic_synthesis_prompt_record(
                    evidence_pack,
                    style,
                    prompt_id=prompt_id,
                )
            )
            continue
        if task == "peer_review_critique":
            record = paper_review_prompt_record(parsed_paper, style)
            record["prompt_id"] = prompt_id
            records.append(record)
            continue
        records.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "style": style,
                "system_guardrails": SYSTEM_GUARDRAILS,
                "zero_shot_prompt": build_prompt(task, style=style, few_shot=False),
                "few_shot_prompt": build_prompt(task, style=style, few_shot=True),
                "input": {
                    "parsed_paper": (
                        None
                        if task == "citation_recommendation"
                        else parsed_paper
                    ),
                    "rag_evidence_pack": evidence_pack,
                },
                "expected_output_contract": {
                    "must_include_source_ids": True,
                    "must_separate_findings_from_assumptions": True,
                    "must_not_fabricate_metadata": True,
                },
            }
        )
    return records


def react_tool_call_examples() -> list[dict[str, Any]]:
    return [
        {
            "user_query": "Find recent papers extending retrieval augmented generation for scientific literature.",
            "tool_call": {
                "tool": "search_offline",
                "arguments": {
                    "query": "retrieval augmented generation scientific literature",
                    "top_k": 10,
                    "preferred_categories": ["cs.CL", "cs.AI", "cs.LG"],
                },
            },
            "backend_rule": "The backend executes the structured call; the model does not fetch papers directly.",
        },
        {
            "user_query": "Add citation counts for the top recommendations if cached live data exists.",
            "tool_call": {
                "tool": "search_cached_live",
                "arguments": {
                    "query": "top recommendation citation counts",
                    "cache_namespace": "openalex_semantic_scholar",
                    "network_allowed": False,
                },
            },
            "backend_rule": "Cache is checked first; live network use is a separate explicit mode.",
        },
        {
            "user_query": "Show details for arXiv paper 2504.02377.",
            "tool_call": {
                "tool": "fetch_paper_details",
                "arguments": {"paper_id": "arxiv_2504_02377", "source": "arxiv_api"},
            },
            "backend_rule": "The detail fetcher returns a PaperRecord-compatible object.",
        },
    ]
