"""Six train-only anchor rows disjoint from the fixed evaluation prompts."""

from __future__ import annotations

from lora_dataset.io import chat_record


SYSTEM = (
    "You are a local research-paper assistant. Use only supplied paper fields or "
    "evidence. Do not invent bibliographic metadata or unsupported findings."
)


def build_seed_rows() -> list[dict]:
    """Return local format-anchor rows excluded from evaluation prompts."""
    examples = [
        (
            "TRAIN_SEED_SUMMARY",
            "uploaded_paper_summary",
            "Summarize this ParsedPaper only: abstract says a sparse retriever is evaluated "
            "against a keyword baseline; method and result sections are absent.",
            "## Scope\nThe paper evaluates a sparse retriever against a keyword baseline.\n\n"
            "## Method\nNot stated in the supplied ParsedPaper.\n\n"
            "## Results\nNot stated in the supplied ParsedPaper.",
        ),
        (
            "TRAIN_SEED_PAPER_QA",
            "paper_question_answering",
            "ParsedPaper result: accuracy increased from 0.71 to 0.78. "
            "Question: What improvement is reported?",
            "The supplied paper reports an accuracy increase from 0.71 to 0.78.",
        ),
        (
            "TRAIN_SEED_REVIEW",
            "peer_review_critique",
            "Review a ParsedPaper whose abstract states the approach but provides no baseline, "
            "metric, dataset size, or reproducibility details.",
            "Strengths: The research objective and proposed approach are stated clearly.\n"
            "Weaknesses: The supplied paper does not identify a baseline or evaluation metric.\n"
            "Missing evidence: Dataset size and reproducibility details are absent.\n"
            "Suggested improvements: Add baselines, metrics, dataset statistics, and implementation details.",
        ),
        (
            "TRAIN_SEED_RAG",
            "topic_search_synthesis",
            "Evidence: [S1] lexical retrieval is efficient; [S2] dense retrieval improves semantic matching. "
            "Use APS citations [1] and [2].",
            '{"summary":"The evidence contrasts efficient lexical retrieval [S1][1] with '
            'semantic dense retrieval [S2][2].","key_findings":["Lexical retrieval is '
            'efficient [S1][1].","Dense retrieval improves semantic matching [S2][2]."],'
            '"research_gaps":["The supplied evidence does not establish which method performs '
            'better on the project corpus."]}',
        ),
        (
            "TRAIN_SEED_CITATION",
            "citation_recommendation",
            "Recommend supplied paper [S1]. Metadata: A. Researcher, Grounded Search, "
            "arXiv:2401.00001 (2024).",
            "## Recommended citations\n- [1] Grounded Search [S1]\n\n"
            "## References\n[1] A. Researcher, Grounded Search, arXiv:2401.00001 (2024).",
        ),
        (
            "TRAIN_SEED_REFUSAL",
            "refusal",
            "The supplied paper does not state its training cost. What exact GPU-hours did it use?",
            "The supplied paper does not provide GPU-hour information, so an exact value cannot be given.",
        ),
    ]
    return [
        chat_record(
            user_content=user,
            assistant_content=assistant,
            source="local_train_anchor",
            task=task,
            prompt_id=prompt_id,
            license_note="Project-local train-only anchor; excluded from fixed evaluation prompts.",
            system_content=SYSTEM,
        )
        for prompt_id, task, user, assistant in examples
    ]
