"""Build RagEvidencePack JSON for the LLM synthesis module.

Maps ranked recommendations to bounded abstract snippets, attaches prompt
template strings for schema compatibility, and validates the export shape.
"""

from __future__ import annotations

from .schemas import validate_rag_evidence_pack

# Prompt templates are kept for schema compatibility. Live prompts live in
# modules/llm/app/prompt_library.py.
PROMPT_TEMPLATES = {
    "zero_shot": (
        "You are a research assistant. Using ONLY the evidence snippets below, "
        "answer the query. Cite each claim using the source_id in brackets.\n\n"
        "Query: {query}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Answer (cite sources):"
    ),
    "few_shot": (
        "You are a research assistant. Below are examples of how to answer using evidence.\n\n"
        "Example:\n"
        "Query: What is transfer learning?\n"
        "Evidence: [S1] Transfer learning reuses a model trained on one task for another...\n"
        "Answer: Transfer learning reuses pretrained models for new tasks [S1].\n\n"
        "Now answer the following using ONLY the evidence below:\n"
        "Query: {query}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Answer (cite sources):"
    ),
    "chain_of_thought": (
        "You are a research assistant. Think step by step before answering.\n\n"
        "Query: {query}\n\n"
        "Evidence:\n{evidence}\n\n"
        "Step 1 - Identify the most relevant papers from the evidence.\n"
        "Step 2 - Extract key findings from those papers.\n"
        "Step 3 - Synthesise a coherent answer citing each source.\n\n"
        "Answer:"
    ),
}


# Most arXiv abstracts are 1,000-2,000 characters. A 300-char cap discarded the
# bulk of each abstract before the LLM ever saw it, which left the synthesizer
# unable to actually summarise or explain a paper. Keep enough of the abstract
# for grounded synthesis while still bounding prompt size.
SNIPPET_MAX_CHARS = 2000


def _snippet(paper: dict, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    abstract = paper.get("abstract", "")
    return abstract[:max_chars].rstrip() + ("..." if len(abstract) > max_chars else "")


def build_rag_evidence_pack(
    query: str,
    recommendations: list[dict],
    retrieval_mode: str = "offline",
    max_snippets: int = 10,
    prompt_strategy: str = "chain_of_thought",
) -> dict:
    """Build a validated RagEvidencePack from ranked recommendations.

    Truncates abstracts to ``SNIPPET_MAX_CHARS`` and renders zero-shot,
    few-shot, and chain-of-thought prompt strings for downstream selection.
    """
    snippets = []
    for rec in recommendations[:max_snippets]:
        paper = rec["paper"]
        snippet = {
            "source_id": paper["paper_id"],
            "title": paper["title"],
            "snippet": _snippet(paper),
            "metadata": {
                "year": (paper.get("published_date") or "")[:4] or None,
                "authors": paper.get("authors", []),
                "venue": paper.get("venue"),
                "doi": paper.get("doi"),
                "categories": paper.get("categories", []),
                "citation_count": None,
                "url": paper.get("url"),
            },
        }
        snippets.append(snippet)

    # Build evidence string for prompt templates
    evidence_str = "\n".join(
        f"[{s['source_id']}] {s['title']}: {s['snippet']}"
        for s in snippets
    )

    # Render all three prompt templates (Prompt Engineering technique)
    rendered_prompts = {
        strategy: template.format(query=query, evidence=evidence_str)
        for strategy, template in PROMPT_TEMPLATES.items()
    }

    pack = {
        "query": query,
        "retrieval_mode": retrieval_mode,
        "prompt_strategy": prompt_strategy,
        "prompt_templates": rendered_prompts,
        "candidates": recommendations,
        "evidence_snippets": snippets,
    }

    validate_rag_evidence_pack(pack)
    return pack
