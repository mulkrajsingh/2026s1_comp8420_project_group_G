"""Generate a flowchart of the retrieval and RAG pipeline.

Draws offline indexing steps and per-query ranking paths used by
``recommend-topic``, including hybrid fusion and RagEvidencePack export.

Usage:
    python modules/retrieval/scripts/generate_workflow_diagram.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = REPO_ROOT / "modules/retrieval/results/retrieval/system_workflow_diagram.png"


def box(ax, cx, cy, w, h, text, facecolor="white", edgecolor="#333333",
        fontsize=8.5, fontweight="normal"):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2, zorder=3,
    ))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, zorder=4, linespacing=1.35)
    return cx, cy, w, h


def arrow(ax, p1, p2, rad=0.0, color="#555555", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=13, color=color, lw=lw,
        linestyle=ls, zorder=2,
    ))


def main() -> None:
    """Render and save the retrieval pipeline workflow diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.3)
    ax.axis("off")

    ax.text(5, 13.95, "Bank's Retrieval & RAG Pipeline — System Activation Workflow",
            ha="center", va="center", fontsize=12.5, fontweight="bold")

    # ---- Offline / build-time panel -----------------------------------
    ax.add_patch(Rectangle((0.2, 11.0), 9.6, 2.5, facecolor="#f2f2f2",
                            edgecolor="#aaaaaa", zorder=0))
    ax.text(0.4, 13.3, "OFFLINE — BUILD-TIME  (build-retrieval-index, run once)",
            fontsize=8, fontweight="bold", color="#666666", ha="left", va="top", zorder=1)

    box(ax, 5, 12.85, 5.5, 0.55, "Load 5,000-paper corpus\n(dev_5k_enriched.jsonl)")

    box(ax, 1.5, 11.55, 1.9, 0.95, "TF-IDF Index\n(1-2 gram\nTfidfVectorizer)", fontsize=7.8)
    box(ax, 3.7, 11.55, 1.9, 0.95, "BM25 Index\n(tokenised\ncorpus)", fontsize=7.8)
    box(ax, 5.9, 11.55, 1.9, 0.95, "SPECTER2 Embeddings\n(5000 x 768,\nspecter2_base)", fontsize=7.5)
    box(ax, 8.1, 11.55, 1.9, 0.95, "Section-Aware Index\n(sentence-role\nclassifier + 4 sub-emb.)", fontsize=7.2)

    arrow(ax, (5, 12.575), (1.5, 12.025), rad=-0.25)
    arrow(ax, (5, 12.575), (3.7, 12.025), rad=-0.12)
    arrow(ax, (5, 12.575), (5.9, 12.025), rad=0.12)
    arrow(ax, (5, 12.575), (8.1, 12.025), rad=0.25)

    # SPECTER2 -> Section-Aware dependency
    arrow(ax, (6.85, 11.55), (7.15, 11.55))

    # ---- Query-time / trigger panel ------------------------------------
    ax.add_patch(Rectangle((0.2, 0.3), 9.6, 10.5, facecolor="#eaf3fb",
                            edgecolor="#6f9fc9", zorder=0))
    ax.text(0.4, 10.55, "QUERY-TIME — RUNS ON EVERY TRIGGERED REQUEST",
            fontsize=8, fontweight="bold", color="#2c5d85", ha="left", va="top", zorder=1)

    box(ax, 5, 10.0, 9.0, 0.8,
        'TRIGGER\nIncoming query, e.g.\n'
        'python -m app.cli recommend-topic --query "..."\n'
        '(CLI / notebook / future API call)',
        facecolor="#ffe2b3", edgecolor="#cc7a00", fontsize=8.5, fontweight="bold")

    # offline indexes feed the ranker
    arrow(ax, (8.1, 11.075), (9.45, 8.7), rad=0.3, color="#888888", ls="--")
    ax.text(9.7, 9.85, "fitted\nretrievers", fontsize=6.5, color="#888888",
            ha="center", style="italic", rotation=90, zorder=1)

    # trigger -> hybrid
    arrow(ax, (5, 9.6), (5, 9.4))

    box(ax, 5, 8.55, 9.0, 1.7,
        "HybridRanker.rank(query, top_k=10)\n"
        "1. Score every paper on 6 signals — TF-IDF, BM25,\n"
        "   SPECTER2 cos-sim, section-aware cos-sim, recency,\n"
        "   category overlap\n"
        "2. Min-max normalise each signal across all 5,000 papers\n"
        "3. Weighted sum -> final_score\n"
        "   (weights: .15 / .15 / .35 / .25 / .05 / .05)\n"
        "4. Sort descending, keep top-K",
        fontsize=8.3)

    arrow(ax, (5, 7.7), (5, 7.475))

    box(ax, 5, 7.0, 9.0, 0.95,
        "Per-result enrichment  (HybridRanker.recommend)\n"
        "Infer relation label (similar / foundational / recent /\n"
        "method_related / same_topic), build reason text,\n"
        "format APA-7 citation -> Recommendation objects",
        fontsize=8.3)

    arrow(ax, (5, 6.525), (5, 6.15))

    box(ax, 5, 5.85, 6.5, 0.6,
        "Write recommendations.json\n(top-K ranked Recommendation objects)",
        fontsize=8.3)

    arrow(ax, (5, 5.55), (5, 5.075))

    box(ax, 5, 4.55, 9.0, 1.05,
        "build_rag_evidence_pack(query, recommendations)\n"
        "Extract evidence snippets (title + 300-char abstract\n"
        "excerpt + metadata) and render 3 prompt templates:\n"
        "zero-shot, few-shot, chain-of-thought",
        fontsize=8.3)

    arrow(ax, (5, 4.025), (5, 3.65))

    box(ax, 5, 3.35, 6.5, 0.6, "Write rag_evidence_pack.json", fontsize=8.3)

    arrow(ax, (5, 3.05), (5, 2.4))

    box(ax, 5, 1.85, 9.0, 1.1,
        "Hand-off to Mulkraj's LLM/RAG synthesis module\n"
        "-> LLM generates final answer using the evidence\n"
        "   snippets + chosen prompt template, citing each\n"
        "   source_id -> returned to user with APA references",
        facecolor="#dff5df", edgecolor="#3a9d3a", fontsize=8.3, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
