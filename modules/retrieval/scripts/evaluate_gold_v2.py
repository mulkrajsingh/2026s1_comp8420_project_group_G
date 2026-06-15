"""Compare retrieval metrics on v1 and v2 gold label sets.

Scores the same fitted retrievers against hand-picked seed labels and the
expanded pooling plus citation-graph labels. Writes side-by-side CSV tables and
a grouped bar chart under results/retrieval/.

Usage:
    python modules/retrieval/scripts/evaluate_gold_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
CORPUS = REPO_ROOT / "modules/dataset/data/processed/dev_5k_enriched.jsonl"
RESULTS_DIR = RETRIEVAL_DIR / "results/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.evaluation import evaluate_retriever  # noqa: E402
from app.fixtures import EVAL_QUERIES, load_papers  # noqa: E402
from app.retrieval.embeddings import EmbeddingRetriever  # noqa: E402
from app.retrieval.hybrid_ranker import HybridRanker, HybridRRFRanker  # noqa: E402
from app.retrieval.section_aware import SectionAwareRetriever  # noqa: E402
from app.retrieval.tfidf_bm25 import BM25Retriever, TfidfRetriever  # noqa: E402

V2_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"


def main() -> None:
    """Evaluate retrievers on v1 and v2 gold and write comparison tables."""
    papers = load_papers(CORPUS)
    print(f"Loaded {len(papers)} papers")

    tfidf = TfidfRetriever().fit(papers)
    bm25 = BM25Retriever().fit(papers)
    print("Fitted TF-IDF + BM25")

    emb = EmbeddingRetriever()
    emb._papers = list(papers)
    emb.load_embeddings(RETRIEVAL_DIR / "data/processed/retrieval_index/embeddings")
    print(f"Loaded precomputed embeddings ({emb.model_name})")

    print("Fitting section-aware retriever...")
    section = SectionAwareRetriever(emb)
    section.fit(papers, show_progress=True)
    print("Section-aware retriever ready")

    hybrid_rrf = HybridRRFRanker(
        bm25_retriever=bm25,
        embedding_retriever=emb,
    )
    hybrid_rrf._papers = papers

    hybrid = HybridRanker(
        tfidf_retriever=tfidf,
        bm25_retriever=bm25,
        embedding_retriever=emb,
        section_aware_retriever=section,
    )
    hybrid._papers = papers

    retrievers = [hybrid_rrf, tfidf, bm25, emb, section, hybrid]

    # v1 gold standard (from app.fixtures)
    queries_v1 = [{"query": q["query"], "relevant_ids": q["relevant_ids"]} for q in EVAL_QUERIES]

    # v2 gold standard (pooling + citation-graph expansion)
    v2_data = json.loads(V2_PATH.read_text())
    queries_v2 = [{"query": q["query"], "relevant_ids": q["relevant_ids_v2"]} for q in v2_data]

    n_v1 = sum(len(q["relevant_ids"]) for q in queries_v1)
    n_v2 = sum(len(q["relevant_ids"]) for q in queries_v2)
    print(f"\nGold standard sizes: v1={n_v1} relevant_ids total, v2={n_v2} relevant_ids total\n")

    v1_summaries, v2_summaries = [], []
    for retriever in retrievers:
        df_v1 = evaluate_retriever(retriever, queries_v1)
        df_v2 = evaluate_retriever(retriever, queries_v2)

        df_v2.to_csv(RESULTS_DIR / f"detail_v2_{retriever.name}.csv", index=False)

        row_v1 = {"retriever": retriever.name}
        row_v2 = {"retriever": retriever.name}
        for col in df_v1.columns:
            if col not in ("query", "n_relevant"):
                row_v1[col] = round(df_v1[col].mean(), 4)
                row_v2[col] = round(df_v2[col].mean(), 4)
        v1_summaries.append(row_v1)
        v2_summaries.append(row_v2)
        print(f"{retriever.name}: v1 P@5={row_v1['P@5']:.3f} nDCG@5={row_v1['nDCG@5']:.3f} MAP={row_v1['MAP']:.3f}  "
              f"|  v2 P@5={row_v2['P@5']:.3f} nDCG@5={row_v2['nDCG@5']:.3f} MAP={row_v2['MAP']:.3f}")

    comp_v1 = pd.DataFrame(v1_summaries).set_index("retriever")
    comp_v2 = pd.DataFrame(v2_summaries).set_index("retriever")
    comp_v2.to_csv(RESULTS_DIR / "retrieval_comparison_v2.csv")
    # Canonical CSV consumed by regenerate_eval_charts.py and the notebook charts.
    comp_v2.to_csv(RESULTS_DIR / "retrieval_comparison.csv")

    # Side-by-side table with deltas for the headline metrics
    headline = ["P@5", "R@5", "nDCG@5", "MAP", "MRR"]
    side_by_side = pd.DataFrame(index=comp_v1.index)
    for m in headline:
        side_by_side[f"{m}_v1"] = comp_v1[m]
        side_by_side[f"{m}_v2"] = comp_v2[m]
        side_by_side[f"{m}_delta"] = (comp_v2[m] - comp_v1[m]).round(4)
    side_by_side.to_csv(RESULTS_DIR / "retrieval_comparison_v1_vs_v2.csv")
    print("\n", side_by_side.to_string())

    # Grouped bar chart: P@5, nDCG@5, MAP for v1 vs v2 per retriever
    metrics_plot = ["P@5", "nDCG@5", "MAP"]
    retriever_names = comp_v1.index.tolist()
    fig, axes = plt.subplots(1, len(metrics_plot), figsize=(13, 4.5), sharey=True)
    x = np.arange(len(retriever_names))
    width = 0.35
    for ax, m in zip(axes, metrics_plot):
        ax.bar(x - width / 2, comp_v1[m], width, label="v1 gold (5/query)", color="#9C27B0", alpha=0.85)
        ax.bar(x + width / 2, comp_v2[m], width, label="v2 gold (pooled+citation)", color="#4CAF50", alpha=0.85)
        ax.set_title(m)
        ax.set_xticks(x)
        ax.set_xticklabels(retriever_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Retrieval evaluation: v1 vs v2 (pooling + citation-graph) gold standard")
    plt.tight_layout()
    out_path = RESULTS_DIR / "gold_v1_vs_v2_chart.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")
    print(f"Saved {RESULTS_DIR / 'retrieval_comparison_v2.csv'}")
    print(f"Saved {RESULTS_DIR / 'retrieval_comparison_v1_vs_v2.csv'}")


if __name__ == "__main__":
    main()
