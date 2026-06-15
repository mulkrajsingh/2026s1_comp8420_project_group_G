"""Regenerate retrieval evaluation charts from saved CSV metrics.

Rebuilds comparison bar charts and nDCG heatmaps from on-disk evaluation CSVs
without rerunning the full retrieval notebook. Output style matches the notebook
plots in modules/retrieval/notebooks/03_rag_recommendation_evaluation.ipynb.

Usage:
    python modules/retrieval/scripts/regenerate_eval_charts.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
RESULTS_DIR = RETRIEVAL_DIR / "results/retrieval"
V2_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"

sys.path.insert(0, str(RETRIEVAL_DIR))

BAR_COLOURS = ["#E91E63", "#9C27B0", "#2196F3", "#4CAF50", "#FF9800"]


def plot_comparison_chart(comparison: pd.DataFrame) -> None:
    metrics_plot = ["P@5", "R@5", "nDCG@5", "MRR"]
    retriever_names = comparison.index.tolist()
    x = np.arange(len(metrics_plot))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(retriever_names):
        colour = BAR_COLOURS[i % len(BAR_COLOURS)]
        vals = [comparison.loc[name, m] if m in comparison.columns else 0 for m in metrics_plot]
        ax.bar(x + i * width, vals, width, label=name, color=colour, alpha=0.85)

    ax.set_xticks(x + width * (len(retriever_names) - 1) / 2)
    ax.set_xticklabels(metrics_plot)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Method Comparison")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    out_path = RESULTS_DIR / "retrieval_comparison_chart.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ndcg_heatmap(comparison: pd.DataFrame, eval_queries_v2: list[dict]) -> None:
    heatmap_data = {}
    for name in comparison.index:
        detail_path = RESULTS_DIR / f"detail_v2_{name}.csv"
        df_q = pd.read_csv(detail_path)
        heatmap_data[name] = df_q["nDCG@10"].values

    query_labels = [q["query"][:40] + "..." for q in eval_queries_v2]
    heatmap_df = pd.DataFrame(heatmap_data, index=query_labels)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("nDCG@10 per Query per Retriever (v2 gold standard)")
    ax.set_xlabel("Retriever")
    ax.set_ylabel("Query")
    plt.tight_layout()
    out_path = RESULTS_DIR / "ndcg_heatmap.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    """Regenerate comparison charts from saved evaluation CSVs."""
    comparison = pd.read_csv(RESULTS_DIR / "retrieval_comparison.csv", index_col=0)
    eval_queries_v2 = json.loads(V2_PATH.read_text())
    print(f"Retrievers: {comparison.index.tolist()}")
    plot_comparison_chart(comparison)
    plot_ndcg_heatmap(comparison, eval_queries_v2)


if __name__ == "__main__":
    main()
