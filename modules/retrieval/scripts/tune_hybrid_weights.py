"""Grid-search hybrid ensemble weights on cached component scores.

Loads precomputed score arrays and searches weight combinations against the v2
gold standard while reporting v1 metrics for the best combos.

Usage:
    python modules/retrieval/scripts/tune_hybrid_weights.py
"""

from __future__ import annotations

import itertools
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
RESULTS_DIR = RETRIEVAL_DIR / "results/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))

from app.evaluation import (  # noqa: E402
    average_precision,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    recall_at_k,
)
from app.fixtures import EVAL_QUERIES  # noqa: E402

SCORES_NPZ = RETRIEVAL_DIR / "data/processed/hybrid_component_scores.npz"
META_PATH = RETRIEVAL_DIR / "data/processed/hybrid_component_meta.json"
V2_PATH = RETRIEVAL_DIR / "data/processed/eval_queries_v2.json"

TUNED = ("tfidf", "bm25", "embedding", "section_aware")
FIXED_WEIGHTS = {"recency": 0.05, "category": 0.05}
TUNED_SUM = 0.9
STEP = 0.05
DEFAULT_WEIGHTS = {"tfidf": 0.15, "bm25": 0.15, "embedding": 0.35, "section_aware": 0.25}
K_LIST = [5, 10]


def grid_combos(step: float = STEP, total: float = TUNED_SUM, n: int = 4):
    """Yield all n-tuples of non-negative multiples of `step` summing to `total`."""
    units = round(total / step)
    for combo in itertools.product(range(units + 1), repeat=n):
        if sum(combo) == units:
            yield tuple(round(c * step, 4) for c in combo)


def evaluate_weights(weights: dict, per_query: list[dict], paper_ids: np.ndarray,
                      queries: list[str], relevant_map: dict[str, set[str]]) -> dict:
    """Average evaluate_retriever-style metrics over all queries for one weight setting."""
    all_w = {**weights, **FIXED_WEIGHTS}
    rows = []
    for qi, query in enumerate(queries):
        comp = per_query[qi]
        final = sum(all_w[k] * comp[k] for k in all_w)
        order = np.argsort(-final)[:max(K_LIST)]
        retrieved_ids = paper_ids[order].tolist()
        relevant = relevant_map[query]
        row = {}
        for k in K_LIST:
            row[f"P@{k}"] = precision_at_k(retrieved_ids, relevant, k)
            row[f"R@{k}"] = recall_at_k(retrieved_ids, relevant, k)
            row[f"F1@{k}"] = f1_at_k(retrieved_ids, relevant, k)
            row[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
        row["MRR"] = reciprocal_rank(retrieved_ids, relevant)
        row["MAP"] = average_precision(retrieved_ids, relevant)
        rows.append(row)
    return pd.DataFrame(rows).mean().to_dict()


def main() -> None:
    """Run the hybrid weight grid search and write tuning artifacts."""
    if not SCORES_NPZ.exists():
        sys.exit(f"Missing {SCORES_NPZ} -- run extract_hybrid_component_scores.py first")

    data = np.load(SCORES_NPZ)
    meta = json.loads(META_PATH.read_text())
    paper_ids = np.array(meta["paper_ids"])
    queries = meta["queries"]
    n_queries = len(queries)

    per_query = []
    for qi in range(n_queries):
        comp = {k: data[f"{k}_{qi}"] for k in (*TUNED, *FIXED_WEIGHTS)}
        per_query.append(comp)

    # Sanity check: category score should be 0 everywhere (query_categories=[] always)
    cat_nonzero = sum(int(np.any(per_query[qi]["category"] != 0)) for qi in range(n_queries))
    if cat_nonzero == 0:
        print("Note: 'category' component is 0.0 for all papers/queries (query_categories=[] "
              "in this setup) -- its 0.05 weight currently contributes nothing to ranking.\n")

    v2_data = json.loads(V2_PATH.read_text())
    relevant_v2 = {q["query"]: set(q["relevant_ids_v2"]) for q in v2_data}
    relevant_v1 = {q["query"]: set(q["relevant_ids"]) for q in EVAL_QUERIES}

    # Baseline: current _DEFAULT_WEIGHTS
    baseline_v2 = evaluate_weights(DEFAULT_WEIGHTS, per_query, paper_ids, queries, relevant_v2)
    baseline_v1 = evaluate_weights(DEFAULT_WEIGHTS, per_query, paper_ids, queries, relevant_v1)
    print(f"Current _DEFAULT_WEIGHTS (tuned 4): {DEFAULT_WEIGHTS} (+ recency=0.05, category=0.05)")
    print(f"  v2: P@5={baseline_v2['P@5']:.4f} R@5={baseline_v2['R@5']:.4f} "
          f"nDCG@5={baseline_v2['nDCG@5']:.4f} MAP={baseline_v2['MAP']:.4f} MRR={baseline_v2['MRR']:.4f}")
    print(f"  v1: P@5={baseline_v1['P@5']:.4f} R@5={baseline_v1['R@5']:.4f} "
          f"nDCG@5={baseline_v1['nDCG@5']:.4f} MAP={baseline_v1['MAP']:.4f} MRR={baseline_v1['MRR']:.4f}")

    # Grid search against v2
    results = []
    for combo in grid_combos():
        weights = dict(zip(TUNED, combo))
        metrics = evaluate_weights(weights, per_query, paper_ids, queries, relevant_v2)
        results.append({**weights, **metrics})

    df = pd.DataFrame(results)
    df = df.sort_values(["nDCG@5", "MAP", "P@5"], ascending=False).reset_index(drop=True)
    print(f"\nEvaluated {len(df)} weight combinations (step={STEP}, sum={TUNED_SUM}) against v2 gold")

    cols = [*TUNED, "P@5", "R@5", "nDCG@5", "MAP", "MRR", "P@10", "R@10", "nDCG@10"]
    print("\nTop 10 by nDCG@5 (v2 gold):")
    print(df[cols].head(10).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df[cols].head(50).to_csv(RESULTS_DIR / "hybrid_weight_tuning_top50.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'hybrid_weight_tuning_top50.csv'}")

    best = df.iloc[0]
    best_weights = {k: float(best[k]) for k in TUNED}
    best_v1 = evaluate_weights(best_weights, per_query, paper_ids, queries, relevant_v1)

    print(f"\nBest weights (v2): {best_weights} (+ recency=0.05, category=0.05)")
    print(f"{'metric':<8} {'default v2':>10} {'best v2':>10} {'delta':>8}   "
          f"{'default v1':>10} {'best v1':>10} {'delta':>8}")
    for m in ("P@5", "R@5", "nDCG@5", "MAP", "MRR"):
        d_v2, b_v2 = baseline_v2[m], best[m]
        d_v1, b_v1 = baseline_v1[m], best_v1[m]
        print(f"{m:<8} {d_v2:>10.4f} {b_v2:>10.4f} {b_v2 - d_v2:>+8.4f}   "
              f"{d_v1:>10.4f} {b_v1:>10.4f} {b_v1 - d_v1:>+8.4f}")

    # Sensitivity: for each tuned component, mean nDCG@5 (v2) across all combos
    # at each weight value -- shows the marginal impact of that retriever's weight.
    fig, axes = plt.subplots(1, len(TUNED), figsize=(13, 4), sharey=True)
    for ax, comp in zip(axes, TUNED):
        grouped = df.groupby(comp)["nDCG@5"].agg(["mean", "std"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker="o",
                     color="#1976D2", ecolor="#90CAF9", capsize=3)
        ax.axvline(DEFAULT_WEIGHTS[comp], color="#9C27B0", linestyle="--", alpha=0.7,
                    label="current default")
        ax.set_title(comp)
        ax.set_xlabel("weight")
    axes[0].set_ylabel("mean nDCG@5 (v2)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hybrid Ranker: marginal impact of each retriever's weight on nDCG@5 (v2 gold)")
    plt.tight_layout()
    out_path = RESULTS_DIR / "hybrid_weight_sensitivity.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
