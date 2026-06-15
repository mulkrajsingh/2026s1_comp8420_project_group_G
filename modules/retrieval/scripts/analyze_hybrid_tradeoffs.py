"""Analyse v1 versus v2 trade-offs in hybrid weight tuning.

Re-scores the weight grid on both gold standards to find combos that improve
v2 without regressing v1 metrics.

Usage:
    python modules/retrieval/scripts/analyze_hybrid_tradeoffs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RETRIEVAL_DIR = REPO_ROOT / "modules/retrieval"
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(RETRIEVAL_DIR / "scripts"))

from app.fixtures import EVAL_QUERIES  # noqa: E402
from tune_hybrid_weights import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FIXED_WEIGHTS,
    META_PATH,
    SCORES_NPZ,
    TUNED,
    V2_PATH,
    evaluate_weights,
    grid_combos,
)

RESULTS_DIR = RETRIEVAL_DIR / "results/retrieval"


def main() -> None:
    """Score weight combos on v1 and v2 gold and print balanced candidates."""
    data = np.load(SCORES_NPZ)
    meta = json.loads(META_PATH.read_text())
    paper_ids = np.array(meta["paper_ids"])
    queries = meta["queries"]

    per_query = []
    for qi in range(len(queries)):
        comp = {k: data[f"{k}_{qi}"] for k in (*TUNED, *FIXED_WEIGHTS)}
        per_query.append(comp)

    v2_data = json.loads(V2_PATH.read_text())
    relevant_v2 = {q["query"]: set(q["relevant_ids_v2"]) for q in v2_data}
    relevant_v1 = {q["query"]: set(q["relevant_ids"]) for q in EVAL_QUERIES}

    baseline_v1 = evaluate_weights(DEFAULT_WEIGHTS, per_query, paper_ids, queries, relevant_v1)
    baseline_v2 = evaluate_weights(DEFAULT_WEIGHTS, per_query, paper_ids, queries, relevant_v2)
    print("Default weights:", DEFAULT_WEIGHTS)
    print(f"  v1: P@5={baseline_v1['P@5']:.4f} R@5={baseline_v1['R@5']:.4f} "
          f"nDCG@5={baseline_v1['nDCG@5']:.4f} MAP={baseline_v1['MAP']:.4f} MRR={baseline_v1['MRR']:.4f}")
    print(f"  v2: P@5={baseline_v2['P@5']:.4f} R@5={baseline_v2['R@5']:.4f} "
          f"nDCG@5={baseline_v2['nDCG@5']:.4f} MAP={baseline_v2['MAP']:.4f} MRR={baseline_v2['MRR']:.4f}")

    rows = []
    for combo in grid_combos():
        weights = dict(zip(TUNED, combo))
        m1 = evaluate_weights(weights, per_query, paper_ids, queries, relevant_v1)
        m2 = evaluate_weights(weights, per_query, paper_ids, queries, relevant_v2)
        row = {**weights}
        for k in ("P@5", "R@5", "nDCG@5", "MAP", "MRR"):
            row[f"{k}_v1"] = m1[k]
            row[f"{k}_v2"] = m2[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "hybrid_weight_tuning_v1_v2_full.csv", index=False)
    print(f"\nSaved full {len(df)}-combo grid (v1+v2 metrics) to "
          f"{RESULTS_DIR / 'hybrid_weight_tuning_v1_v2_full.csv'}")

    # Balanced candidates: do not regress v1 MAP or v1 nDCG@5 vs default,
    # while improving v2 nDCG@5 and v2 MAP vs default.
    mask = (
        (df["MAP_v1"] >= baseline_v1["MAP"] - 1e-9)
        & (df["nDCG@5_v1"] >= baseline_v1["nDCG@5"] - 1e-9)
        & (df["nDCG@5_v2"] >= baseline_v2["nDCG@5"] - 1e-9)
        & (df["MAP_v2"] >= baseline_v2["MAP"] - 1e-9)
    )
    balanced = df[mask].copy()
    print(f"\n{len(balanced)}/{len(df)} combos improve-or-match v1 (MAP, nDCG@5) "
          f"AND v2 (MAP, nDCG@5) simultaneously vs. default.")
    if len(balanced):
        balanced["combined"] = balanced["nDCG@5_v1"] + balanced["nDCG@5_v2"] + balanced["MAP_v1"] + balanced["MAP_v2"]
        balanced = balanced.sort_values("combined", ascending=False)
        cols = [*TUNED, "P@5_v1", "R@5_v1", "nDCG@5_v1", "MAP_v1",
                "P@5_v2", "R@5_v2", "nDCG@5_v2", "MAP_v2"]
        print(balanced[cols].head(10).to_string(index=False))
        balanced[cols].head(20).to_csv(RESULTS_DIR / "hybrid_weight_balanced_candidates.csv", index=False)
        print(f"Saved {RESULTS_DIR / 'hybrid_weight_balanced_candidates.csv'}")

    # Slightly relaxed: allow up to 2% relative regression on v1 MAP/nDCG@5
    tol = 0.02
    mask2 = (
        (df["MAP_v1"] >= baseline_v1["MAP"] * (1 - tol))
        & (df["nDCG@5_v1"] >= baseline_v1["nDCG@5"] * (1 - tol))
    )
    relaxed = df[mask2].copy().sort_values("nDCG@5_v2", ascending=False)
    print(f"\nTop combos allowing <={tol:.0%} v1 regression, sorted by v2 nDCG@5:")
    cols = [*TUNED, "P@5_v1", "R@5_v1", "nDCG@5_v1", "MAP_v1",
            "P@5_v2", "R@5_v2", "nDCG@5_v2", "MAP_v2"]
    print(relaxed[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
