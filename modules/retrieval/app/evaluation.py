"""Retrieval evaluation metrics: Precision@K, Recall@K, F1@K, MRR, MAP, nDCG.

Gold labels are provided as sets of relevant paper_ids per query.
All metrics follow standard IR definitions.
"""

from __future__ import annotations

import math

import pandas as pd


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    return sum(1 for pid in top_k if pid in relevant) / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / len(relevant)


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits, total_ap = 0, 0.0
    for i, pid in enumerate(retrieved, start=1):
        if pid in relevant:
            hits += 1
            total_ap += hits / i
    return total_ap / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary relevance nDCG (relevant = 1, not relevant = 0)."""
    def dcg(items):
        return sum(
            (1.0 / math.log2(i + 2)) for i, pid in enumerate(items) if pid in relevant
        )

    actual_dcg = dcg(retrieved[:k])
    ideal_dcg = dcg(sorted(retrieved[:k], key=lambda pid: pid in relevant, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Query-level evaluation
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever,
    queries: list[dict],
    k_list: list[int] | None = None,
) -> pd.DataFrame:
    """Run evaluation for one retriever across all queries.

    queries: list of {"query": str, "relevant_ids": list[str]}
    Returns a DataFrame with one row per query and metric columns.
    """
    if k_list is None:
        k_list = [5, 10]

    rows = []
    for q in queries:
        text = q["query"]
        relevant = set(q["relevant_ids"])
        results = retriever.retrieve(text, top_k=max(k_list))
        retrieved_ids = [paper["paper_id"] for paper, _ in results]

        row = {"query": text, "n_relevant": len(relevant)}
        for k in k_list:
            row[f"P@{k}"] = round(precision_at_k(retrieved_ids, relevant, k), 4)
            row[f"R@{k}"] = round(recall_at_k(retrieved_ids, relevant, k), 4)
            row[f"F1@{k}"] = round(f1_at_k(retrieved_ids, relevant, k), 4)
            row[f"nDCG@{k}"] = round(ndcg_at_k(retrieved_ids, relevant, k), 4)
        row["MRR"] = round(reciprocal_rank(retrieved_ids, relevant), 4)
        row["MAP"] = round(average_precision(retrieved_ids, relevant), 4)
        rows.append(row)

    return pd.DataFrame(rows)


def compare_retrievers(
    retrievers: list,
    queries: list[dict],
    k_list: list[int] | None = None,
) -> pd.DataFrame:
    """Return a summary comparison table (mean metrics) for multiple retrievers."""
    if k_list is None:
        k_list = [5, 10]

    summary_rows = []
    for retriever in retrievers:
        df = evaluate_retriever(retriever, queries, k_list=k_list)
        row = {"retriever": retriever.name}
        for col in df.columns:
            if col not in ("query", "n_relevant"):
                row[col] = round(df[col].mean(), 4)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).set_index("retriever")
