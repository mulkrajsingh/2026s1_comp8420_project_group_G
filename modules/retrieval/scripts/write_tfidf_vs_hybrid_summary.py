"""Write a TF-IDF versus hybrid_rrf comparison summary from evaluation CSVs.

Reads on-disk retrieval comparison tables and emits a short markdown report
highlighting precision and nDCG gaps between the lexical and production rankers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _format_pair(frame: pd.DataFrame, metric: str = "P@5") -> str:
    subset = frame[frame["retriever"].isin(["tfidf", "hybrid_rrf"])].copy()
    subset = subset.sort_values(metric, ascending=False)
    lines = [
        "| Retriever | P@5 | nDCG@5 | MAP |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in subset.iterrows():
        lines.append(
            f"| {row['retriever']} | {row['P@5']:.3f} | "
            f"{row['nDCG@5']:.3f} | {row['MAP']:.3f} |"
        )
    return "\n".join(lines)


def write_summary(out_dir: Path) -> Path:
    keyword_path = out_dir / "retrieval_comparison_keyword.csv"
    user_path = out_dir / "retrieval_comparison_user.csv"
    if not keyword_path.is_file() or not user_path.is_file():
        raise FileNotFoundError(
            "Run evaluate-retrieval first to produce keyword and user-like CSVs."
        )

    keyword = pd.read_csv(keyword_path)
    user = pd.read_csv(user_path)
    lines = [
        "# TF-IDF vs Hybrid RRF Comparison",
        "",
        "Head-to-head comparison on the same 5,000-paper corpus. Full multi-retriever",
        "tables remain in `retrieval_comparison_keyword.csv` and",
        "`retrieval_comparison_user.csv`.",
        "",
        "## Keyword benchmark (5 legacy keyword-style queries)",
        "",
        _format_pair(keyword),
        "",
        "TF-IDF leads on this short keyword set.",
        "",
        "## User-like benchmark (10 natural-language queries)",
        "",
        _format_pair(user),
        "",
        "Hybrid RRF (BM25 + SPECTER2 reciprocal rank fusion) leads on user-like",
        "frontend topic phrasing.",
        "",
        "## Production default",
        "",
        "The integration UI defaults to `hybrid_rrf` to satisfy hybrid-search",
        "requirements and better match natural-language topic queries. TF-IDF remains",
        "available as a lexical baseline for controlled comparison.",
        "",
        "Do not generalise either benchmark beyond its query set.",
        "",
    ]
    out_path = out_dir / "tfidf_vs_hybrid_rrf.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    """Parse CLI arguments and write the TF-IDF versus hybrid summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/retrieval",
        help="Directory containing retrieval_comparison_*.csv files.",
    )
    args = parser.parse_args()
    out_path = write_summary(Path(args.out))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
