"""CLI entry point for the retrieval module.

Commands
--------
build-retrieval-index   Build and save TF-IDF/BM25/embedding indexes.
recommend-topic         Retrieve + rank recommendations for a topic query.
evaluate-retrieval      Run Precision@K, Recall@K, MRR, nDCG comparison table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fixtures import EVAL_QUERIES, load_eval_queries, load_papers
from .session_append import append_session_event
from .io_utils import write_json
from .rag_pack import build_rag_evidence_pack
from .retrieval.embeddings import file_sha256, matching_saved_index
from .citation import format_apa
from .retrieval.tfidf_bm25 import BM25Retriever, TfidfRetriever

RETRIEVAL_STRATEGIES = ("hybrid_rrf", "tfidf")

DEFAULT_INDEX_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "retrieval_index"
)


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_build_retrieval_index(args: argparse.Namespace) -> None:
    papers_path = Path(args.papers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading papers from {papers_path}...")
    papers = load_papers(papers_path)
    print(f"  {len(papers)} papers loaded.")

    # TF-IDF index (always built — no heavy dependencies)
    print("Building TF-IDF index...")
    tfidf = TfidfRetriever()
    tfidf.fit(papers)
    # Persist vocab
    vocab = tfidf.vectorizer.vocabulary_
    write_json(out_dir / "tfidf_vocab.json", {k: int(v) for k, v in vocab.items()})
    print(f"  TF-IDF vocab size: {len(vocab)}")

    # BM25 index (tokenised corpus)
    print("Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.fit(papers)
    print("  BM25 index ready.")

    # Embedding index (SPECTER2 or fallback)
    if not args.skip_embeddings:
        try:
            from .retrieval.embeddings import EmbeddingRetriever
            print("Building embedding index (this may take a moment)...")
            emb = EmbeddingRetriever()
            emb.fit(papers, show_progress=True)
            emb.save(out_dir / "embeddings")
            print(f"  Embedding index saved to {out_dir / 'embeddings'} (model: {emb.model_name})")
        except Exception as exc:
            print(f"  Warning: embedding index skipped — {exc}")

    # The source corpus remains canonical in modules/dataset. The index only
    # stores derived data and enough metadata to validate its row count.
    write_json(
        out_dir / "index_meta.json",
        {
            "n_papers": len(papers),
            "tfidf_vocab_size": len(vocab),
            "papers_sha256": file_sha256(papers_path),
        },
    )
    print(f"\nRetrieval index saved to: {out_dir}")


def _tfidf_recommendations(
    papers: list[dict],
    query: str,
    *,
    top_k: int,
) -> list[dict]:
    """Map TF-IDF retrieve tuples into the recommendation JSON shape."""
    ranked = TfidfRetriever().fit(papers).retrieve(query, top_k=top_k)
    recommendations = []
    for paper, score in ranked:
        recommendations.append(
            {
                "paper": paper,
                "score": round(score, 4),
                "reason": "TF-IDF cosine similarity over title and abstract.",
                "evidence": [paper["paper_id"]],
                "apa_citation": format_apa(paper),
                "relation": "same_topic",
            }
        )
    return recommendations


def _hybrid_rrf_recommendations(
    papers: list[dict],
    papers_path: Path,
    query: str,
    *,
    top_k: int,
    embedding_model: str | None,
) -> list[dict]:
    from .retrieval.embeddings import EmbeddingRetriever
    from .retrieval.hybrid_ranker import HybridRRFRanker

    print("Building BM25 index...")
    bm25 = BM25Retriever().fit(papers)
    index_model = matching_saved_index(
        DEFAULT_INDEX_DIR,
        papers_path,
        embedding_model,
    )
    if index_model:
        print(f"Loading saved {index_model} embeddings...")
        emb_retriever = EmbeddingRetriever(index_model)
        emb_retriever.load_embeddings(
            DEFAULT_INDEX_DIR / "embeddings",
            papers,
        )
    else:
        print("Building embeddings for this corpus...")
        emb_retriever = EmbeddingRetriever(embedding_model)
        emb_retriever.fit(papers, show_progress=True)

    ranker = HybridRRFRanker(
        bm25_retriever=bm25,
        embedding_retriever=emb_retriever,
    )
    ranker._papers = papers

    print(f"\nRanking top {top_k} recommendations (BM25 + SPECTER2 RRF)...")
    return ranker.recommend(query, top_k=top_k)


def cmd_recommend_topic(args: argparse.Namespace) -> None:
    papers_path = Path(args.papers)
    out_path = Path(args.out)
    query = args.query
    strategy = args.retrieval_strategy

    append_session_event(
        "user_input",
        {
            "command": "recommend-topic",
            "query": query,
            "papers": str(papers_path),
            "out": str(out_path),
            "retrieval_strategy": strategy,
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
        },
    )

    print(f"Query: {query!r}")
    papers = load_papers(papers_path)
    print(f"Corpus: {len(papers)} papers")

    if strategy == "tfidf":
        print(f"\nRanking top {args.top_k} recommendations (TF-IDF)...")
        recommendations = _tfidf_recommendations(
            papers,
            query,
            top_k=args.top_k,
        )
    else:
        recommendations = _hybrid_rrf_recommendations(
            papers,
            papers_path,
            query,
            top_k=args.top_k,
            embedding_model=args.embedding_model,
        )

    # Write recommendations
    write_json(out_path, recommendations)
    print(f"Recommendations: {out_path}")

    # Write RAG evidence pack
    pack_path = out_path.parent / "rag_evidence_pack.json"
    pack = build_rag_evidence_pack(query, recommendations, retrieval_mode="offline")
    write_json(pack_path, pack)
    print(f"RAG evidence pack: {pack_path}")

    append_session_event(
        "retrieval",
        {
            "query": query,
            "retrieval_mode": "offline",
            "retrieval_strategy": strategy,
            "top_papers": [
                {
                    "title": rec["paper"].get("title"),
                    "score": rec.get("score"),
                    "apa_citation": rec.get("apa_citation"),
                }
                for rec in recommendations[:5]
            ],
        },
    )
    append_session_event(
        "user_output",
        {
            "artifact_paths": [str(out_path), str(pack_path)],
            "recommended_count": len(recommendations),
            "citation_count": sum(1 for r in recommendations if r.get("apa_citation")),
        },
    )

    # Print top 5 for quick review
    print("\nTop 5 recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. [{rec['score']:.3f}] {rec['paper']['title']}")
        print(f"       {rec['apa_citation']}")


def _build_eval_retrievers(papers: list[dict]):
    retrievers = [
        TfidfRetriever().fit(papers),
        BM25Retriever().fit(papers),
    ]

    try:
        from .retrieval.embeddings import EmbeddingRetriever
        from .retrieval.hybrid_ranker import HybridRanker, HybridRRFRanker
        from .retrieval.section_aware import SectionAwareRetriever

        emb = EmbeddingRetriever()
        emb.fit(papers, show_progress=True)
        bm25 = BM25Retriever().fit(papers)
        hybrid_rrf = HybridRRFRanker(
            bm25_retriever=bm25,
            embedding_retriever=emb,
        )
        hybrid_rrf._papers = papers
        section = SectionAwareRetriever(emb)
        section.fit(papers, show_progress=False)
        hybrid = HybridRanker(
            tfidf_retriever=TfidfRetriever().fit(papers),
            bm25_retriever=BM25Retriever().fit(papers),
            embedding_retriever=emb,
            section_aware_retriever=section,
        )
        hybrid._papers = papers
        retrievers = [hybrid_rrf] + retrievers + [emb, section, hybrid]
    except Exception as exc:
        print(f"  Warning: advanced retrievers skipped — {exc}")
    return retrievers


def _run_eval_suite(
    retrievers,
    queries: list[dict],
    out_dir: Path,
    *,
    label: str,
) -> None:
    from .evaluation import compare_retrievers, evaluate_retriever

    print(f"Evaluating {label} set on {len(queries)} queries...")
    comparison = compare_retrievers(retrievers, queries, k_list=[5, 10])
    print(f"\n=== Retrieval Comparison Table ({label}) ===")
    print(comparison.to_string())

    csv_name = f"retrieval_comparison_{label}.csv"
    comparison.to_csv(out_dir / csv_name)
    write_json(
        out_dir / f"retrieval_comparison_{label}.json",
        json.loads(comparison.to_json()),
    )
    print(f"\nSaved {label} results to {out_dir / csv_name}")

    for retriever in retrievers:
        df = evaluate_retriever(retriever, queries, k_list=[5, 10])
        df.to_csv(out_dir / f"detail_{label}_{retriever.name}.csv", index=False)


def cmd_evaluate_retrieval(args: argparse.Namespace) -> None:
    papers_path = Path(args.papers)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = load_papers(papers_path)
    print(f"Loaded {len(papers)} papers from {papers_path}")
    retrievers = _build_eval_retrievers(papers)
    query_set = args.query_set
    if query_set in {"keyword", "all"}:
        _run_eval_suite(retrievers, EVAL_QUERIES, out_dir, label="keyword")
    if query_set in {"user", "all"}:
        _run_eval_suite(retrievers, load_eval_queries("user"), out_dir, label="user")
    if query_set == "all":
        combined = load_eval_queries("all")
        _run_eval_suite(retrievers, combined, out_dir, label="combined")
        _write_eval_summary(out_dir)


def _write_eval_summary(out_dir: Path) -> None:
    import pandas as pd

    keyword_path = out_dir / "retrieval_comparison_keyword.csv"
    user_path = out_dir / "retrieval_comparison_user.csv"
    if not keyword_path.is_file() or not user_path.is_file():
        return

    keyword = pd.read_csv(keyword_path)
    user = pd.read_csv(user_path)
    lines = [
        "# Retrieval Evaluation Summary",
        "",
        "Two benchmarks are reported on the same 5,000-paper corpus:",
        "",
        "1. **Keyword benchmark** (`retrieval_comparison_keyword.csv`): five legacy",
        "   keyword-style queries used for baseline tuning.",
        "2. **User-like benchmark** (`retrieval_comparison_user.csv`): ten",
        "   natural-language queries with manually reviewed gold labels.",
        "",
        "Do not generalise either table beyond its query set.",
        "",
        "## Keyword benchmark (P@5)",
        "",
        _format_top_rows(keyword, "P@5"),
        "",
        "## User-like benchmark (P@5)",
        "",
        _format_top_rows(user, "P@5"),
        "",
        "## Production choice",
        "",
        "Production retrieval uses BM25 + SPECTER2 reciprocal rank fusion to match",
        "assignment hybrid-search requirements and real user phrasing. TF-IDF remains",
        "a lexical baseline on the keyword benchmark; the user-like benchmark is the",
        "better proxy for frontend topic queries.",
        "",
    ]
    (out_dir / "retrieval_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _format_top_rows(frame, metric: str) -> str:
    ordered = frame.sort_values(metric, ascending=False)
    rows = ["| Retriever | P@5 | nDCG@5 | MAP |", "| --- | ---: | ---: | ---: |"]
    for _, row in ordered.iterrows():
        rows.append(
            f"| {row['retriever']} | {row['P@5']:.3f} | "
            f"{row['nDCG@5']:.3f} | {row['MAP']:.3f} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Retrieval module: RAG, recommendation, and evaluation commands.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # build-retrieval-index
    idx = sub.add_parser("build-retrieval-index", help="Build offline retrieval indexes.")
    idx.add_argument("--papers", required=True, help="Path to PaperRecord JSONL.")
    idx.add_argument("--out", required=True, help="Output directory for index artifacts.")
    idx.add_argument("--skip-embeddings", action="store_true", help="Skip embedding model (faster).")
    idx.set_defaults(func=cmd_build_retrieval_index)

    # recommend-topic
    rec = sub.add_parser("recommend-topic", help="Retrieve and rank recommendations for a query.")
    rec.add_argument("--query", required=True, help="Topic query string.")
    rec.add_argument("--papers", required=True, help="Path to PaperRecord JSONL.")
    rec.add_argument("--out", required=True, help="Output path for recommendations JSON.")
    rec.add_argument("--top-k", type=int, default=10, help="Number of recommendations.")
    rec.add_argument(
        "--retrieval-strategy",
        choices=RETRIEVAL_STRATEGIES,
        default="hybrid_rrf",
        help="Ranking strategy: hybrid_rrf (BM25 + dense RRF) or tfidf baseline.",
    )
    rec.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Exact sentence-transformer model for hybrid RRF retrieval. "
            "The command fails rather than substituting another model."
        ),
    )
    rec.set_defaults(func=cmd_recommend_topic)

    # evaluate-retrieval
    ev = sub.add_parser("evaluate-retrieval", help="Run retrieval evaluation and save comparison table.")
    ev.add_argument("--papers", required=True, help="Path to PaperRecord JSONL.")
    ev.add_argument("--out", required=True, help="Output directory for evaluation results.")
    ev.add_argument(
        "--query-set",
        choices=("keyword", "user", "all"),
        default="all",
        help="Evaluate keyword queries, user-like queries, or both.",
    )
    ev.set_defaults(func=cmd_evaluate_retrieval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
