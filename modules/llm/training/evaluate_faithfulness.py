"""Audit generated outputs for source-ID and citation-grounding safety."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.faithfulness import check_generation  # noqa: E402
from app.io_utils import read_jsonl, write_json  # noqa: E402


def _prompt_pack_by_id(test_set: Path) -> dict[str, dict[str, Any]]:
    return {
        record["prompt_id"]: record["input"]["rag_evidence_pack"]
        for record in read_jsonl(test_set)
    }


def main() -> None:
    """Write per-output faithfulness CSV and summary JSON for model generations."""
    parser = argparse.ArgumentParser(description="Evaluate evidence grounding for generated outputs.")
    parser.add_argument("--generations", default="results/model_comparison/model_generations.jsonl")
    parser.add_argument("--test-set", default="data/eval/fixed_prompts.jsonl")
    parser.add_argument("--out", default="results/model_comparison/faithfulness_audit.csv")
    parser.add_argument("--summary", default="results/model_comparison/faithfulness_audit_summary.json")
    args = parser.parse_args()

    packs = _prompt_pack_by_id(Path(args.test_set))
    rows: list[dict[str, Any]] = []
    for generation in read_jsonl(Path(args.generations)):
        prompt_id = generation["prompt_id"]
        result = check_generation(generation.get("text", ""), packs[prompt_id])
        rows.append(
            {
                "prompt_id": prompt_id,
                "task": generation["task"],
                "backend": generation["backend"],
                "model": generation["model"],
                "strategy": generation["strategy"],
                "passes_basic_faithfulness": result["passes_basic_faithfulness"],
                "used_source_ids": ",".join(result["used_source_ids"]),
                "unsupported_source_ids": ",".join(result["unsupported_source_ids"]),
                "metadata_citations_found": result["metadata_citations_found"],
                "error": generation.get("error") or "",
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for row in rows if row["passes_basic_faithfulness"] == "True" or row["passes_basic_faithfulness"] is True)
    write_json(
        Path(args.summary),
        {
            "total_outputs": total,
            "passed_basic_faithfulness": passed,
            "pass_rate": round(passed / total, 3) if total else 0.0,
            "caveat": "Source-id checks do not replace human semantic faithfulness review.",
        },
    )
    print(f"Wrote faithfulness audit: {out_path}")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()

