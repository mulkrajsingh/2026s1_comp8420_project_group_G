"""Evaluate the PDF-NLP pipeline on five real research PDFs.

Loads a fixed manifest and provisional annotation file, runs production parsing
and NLP, and writes per-paper metrics plus aggregate comparison tables.
Annotations are for evaluation only and do not affect inference code paths.
"""

from __future__ import annotations

import csv
import difflib
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from paper_analysis import analyze_parsed_paper
from pdf_parser import build_parsed_paper


def _normalise(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _section_present(parsed: dict[str, Any], expected: str) -> bool:
    aliases = {
        "background": "introduction",
        "experiments": "results",
        "evaluation": "results",
        "discussion": "discussion",
        "related work": "related_work",
        "related_work": "related_work",
        "limitations": "limitations",
    }
    if expected in {"references", "bibliography"}:
        return bool(parsed.get("references"))
    key = aliases.get(expected, expected)
    return bool(str((parsed.get("sections") or {}).get(key) or "").strip())


def _pos_accuracy(tokens: list[dict[str, Any]], gold: list[dict[str, str]]) -> float:
    correct = 0
    for expected in gold:
        candidates = [
            token
            for token in tokens
            if token["text"].lower() == expected["text"].lower()
        ]
        if any(token["pos"] == expected["pos"] for token in candidates):
            correct += 1
    return _safe_div(correct, len(gold))


def _entity_metrics(
    mentions: list[dict[str, Any]],
    gold: list[dict[str, str]],
    track: str,
) -> dict[str, Any]:
    if track == "scier":
        selected = [
            row for row in mentions if row["source"] == "nadiyah_scier_distilbert"
        ]
    elif track == "baseline":
        selected = [
            row
            for row in mentions
            if row["source"] in {"baseline_gazetteer", "metric_regex", "spacy_org"}
        ]
    else:
        selected = mentions
    predicted = {
        (str(row["type"]).lower(), _normalise(str(row["text"])))
        for row in selected
        if str(row["type"]).lower() in {"method", "dataset", "task"}
    }
    expected = {
        (str(row["type"]).lower(), _normalise(str(row["text"])))
        for row in gold
    }

    def matches(left: tuple[str, str], right: tuple[str, str]) -> bool:
        return left[0] == right[0] and (
            left[1] == right[1] or left[1] in right[1] or right[1] in left[1]
        )

    matched_predicted = {
        pred for pred in predicted if any(matches(pred, item) for item in expected)
    }
    matched_expected = {
        item for item in expected if any(matches(pred, item) for pred in predicted)
    }
    precision = _safe_div(len(matched_predicted), len(predicted))
    recall = _safe_div(len(matched_expected), len(expected))
    return {
        "track": track,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "predicted": len(predicted),
        "expected": len(expected),
    }


def _keyphrase_metrics(
    phrases: list[dict[str, Any]],
    concepts: list[str],
) -> tuple[float, float]:
    predicted = [_normalise(row["text"]) for row in phrases[:10]]
    expected = [_normalise(item) for item in concepts]

    def match(left: str, right: str) -> bool:
        return left == right or left in right or right in left

    precision = _safe_div(
        sum(any(match(pred, item) for item in expected) for pred in predicted),
        len(predicted),
    )
    recall = _safe_div(
        sum(any(match(pred, item) for pred in predicted) for item in expected),
        len(expected),
    )
    return precision, recall


def _summary_metrics(
    summary: dict[str, Any],
    concepts: list[str],
    source_text: str,
) -> tuple[float, float]:
    text = _normalise(summary.get("text") or "")
    retention = _safe_div(
        sum(_normalise(concept) in text for concept in concepts),
        len(concepts),
    )
    source_normalised = _normalise(source_text)
    sentences = summary.get("sentences") or []
    traceability = _safe_div(
        sum(_normalise(row["text"]) in source_normalised for row in sentences),
        len(sentences),
    )
    return retention, traceability


def _structural_f1(
    checks: list[dict[str, Any]],
    expected_codes: list[str],
) -> float:
    predicted = {row["code"] for row in checks}
    expected = set(expected_codes)
    matched = predicted & expected
    precision = _safe_div(len(matched), len(predicted))
    recall = _safe_div(len(matched), len(expected))
    return _f1(precision, recall)


def evaluate_real_papers(
    manifest_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> Path:
    """Run production inference and write report-ready metrics and outputs."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    per_paper_dir = output_dir / "per_paper"
    per_paper_dir.mkdir(parents=True, exist_ok=True)

    paper_rows: list[dict[str, Any]] = []
    ner_rows: list[dict[str, Any]] = []
    keyphrase_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for item in manifest.get("papers", []):
        paper_id = item["id"]
        pdf_path = (manifest_path.parent / item["path"]).resolve()
        try:
            if not pdf_path.is_file():
                raise FileNotFoundError(pdf_path)
            if _sha256(pdf_path) != item["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {pdf_path}")
            if pdf_path.stat().st_size != item["bytes"]:
                raise ValueError(f"size mismatch: {pdf_path}")
            parsed, debug = build_parsed_paper(pdf_path)
            enriched, analysis = analyze_parsed_paper(parsed)
            gold = annotations["papers"][paper_id]
            expected_sections = gold.get("expected_sections") or []
            section_recall = _safe_div(
                sum(_section_present(enriched, name) for name in expected_sections),
                len(expected_sections),
            )
            title_similarity = difflib.SequenceMatcher(
                None,
                _normalise(enriched["metadata"].get("title") or ""),
                _normalise(gold["title"]),
            ).ratio()
            pos_accuracy = _pos_accuracy(
                analysis["pos"]["tokens"],
                gold.get("pos") or [],
            )
            structural_f1 = _structural_f1(
                analysis["structural_checks"],
                gold.get("expected_structural_issues") or [],
            )
            paper_rows.append(
                {
                    "paper_id": paper_id,
                    "title_similarity": round(title_similarity, 4),
                    "section_recall": round(section_recall, 4),
                    "reference_count": len(enriched["references"]),
                    "expected_reference_count": gold["reference_count"],
                    "reference_absolute_error": abs(
                        len(enriched["references"]) - gold["reference_count"]
                    ),
                    "pos_accuracy": round(pos_accuracy, 4),
                    "structural_f1": round(structural_f1, 4),
                    "runtime_seconds": round(
                        sum(analysis["timings_seconds"].values()),
                        4,
                    ),
                }
            )
            for track in ("scier", "baseline", "hybrid"):
                row = _entity_metrics(
                    analysis["entity_mentions"],
                    gold.get("entities") or [],
                    track,
                )
                row["paper_id"] = paper_id
                ner_rows.append(row)
            precision, recall = _keyphrase_metrics(
                analysis["keyphrases"],
                gold.get("key_concepts") or [],
            )
            keyphrase_rows.append(
                {
                    "paper_id": paper_id,
                    "precision_at_10": round(precision, 4),
                    "concept_recall": round(recall, 4),
                }
            )
            source_text = "\n".join(
                value
                for value in enriched["sections"].values()
                if isinstance(value, str)
            )
            retention, traceability = _summary_metrics(
                analysis["extractive_summary"],
                gold.get("key_concepts") or [],
                source_text,
            )
            summary_rows.append(
                {
                    "paper_id": paper_id,
                    "concept_retention": round(retention, 4),
                    "source_traceability": round(traceability, 4),
                }
            )
            paper_output = per_paper_dir / f"{paper_id}.json"
            paper_output.write_text(
                json.dumps(
                    {
                        "input": {
                            "paper_id": paper_id,
                            "pdf_sha256": item["sha256"],
                            "source_url": item["source_url"],
                        },
                        "parsed_paper": enriched,
                        "debug_summary": {
                            "page_count": debug["page_count"],
                            "headings": debug["headings"],
                            "warnings": debug["warnings"],
                        },
                    },
                    indent=2,
                    ensure_ascii=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            failures.append(f"- `{paper_id}`: {type(exc).__name__}: {exc}")

    _write_csv(output_dir / "paper_metrics.csv", paper_rows)
    _write_csv(output_dir / "ner_metrics.csv", ner_rows)
    _write_csv(output_dir / "keyphrase_metrics.csv", keyphrase_rows)
    _write_csv(output_dir / "summary_metrics.csv", summary_rows)
    (output_dir / "failure_cases.md").write_text(
        "# Production PDF-NLP Failure Cases\n\n"
        + ("\n".join(failures) if failures else "No runtime failures recorded.\n"),
        encoding="utf-8",
    )

    def avg(rows: list[dict[str, Any]], field: str) -> float:
        values = [float(row[field]) for row in rows]
        return mean(values) if values else 0.0

    ner_track_rows = {
        track: [row for row in ner_rows if row["track"] == track]
        for track in ("scier", "baseline", "hybrid")
    }
    report = output_dir / "comparison_report.md"
    report.write_text(
        "\n".join(
            [
                "# Production PDF-NLP Real-Paper Evaluation",
                "",
                "**Corpus:** five real research-paper PDFs; no mock paper data.",
                "",
                "**Annotations:** provisional review labels, not validated ground truth.",
                "",
                "Results apply only to this local five-paper corpus and must not be generalised.",
                "",
                "## Aggregate Metrics",
                "",
                "| Metric | Result |",
                "| --- | ---: |",
                f"| Successful papers | {len(paper_rows)}/5 |",
                f"| Parser title similarity | {avg(paper_rows, 'title_similarity'):.3f} |",
                f"| Parser section recall | {avg(paper_rows, 'section_recall'):.3f} |",
                f"| POS accuracy | {avg(paper_rows, 'pos_accuracy'):.3f} |",
                f"| SciER NER F1 | {avg(ner_track_rows['scier'], 'f1'):.3f} |",
                f"| Baseline NER F1 | {avg(ner_track_rows['baseline'], 'f1'):.3f} |",
                f"| Hybrid NER F1 | {avg(ner_track_rows['hybrid'], 'f1'):.3f} |",
                f"| Keyphrase Precision@10 | {avg(keyphrase_rows, 'precision_at_10'):.3f} |",
                f"| Keyphrase concept recall | {avg(keyphrase_rows, 'concept_recall'):.3f} |",
                f"| Extractive summary concept retention | {avg(summary_rows, 'concept_retention'):.3f} |",
                f"| Extractive summary source traceability | {avg(summary_rows, 'source_traceability'):.3f} |",
                f"| Structural checklist F1 | {avg(paper_rows, 'structural_f1'):.3f} |",
                "",
                "## Provenance",
                "",
                "- SciER track: Nadiyah's fine-tuned DistilBERT at threshold 0.7.",
                "- Baseline track: deterministic gazetteer, metric regex, and spaCy organisations.",
                "- Hybrid track: union of SciER and baseline mentions.",
                "- Keyphrases: Nadiyah's KeyBERT approach with local MiniLM.",
                "- Summary: deterministic extractive TextRank; BART remains historical comparison only.",
                "",
                f"Runtime failures: {len(failures)}. See `failure_cases.md`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report
