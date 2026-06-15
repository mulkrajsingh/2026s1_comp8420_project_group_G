#!/usr/bin/env python3
"""Generate EDA artifacts for the LoRA training JSONL."""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/final_dataset/research_lora_train.jsonl"
OUT = ROOT / "results/lora_dataset_eda"
FIG = OUT / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(OUT / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd


SOURCE_COLORS = {
    "project_arxiv_rag": "#2457C5",
    "researchqa": "#0E8A83",
    "qasper": "#2D7D46",
    "peerread": "#C66A1A",
    "scicite": "#6A4CC2",
    "scitldr": "#B23A3A",
    "local_fixed_prompt": "#5B6673",
}

SOURCE_LABELS = {
    "project_arxiv_rag": "Project arXiv RAG",
    "researchqa": "ResearchQA",
    "qasper": "QASPER",
    "peerread": "PeerRead",
    "scicite": "SciCite",
    "scitldr": "SciTLDR",
    "local_fixed_prompt": "Local fixed prompts",
}


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    return int(pd.Series(values).quantile(q, interpolation="nearest"))


def safe_task_name(task: str) -> str:
    return task.replace("_", " ")


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    bad_json = 0
    bad_roles = 0
    for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            bad_json += 1
            raise ValueError(f"Malformed JSON on line {line_no}: {exc}") from exc

        roles = [m.get("role") for m in row.get("messages", [])]
        if roles != ["system", "user", "assistant"]:
            bad_roles += 1
        rows.append(row)

    if bad_json or bad_roles:
        raise ValueError(f"Dataset validation failed: bad_json={bad_json}, bad_roles={bad_roles}")
    return rows


def collect(rows: list[dict]) -> dict:
    source_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    source_task: dict[str, Counter[str]] = defaultdict(Counter)
    user_words: list[int] = []
    assistant_words: list[int] = []
    total_words: list[int] = []
    source_id_counts: list[int] = []
    rows_with_evidence_passages = 0
    rows_with_source_ids = 0

    for row in rows:
        source = row.get("source", "<missing>")
        task = row.get("task", "<missing>")
        messages = row["messages"]
        user = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        assistant = "\n".join(m.get("content", "") for m in messages if m.get("role") == "assistant")
        all_text = "\n".join(m.get("content", "") for m in messages)
        source_ids = set(re.findall(r"\[S\d+\]", all_text))

        source_counts[source] += 1
        task_counts[task] += 1
        source_task[source][task] += 1
        user_words.append(word_count(user))
        assistant_words.append(word_count(assistant))
        total_words.append(word_count(all_text))
        source_id_counts.append(len(source_ids))
        rows_with_evidence_passages += int("Evidence passages:" in user)
        rows_with_source_ids += int(bool(source_ids))

    return {
        "source_counts": source_counts,
        "task_counts": task_counts,
        "source_task": source_task,
        "user_words": user_words,
        "assistant_words": assistant_words,
        "total_words": total_words,
        "source_id_counts": source_id_counts,
        "rows_with_evidence_passages": rows_with_evidence_passages,
        "rows_with_source_ids": rows_with_source_ids,
    }


def length_summary(label: str, values: list[int]) -> dict[str, int | float | str]:
    return {
        "field": label,
        "min": min(values),
        "p25": percentile(values, 0.25),
        "median": median(values),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "mean": round(mean(values), 1),
    }


def write_csvs(stats: dict) -> None:
    with (OUT / "source_task_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "task", "rows"])
        for source, tasks in sorted(stats["source_task"].items()):
            for task, count in sorted(tasks.items()):
                writer.writerow([source, task, count])

    with (OUT / "length_stats.csv").open("w", newline="", encoding="utf-8") as f:
        rows = [
            length_summary("user_words", stats["user_words"]),
            length_summary("assistant_words", stats["assistant_words"]),
            length_summary("total_words", stats["total_words"]),
            length_summary("source_id_markers", stats["source_id_counts"]),
        ]
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#D7DEE8")
    ax.tick_params(colors="#5B6673", labelsize=10)
    ax.grid(axis="x", color="#E6EAF0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_source_mix(source_counts: Counter[str]) -> None:
    items = source_counts.most_common()
    labels = [SOURCE_LABELS.get(k, k) for k, _ in items]
    counts = [v for _, v in items]
    colors = [SOURCE_COLORS.get(k, "#2457C5") for k, _ in items]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=180)
    ax.barh(range(len(labels)), counts, color=colors)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("LoRA training source mix", loc="left", fontsize=16, fontweight="bold", color="#17202A")
    style_axes(ax)
    for i, count in enumerate(counts):
        ax.text(count + 80, i, f"{count:,}", va="center", fontsize=9, color="#17202A")
    fig.tight_layout()
    fig.savefig(FIG / "source_mix.png", bbox_inches="tight")
    plt.close(fig)


def plot_task_mix(task_counts: Counter[str]) -> None:
    items = task_counts.most_common(12)
    labels = [safe_task_name(k) for k, _ in items]
    counts = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.barh(range(len(labels)), counts, color="#0E8A83")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Rows")
    ax.set_title("Top LoRA task families", loc="left", fontsize=14, fontweight="bold", color="#17202A")
    style_axes(ax)
    for i, count in enumerate(counts):
        ax.text(count + 60, i, f"{count:,}", va="center", fontsize=9, color="#17202A")
    fig.tight_layout()
    fig.savefig(FIG / "task_mix.png", bbox_inches="tight")
    plt.close(fig)


def plot_lengths(stats: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=180)
    ax.hist(stats["user_words"], bins=40, range=(0, 2800), color="#2457C5", alpha=0.78, label="Prompt")
    ax.hist(stats["assistant_words"], bins=40, range=(0, 2800), color="#C66A1A", alpha=0.7, label="Answer")
    ax.axvline(median(stats["total_words"]), color="#17202A", linestyle="--", linewidth=1.4, label="Total median")
    ax.axvline(percentile(stats["total_words"], 0.95), color="#B23A3A", linestyle=":", linewidth=1.8, label="Total p95")
    ax.set_xlabel("Words per row")
    ax.set_ylabel("Rows")
    ax.set_title("Prompt and answer length distribution", loc="left", fontsize=14, fontweight="bold", color="#17202A")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "length_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_evidence(stats: dict, total: int) -> None:
    labels = ["Rows with [S#]\nsource IDs", "Rows with evidence\npassages"]
    values = [stats["rows_with_source_ids"], stats["rows_with_evidence_passages"]]
    colors = ["#2D7D46", "#2457C5"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, total)
    ax.set_ylabel("Rows")
    ax.set_title("Evidence-grounding coverage", loc="left", fontsize=14, fontweight="bold", color="#17202A")
    style_axes(ax)
    ax.grid(axis="y", color="#E6EAF0", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.025,
            f"{value:,}\n({value / total:.1%})",
            ha="center",
            fontsize=10,
            color="#17202A",
        )
    fig.tight_layout()
    fig.savefig(FIG / "evidence_coverage.png", bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict], stats: dict) -> None:
    total = len(rows)
    source_lines = "\n".join(
        f"| `{source}` | {count:,} | {count / total:.1%} |"
        for source, count in stats["source_counts"].most_common()
    )
    task_lines = "\n".join(
        f"| `{task}` | {count:,} |"
        for task, count in stats["task_counts"].most_common()
    )
    total_median = int(median(stats["total_words"]))
    total_p95 = percentile(stats["total_words"], 0.95)
    project_aligned = stats["source_counts"]["project_arxiv_rag"] + stats["source_counts"]["local_fixed_prompt"]

    summary = f"""# LoRA Training Dataset EDA

Generated from `data/processed/final_dataset/research_lora_train.jsonl`.

## Validation

- Rows parsed: {total:,}
- Malformed JSON rows: 0
- Rows with required `system`, `user`, `assistant` message roles: {total:,}
- Distinct sources: {len(stats["source_counts"])}
- Distinct task labels: {len(stats["task_counts"])}

## Source Mix

| Source | Rows | Share |
| --- | ---: | ---: |
{source_lines}

## Task Mix

| Task | Rows |
| --- | ---: |
{task_lines}

## Length And Evidence Coverage

- Median total row length: {total_median:,} words.
- P95 total row length: {total_p95:,} words, driven by longer project RAG prompts.
- Rows with `[S#]` source-ID markers: {stats["rows_with_source_ids"]:,} ({stats["rows_with_source_ids"] / total:.1%}).
- Rows with explicit evidence-passage blocks: {stats["rows_with_evidence_passages"]:,} ({stats["rows_with_evidence_passages"] / total:.1%}).
- Project-aligned rows: {project_aligned:,} (`project_arxiv_rag` plus `local_fixed_prompt`).

## PPT Readout

The dataset combines broad academic supervision with project-specific RAG behavior. This supports a careful claim that the adapter training data is intentionally mixed for research-assistant structure, evidence grounding, citation behavior, and scientific summarization. It does not by itself prove adapter quality; model-quality claims still require real training and evaluation logs.
"""
    (OUT / "lora_dataset_eda_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows(DATASET)
    stats = collect(rows)
    write_csvs(stats)
    plot_source_mix(stats["source_counts"])
    plot_task_mix(stats["task_counts"])
    plot_lengths(stats)
    plot_evidence(stats, len(rows))
    write_summary(rows, stats)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
