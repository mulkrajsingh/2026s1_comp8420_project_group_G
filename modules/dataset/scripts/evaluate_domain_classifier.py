"""Evaluate TF-IDF baselines for arXiv category classification.

Trains logistic regression and linear SVM on title-plus-abstract text, scores
a stratified held-out split, and writes confusion matrices plus comparison
metrics. Labels come from the first matching target category on each record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from scripts.build_balanced_corpus import TARGET_CATEGORIES
except ModuleNotFoundError:
    from build_balanced_corpus import TARGET_CATEGORIES


def _primary_category(categories: list[str]) -> str | None:
    category_set = set(categories)
    return next(
        (category for category in TARGET_CATEGORIES if category in category_set),
        None,
    )


def _load_records(corpus_path: Path) -> tuple[list[str], list[str], dict[str, int]]:
    texts: list[str] = []
    labels: list[str] = []
    paper_ids: set[str] = set()
    duplicate_ids = 0

    with corpus_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            categories = row.get("categories") or []
            if not isinstance(categories, list):
                categories = str(categories).split()
            label = _primary_category([str(value) for value in categories])
            title = str(row.get("title") or "").strip()
            abstract = str(row.get("abstract") or "").strip()
            paper_id = str(row.get("paper_id") or row.get("arxiv_id") or "").strip()
            if label is None or not title or not abstract:
                continue
            if paper_id and paper_id in paper_ids:
                duplicate_ids += 1
                continue
            if paper_id:
                paper_ids.add(paper_id)
            texts.append(f"{title}. {abstract}")
            labels.append(label)

    counts = Counter(labels)
    missing = [category for category in TARGET_CATEGORIES if counts[category] < 5]
    if missing:
        raise ValueError(
            "Each target category needs at least five records; "
            f"insufficient categories: {missing}"
        )
    return texts, labels, {"duplicate_paper_ids_removed": duplicate_ids}


def _write_confusion_csv(
    path: Path,
    matrix: Any,
    labels: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *[int(value) for value in row]])


def _write_confusion_plot(
    path: Path,
    matrix: Any,
    labels: list[str],
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted category",
        ylabel="Actual category",
        title="TF-IDF + Logistic Regression Confusion Matrix",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_domain_classifier(
    corpus_path: Path,
    output_dir: Path,
    *,
    test_size: float = 0.2,
    seed: int = 8420,
    max_features: int = 30_000,
) -> dict[str, Any]:
    """Train and evaluate the baseline on a stratified held-out split."""
    texts, labels, data_checks = _load_records(corpus_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    ordered_labels = list(TARGET_CATEGORIES)
    matrix = confusion_matrix(y_test, predictions, labels=ordered_labels)
    per_class = classification_report(
        y_test,
        predictions,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )
    svm_pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=max_features,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    svm_pipeline.fit(x_train, y_train)
    svm_predictions = svm_pipeline.predict(x_test)
    svm_matrix = confusion_matrix(
        y_test,
        svm_predictions,
        labels=ordered_labels,
    )
    svm_per_class = classification_report(
        y_test,
        svm_predictions,
        labels=ordered_labels,
        output_dict=True,
        zero_division=0,
    )

    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    classifier: LogisticRegression = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    top_terms = {}
    for class_index, label in enumerate(classifier.classes_):
        indices = classifier.coef_[class_index].argsort()[-12:][::-1]
        top_terms[str(label)] = [str(feature_names[index]) for index in indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion_csv = output_dir / "confusion_matrix.csv"
    confusion_png = output_dir / "confusion_matrix.png"
    svm_confusion_csv = output_dir / "linear_svm_confusion_matrix.csv"
    svm_confusion_png = output_dir / "linear_svm_confusion_matrix.png"
    _write_confusion_csv(confusion_csv, matrix, ordered_labels)
    _write_confusion_plot(confusion_png, matrix, ordered_labels)
    _write_confusion_csv(svm_confusion_csv, svm_matrix, ordered_labels)
    _write_confusion_plot(svm_confusion_png, svm_matrix, ordered_labels)

    logistic_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "weighted_f1": f1_score(y_test, predictions, average="weighted"),
        "per_class": {
            label: per_class[label]
            for label in ordered_labels
        },
    }
    svm_metrics = {
        "accuracy": accuracy_score(y_test, svm_predictions),
        "macro_f1": f1_score(y_test, svm_predictions, average="macro"),
        "weighted_f1": f1_score(y_test, svm_predictions, average="weighted"),
        "per_class": {
            label: svm_per_class[label]
            for label in ordered_labels
        },
    }
    selected_model = max(
        (
            ("logistic_regression", logistic_metrics),
            ("linear_svm", svm_metrics),
        ),
        key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]),
    )[0]
    comparison_csv = output_dir / "model_comparison.csv"
    with comparison_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "accuracy", "macro_f1", "weighted_f1"),
            lineterminator="\n",
        )
        writer.writeheader()
        for model_name, metrics in (
            ("logistic_regression", logistic_metrics),
            ("linear_svm", svm_metrics),
        ):
            writer.writerow(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                }
            )

    label_counts = Counter(labels)
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    majority_baseline = max(test_counts.values()) / len(y_test)
    result = {
        "technique": "TF-IDF (1-2 grams) + logistic regression / Linear SVM",
        "corpus": str(corpus_path),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "seed": seed,
        "test_size": test_size,
        "records": len(labels),
        "train_records": len(y_train),
        "test_records": len(y_test),
        "label_counts": dict(sorted(label_counts.items())),
        "train_label_counts": dict(sorted(train_counts.items())),
        "test_label_counts": dict(sorted(test_counts.items())),
        "accuracy": logistic_metrics["accuracy"],
        "macro_f1": logistic_metrics["macro_f1"],
        "weighted_f1": logistic_metrics["weighted_f1"],
        "majority_baseline_accuracy": majority_baseline,
        "model_comparison": {
            "logistic_regression": logistic_metrics,
            "linear_svm": svm_metrics,
        },
        "selected_by_macro_f1": selected_model,
        "per_class": logistic_metrics["per_class"],
        "vocabulary_size": len(feature_names),
        "top_weighted_terms": top_terms,
        "data_checks": data_checks,
        "artifacts": {
            "confusion_matrix_csv": str(confusion_csv),
            "confusion_matrix_png": str(confusion_png),
            "linear_svm_confusion_matrix_csv": str(svm_confusion_csv),
            "linear_svm_confusion_matrix_png": str(svm_confusion_png),
            "model_comparison_csv": str(comparison_csv),
        },
        "runtime": {
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Labels are derived from the first matching target arXiv category.",
            "Metrics use one deterministic internal held-out split, not an external test set.",
            "Multi-label category prediction and SPECTER2 classification are outside this comparison.",
        ],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for classifier evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=8420)
    parser.add_argument("--max-features", type=int, default=30_000)
    return parser


def main() -> None:
    """Parse CLI arguments and write evaluation artifacts."""
    args = build_parser().parse_args()
    result = evaluate_domain_classifier(
        args.corpus,
        args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        max_features=args.max_features,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
