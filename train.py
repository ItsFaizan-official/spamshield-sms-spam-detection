"""Train, compare, and evaluate SpamShield's TF-IDF classifiers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import load_dataset

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"
MODEL_PATH = MODEL_DIR / "spam_classifier.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def build_models() -> dict[str, Pipeline]:
    """Create three pipelines, each with its own TF-IDF vectorizer."""
    def make_pipeline(classifier: BaseEstimator) -> Pipeline:
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.98,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    return {
        "Multinomial Naive Bayes": make_pipeline(MultinomialNB()),
        "Logistic Regression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        ),
        "Linear SVM": make_pipeline(LinearSVC(class_weight="balanced", random_state=42)),
    }


def measure_model(labels: pd.Series, predictions: Iterable[str]) -> dict[str, float]:
    """Calculate overall accuracy and spam-class precision, recall, and F1."""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "spam_precision": precision_score(
            labels, predictions, pos_label="spam", zero_division=0
        ),
        "spam_recall": recall_score(
            labels, predictions, pos_label="spam", zero_division=0
        ),
        "spam_f1": f1_score(labels, predictions, pos_label="spam", zero_division=0),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()
    print(f"Dataset size after cleaning: {len(data):,} messages")
    print("Label distribution:")
    print(data["label"].value_counts().to_string())

    development_messages, test_messages, development_labels, test_labels = train_test_split(
        data["message"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )
    train_messages, validation_messages, train_labels, validation_labels = train_test_split(
        development_messages,
        development_labels,
        test_size=0.25,
        random_state=42,
        stratify=development_labels,
    )

    results: list[dict[str, float | str]] = []
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_spam_f1 = -1.0

    for name, pipeline in build_models().items():
        print(f"\nTraining {name}...")
        pipeline.fit(train_messages, train_labels)
        predictions = pipeline.predict(validation_messages)
        metrics = measure_model(validation_labels, predictions)
        results.append({"model": name, **metrics})

        print(
            f"Validation spam F1: {metrics['spam_f1']:.4f} | "
            f"accuracy: {metrics['accuracy']:.4f}"
        )
        if metrics["spam_f1"] > best_spam_f1:
            best_spam_f1 = metrics["spam_f1"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained successfully.")

    comparison = pd.DataFrame(results).sort_values("spam_f1", ascending=False)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    final_messages = pd.concat([train_messages, validation_messages])
    final_labels = pd.concat([train_labels, validation_labels])
    best_pipeline.fit(final_messages, final_labels)
    test_predictions = best_pipeline.predict(test_messages)
    test_metrics = measure_model(test_labels, test_predictions)

    matrix = confusion_matrix(test_labels, test_predictions, labels=["ham", "spam"])
    pd.DataFrame(
        matrix,
        index=["actual_ham", "actual_spam"],
        columns=["predicted_ham", "predicted_spam"],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(test_labels, test_predictions, zero_division=0),
        encoding="utf-8",
    )
    (REPORT_DIR / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    joblib.dump(best_pipeline, MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "selection_metric": "spam-class F1-score on the stratified validation split",
        "split_sizes": {
            "train": len(train_messages),
            "validation": len(validation_messages),
            "test": len(test_messages),
        },
        "random_state": 42,
        "dataset_rows_after_cleaning": int(len(data)),
        "test_metrics": test_metrics,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nValidation model comparison (sorted by spam F1-score):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected model: {best_name}")
    print("Untouched test metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()
