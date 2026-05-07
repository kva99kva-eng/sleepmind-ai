from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import build_train_test_data
from src.model import build_model


def evaluate_thresholds(
    y_true,
    y_prob,
    thresholds: list[float],
) -> pd.DataFrame:
    """Evaluate model quality at different probability thresholds."""
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix_plot(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str,
) -> None:
    """Save confusion matrix plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["not rested", "rested"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_evaluation(
    data_path: str | Path = "data/synthetic/sleep_app_data.csv",
    reports_dir: str | Path = "reports",
) -> dict:
    """Train model, evaluate threshold tuning, and save evaluation artifacts."""
    data_path = Path(data_path)
    reports_dir = Path(reports_dir)

    df = pd.read_csv(data_path)

    X_train, X_test, y_train, y_test = build_train_test_data(df)

    model = build_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = [round(x / 100, 2) for x in range(10, 91, 5)]

    threshold_metrics = evaluate_thresholds(
        y_true=y_test,
        y_prob=y_prob,
        thresholds=thresholds,
    )

    threshold_metrics_path = reports_dir / "threshold_metrics.csv"
    threshold_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_metrics.to_csv(threshold_metrics_path, index=False)

    best_row = threshold_metrics.sort_values("f1", ascending=False).iloc[0]
    best_threshold = float(best_row["threshold"])

    default_threshold = 0.5

    y_pred_default = (y_prob >= default_threshold).astype(int)
    y_pred_best = (y_prob >= best_threshold).astype(int)

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred_default,
        output_path=reports_dir / "figures" / "confusion_matrix_default.png",
        title="Confusion Matrix — Default Threshold 0.50",
    )

    save_confusion_matrix_plot(
        y_true=y_test,
        y_pred=y_pred_best,
        output_path=reports_dir / "figures" / "confusion_matrix_optimized.png",
        title=f"Confusion Matrix — Optimized Threshold {best_threshold:.2f}",
    )

    summary = {
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "default_threshold": default_threshold,
        "default_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_default)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred_default)), 4),
            "precision": round(float(precision_score(y_test, y_pred_default, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_default, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_default, zero_division=0)), 4),
        },
        "optimized_threshold": round(best_threshold, 2),
        "optimized_metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred_best)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred_best)), 4),
            "precision": round(float(precision_score(y_test, y_pred_best, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred_best, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred_best, zero_division=0)), 4),
        },
    }

    summary_path = reports_dir / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


if __name__ == "__main__":
    summary = run_evaluation()

    print("Evaluation summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved files:")
    print("- reports/threshold_metrics.csv")
    print("- reports/evaluation_summary.json")
    print("- reports/figures/confusion_matrix_default.png")
    print("- reports/figures/confusion_matrix_optimized.png")
