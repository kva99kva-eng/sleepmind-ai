from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.features import FEATURE_COLUMNS, build_train_test_data


def build_model(random_state: int = 42) -> Pipeline:
    """
    Build baseline ML model for predicting whether user felt rested.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=9,
        min_samples_leaf=8,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def train_sleep_model(df: pd.DataFrame, random_state: int = 42) -> Tuple[Pipeline, Dict]:
    """
    Train model and calculate quality metrics.
    """
    X_train, X_test, y_train, y_test = build_train_test_data(
        df,
        random_state=random_state,
    )

    pipeline = build_model(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    rf_model = pipeline.named_steps["model"]

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    metrics["feature_importance"] = feature_importance.to_dict(orient="records")

    return pipeline, metrics


def save_model_outputs(
    pipeline: Pipeline,
    metrics: Dict,
    model_path: str | Path = "reports/sleep_model.joblib",
    metrics_path: str | Path = "reports/model_metrics.json",
) -> None:
    """
    Save trained model and metrics.
    """
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic/sleep_app_data.csv")

    model, metrics = train_sleep_model(df)
    save_model_outputs(model, metrics)

    print("Model metrics:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']}")
    print(f"F1: {metrics['f1']}")
    print(f"ROC AUC: {metrics['roc_auc']}")

    print("\nTop feature importance:")
    importance = pd.DataFrame(metrics["feature_importance"])
    print(importance.head(10))
