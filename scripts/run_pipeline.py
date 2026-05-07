from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import save_synthetic_data
from src.metrics import (
    calculate_daily_product_metrics,
    calculate_sleep_improvement_rate,
)
from src.model import save_model_outputs, train_sleep_model


def main() -> None:
    data_path = PROJECT_ROOT / "data" / "synthetic" / "sleep_app_data.csv"
    daily_metrics_path = PROJECT_ROOT / "data" / "processed" / "daily_product_metrics.csv"
    model_path = PROJECT_ROOT / "reports" / "sleep_model.joblib"
    model_metrics_path = PROJECT_ROOT / "reports" / "model_metrics.json"

    print("Step 1/4: Generating synthetic sleep app data...")
    save_synthetic_data(str(data_path))
    df = pd.read_csv(data_path)

    print(f"Dataset saved to: {data_path.relative_to(PROJECT_ROOT)}")
    print(f"Rows: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")

    print("\nStep 2/4: Calculating product analytics metrics...")
    daily_metrics = calculate_daily_product_metrics(df)
    daily_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    daily_metrics.to_csv(daily_metrics_path, index=False)

    improvement_rate = calculate_sleep_improvement_rate(df)

    print(f"Daily product metrics saved to: {daily_metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Sleep improvement rate: {improvement_rate:.2%}")

    print("\nStep 3/4: Training sleep quality prediction model...")
    model, model_metrics = train_sleep_model(df)

    save_model_outputs(
        model,
        model_metrics,
        model_path=model_path,
        metrics_path=model_metrics_path,
    )

    print(f"Model saved to: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Model metrics saved to: {model_metrics_path.relative_to(PROJECT_ROOT)}")

    print("\nStep 4/4: Final model metrics:")
    print(f"Accuracy: {model_metrics['accuracy']}")
    print(f"Balanced accuracy: {model_metrics['balanced_accuracy']}")
    print(f"F1: {model_metrics['f1']}")
    print(f"ROC AUC: {model_metrics['roc_auc']}")

    print("\nTop 5 feature importance:")
    for item in model_metrics["feature_importance"][:5]:
        print(f"- {item['feature']}: {item['importance']:.4f}")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
