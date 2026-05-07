from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "felt_rested"

FEATURE_COLUMNS = [
    "bedtime_hour",
    "wake_time_hour",
    "sleep_duration_hours",
    "time_in_bed_hours",
    "sleep_latency_min",
    "wake_episodes",
    "caffeine_after_16",
    "screen_time_before_bed_min",
    "stress_level",
    "exercise_minutes",
    "sleep_efficiency",
    "weekday",
    "is_weekend",
    "sleep_debt_hours",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and sleep behavior features for ML model."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["weekday"] = data["date"].dt.weekday
    data["is_weekend"] = (data["weekday"] >= 5).astype(int)
    data["sleep_debt_hours"] = (7.0 - data["sleep_duration_hours"]).clip(lower=0)

    return data


def build_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train and test datasets."""
    data = add_features(df)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic/sleep_app_data.csv")
    data_with_features = add_features(df)
    X_train, X_test, y_train, y_test = build_train_test_data(df)

    print("Data with new features:")
    print(data_with_features[["date", "weekday", "is_weekend", "sleep_debt_hours"]].head())

    print("\nTrain/test shapes:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nTarget distribution:")
    print(y_train.value_counts(normalize=True))
