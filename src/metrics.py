from __future__ import annotations

import pandas as pd


def calculate_daily_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily product metrics from user-day sleep app data.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    daily = data.groupby("date").agg(
        active_users=("app_opened", "sum"),
        total_users=("user_id", "nunique"),
        sleep_logs=("sleep_log_completed", "sum"),
        insights_viewed=("insight_viewed", "sum"),
        coach_messages=("coach_message_sent", "sum"),
        plans_completed=("plan_completed", "sum"),
        avg_sleep_duration=("sleep_duration_hours", "mean"),
        avg_sleep_quality=("sleep_quality_score", "mean"),
        avg_stress=("stress_level", "mean"),
    ).reset_index()

    daily["dau_rate"] = daily["active_users"] / daily["total_users"]

    daily["sleep_log_completion_rate"] = (
        daily["sleep_logs"] / daily["active_users"].replace(0, pd.NA)
    )

    daily["insight_view_rate"] = (
        daily["insights_viewed"] / daily["active_users"].replace(0, pd.NA)
    )

    daily["coach_plan_completion_rate"] = (
        daily["plans_completed"] / daily["coach_messages"].replace(0, pd.NA)
    )

    return daily.fillna(0)


def calculate_user_sleep_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate user-level sleep and engagement summary.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    summary = data.groupby("user_id").agg(
        days=("date", "nunique"),
        avg_sleep_duration=("sleep_duration_hours", "mean"),
        avg_sleep_quality=("sleep_quality_score", "mean"),
        avg_sleep_efficiency=("sleep_efficiency", "mean"),
        avg_stress=("stress_level", "mean"),
        avg_screen_time=("screen_time_before_bed_min", "mean"),
        avg_wake_episodes=("wake_episodes", "mean"),
        rested_rate=("felt_rested", "mean"),
        active_days=("app_opened", "sum"),
        completed_plans=("plan_completed", "sum"),
    ).reset_index()

    summary["active_rate"] = summary["active_days"] / summary["days"]

    return summary


def calculate_sleep_improvement_rate(df: pd.DataFrame, window_days: int = 14) -> float:
    """
    Calculate share of users whose recent sleep quality is better
    than their first period.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["user_id", "date"])

    improved_users = []

    for _, user_df in data.groupby("user_id"):
        if len(user_df) < window_days * 2:
            continue

        first_period_quality = user_df.head(window_days)["sleep_quality_score"].mean()
        last_period_quality = user_df.tail(window_days)["sleep_quality_score"].mean()

        improved_users.append(last_period_quality > first_period_quality)

    if not improved_users:
        return 0.0

    return round(sum(improved_users) / len(improved_users), 4)


if __name__ == "__main__":
    df = pd.read_csv("data/synthetic/sleep_app_data.csv")

    daily_metrics = calculate_daily_product_metrics(df)
    user_summary = calculate_user_sleep_summary(df)
    improvement_rate = calculate_sleep_improvement_rate(df)

    print("Daily metrics:")
    print(daily_metrics.head())

    print("\nUser summary:")
    print(user_summary.head())

    print(f"\nSleep improvement rate: {improvement_rate:.2%}")
