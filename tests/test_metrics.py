from src.data_generation import SyntheticDataConfig, generate_sleep_app_data
from src.metrics import calculate_daily_product_metrics, calculate_sleep_improvement_rate, calculate_user_sleep_summary


def test_daily_product_metrics_have_expected_columns_and_ranges():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=10, n_days=14, random_state=3))
    daily = calculate_daily_product_metrics(df)

    expected_columns = {
        "date", "active_users", "total_users", "sleep_logs", "insights_viewed",
        "coach_messages", "plans_completed", "avg_sleep_duration", "avg_sleep_quality",
        "dau_rate", "sleep_log_completion_rate", "insight_view_rate",
        "coach_plan_completion_rate",
    }
    assert expected_columns.issubset(daily.columns)
    assert daily["dau_rate"].between(0, 1).all()
    assert daily["sleep_log_completion_rate"].between(0, 1).all()
    assert daily["insight_view_rate"].between(0, 1).all()


def test_user_summary_has_one_row_per_user():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=8, n_days=10, random_state=4))
    summary = calculate_user_sleep_summary(df)

    assert summary.shape[0] == 8
    assert "active_rate" in summary.columns
    assert summary["active_rate"].between(0, 1).all()


def test_sleep_improvement_rate_is_probability():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=10, n_days=30, random_state=5))
    improvement_rate = calculate_sleep_improvement_rate(df)

    assert 0 <= improvement_rate <= 1
