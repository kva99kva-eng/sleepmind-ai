from src.data_generation import SyntheticDataConfig, generate_sleep_app_data


def test_generate_sleep_app_data_shape_and_required_columns():
    config = SyntheticDataConfig(n_users=5, n_days=7, random_state=123)
    df = generate_sleep_app_data(config)

    assert df.shape[0] == 35
    assert df["user_id"].nunique() == 5

    required_columns = {
        "user_id", "date", "sleep_duration_hours", "time_in_bed_hours",
        "sleep_latency_min", "wake_episodes", "sleep_efficiency",
        "sleep_quality_score", "felt_rested", "app_opened",
        "sleep_log_completed", "insight_viewed", "coach_message_sent",
        "plan_completed",
    }
    assert required_columns.issubset(df.columns)


def test_generate_sleep_app_data_value_ranges():
    config = SyntheticDataConfig(n_users=3, n_days=5, random_state=42)
    df = generate_sleep_app_data(config)

    assert df["sleep_duration_hours"].between(4.0, 10.5).all()
    assert df["sleep_efficiency"].between(0, 1).all()
    assert df["sleep_quality_score"].between(0, 100).all()
    assert set(df["felt_rested"].unique()).issubset({0, 1})
