from src.coach import generate_sleep_coach_message
from src.data_generation import SyntheticDataConfig, generate_sleep_app_data


def test_sleep_coach_message_contains_safe_disclaimer():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=1, n_days=14, random_state=6))
    user_df = df[df["user_id"] == "U0001"]

    message = generate_sleep_coach_message(user_df)

    assert "Weekly sleep insight" in message
    assert "Suggested next steps" in message
    assert "not a medical diagnosis" in message.lower()


def test_sleep_coach_handles_empty_user_dataframe():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=1, n_days=1, random_state=7))
    empty_df = df.iloc[0:0]

    message = generate_sleep_coach_message(empty_df)

    assert "Not enough data" in message
