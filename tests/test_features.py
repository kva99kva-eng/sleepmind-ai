from src.data_generation import SyntheticDataConfig, generate_sleep_app_data
from src.features import FEATURE_COLUMNS, TARGET, add_features, build_train_test_data


def test_add_features_creates_expected_columns():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=5, n_days=10, random_state=1))
    result = add_features(df)

    assert "weekday" in result.columns
    assert "is_weekend" in result.columns
    assert "sleep_debt_hours" in result.columns
    assert result["weekday"].between(0, 6).all()
    assert set(result["is_weekend"].unique()).issubset({0, 1})
    assert (result["sleep_debt_hours"] >= 0).all()


def test_build_train_test_data_outputs_expected_columns():
    df = generate_sleep_app_data(SyntheticDataConfig(n_users=20, n_days=20, random_state=2))
    X_train, X_test, y_train, y_test = build_train_test_data(df)

    assert list(X_train.columns) == FEATURE_COLUMNS
    assert list(X_test.columns) == FEATURE_COLUMNS
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
    assert TARGET == "felt_rested"
