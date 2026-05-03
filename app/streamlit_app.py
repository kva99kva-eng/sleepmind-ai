from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.data_generation import save_synthetic_data
from src.metrics import (
    calculate_daily_product_metrics,
    calculate_sleep_improvement_rate,
    calculate_user_sleep_summary,
)
from src.model import train_sleep_model
from src.coach import generate_sleep_coach_message


st.set_page_config(
    page_title="SleepMind AI",
    page_icon="🌙",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = PROJECT_ROOT / "data" / "synthetic" / "sleep_app_data.csv"

    if not data_path.exists():
        save_synthetic_data(str(data_path))

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_resource
def train_model_cached(df: pd.DataFrame):
    model, metrics = train_sleep_model(df)
    return model, metrics


df = load_data()

daily_metrics = calculate_daily_product_metrics(df)
user_summary = calculate_user_sleep_summary(df)
improvement_rate = calculate_sleep_improvement_rate(df)

st.title("🌙 SleepMind AI")
st.caption("Sleep Product Analytics & AI Coaching Platform")

with st.sidebar:
    st.header("Controls")

    user_ids = sorted(df["user_id"].unique())
    selected_user = st.selectbox("Select user", user_ids, index=0)

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )


if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])
else:
    start_date = df["date"].min()
    end_date = df["date"].max()


filtered_df = df[
    (df["date"] >= start_date)
    & (df["date"] <= end_date)
].copy()

user_df = filtered_df[filtered_df["user_id"] == selected_user].sort_values("date")


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "User Sleep Dashboard",
        "Product Analytics",
        "ML Model",
        "AI Coach",
    ]
)


with tab1:
    st.subheader(f"User Sleep Dashboard: {selected_user}")

    if user_df.empty:
        st.warning("No data for selected period.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Avg sleep duration",
            f"{user_df['sleep_duration_hours'].mean():.1f} h",
        )
        col2.metric(
            "Avg sleep quality",
            f"{user_df['sleep_quality_score'].mean():.0f}/100",
        )
        col3.metric(
            "Avg sleep efficiency",
            f"{user_df['sleep_efficiency'].mean():.0%}",
        )
        col4.metric(
            "Felt rested rate",
            f"{user_df['felt_rested'].mean():.0%}",
        )

        st.write("Sleep duration and quality trend")

        sleep_chart = user_df.set_index("date")[
            [
                "sleep_duration_hours",
                "sleep_quality_score",
            ]
        ]

        st.line_chart(sleep_chart)

        st.write("Recent sleep logs")

        st.dataframe(
            user_df[
                [
                    "date",
                    "bedtime_hour",
                    "wake_time_hour",
                    "sleep_duration_hours",
                    "sleep_latency_min",
                    "wake_episodes",
                    "stress_level",
                    "screen_time_before_bed_min",
                    "sleep_quality_score",
                    "felt_rested",
                ]
            ].tail(14),
            width="stretch",
        )


with tab2:
    st.subheader("Product Analytics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Users", f"{filtered_df['user_id'].nunique():,}")
    col2.metric("Avg DAU rate", f"{daily_metrics['dau_rate'].mean():.0%}")
    col3.metric(
        "Sleep log completion",
        f"{daily_metrics['sleep_log_completion_rate'].mean():.0%}",
    )
    col4.metric("Sleep improvement rate", f"{improvement_rate:.0%}")

    st.write("Daily engagement metrics")

    engagement_chart = daily_metrics.set_index("date")[
        [
            "dau_rate",
            "sleep_log_completion_rate",
            "insight_view_rate",
            "coach_plan_completion_rate",
        ]
    ]

    st.line_chart(engagement_chart)

    st.write("User-level summary")

    st.dataframe(
        user_summary.sort_values("avg_sleep_quality", ascending=False),
        width="stretch",
    )


with tab3:
    st.subheader("ML Model: Predict `felt_rested`")

    model, model_metrics = train_model_cached(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", model_metrics["accuracy"])
    col2.metric("Balanced accuracy", model_metrics["balanced_accuracy"])
    col3.metric("F1", model_metrics["f1"])
    col4.metric("ROC AUC", model_metrics["roc_auc"])

    st.write("Top feature importance")

    importance_df = pd.DataFrame(model_metrics["feature_importance"])

    st.bar_chart(
        importance_df.set_index("feature")["importance"]
    )

    with st.expander("Raw model metrics"):
        st.json(
            {
                key: value
                for key, value in model_metrics.items()
                if key != "classification_report"
            }
        )


with tab4:
    st.subheader("AI Sleep Coach")

    st.markdown(generate_sleep_coach_message(user_df))

    st.info(
        "This is a rule-based coach for the MVP. "
        "It provides wellness-oriented suggestions and does not make medical diagnoses."
    )
