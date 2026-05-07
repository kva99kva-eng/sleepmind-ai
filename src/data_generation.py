from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticDataConfig:
    n_users: int = 250
    n_days: int = 90
    start_date: str = "2026-01-01"
    random_state: int = 42


def clip(value: float, low: float, high: float) -> float:
    """Clip a numeric value to a closed interval and return it as float."""
    return float(max(low, min(high, value)))


def generate_sleep_app_data(
    config: SyntheticDataConfig = SyntheticDataConfig(),
) -> pd.DataFrame:
    """Generate synthetic user-day data for a sleep coaching app."""
    rng = np.random.default_rng(config.random_state)
    start = pd.to_datetime(config.start_date).date()

    rows = []

    for user_idx in range(config.n_users):
        user_id = f"U{user_idx + 1:04d}"
        chronotype_shift = rng.normal(0, 0.7)
        baseline_stress = clip(rng.normal(5.0, 1.8), 1, 10)
        engagement_tendency = clip(rng.beta(4, 3), 0.05, 0.95)
        exercise_tendency = clip(rng.normal(35, 20), 0, 120)
        previous_quality = None

        for day_idx in range(config.n_days):
            current_date = start + timedelta(days=day_idx)
            weekday = current_date.weekday()
            is_weekend = weekday >= 5

            stress_level = clip(rng.normal(baseline_stress, 1.4), 1, 10)
            exercise_minutes = clip(rng.normal(exercise_tendency, 20), 0, 160)

            caffeine_after_16 = rng.random() < (
                0.18 + 0.03 * max(stress_level - 5, 0)
            )

            screen_time_before_bed_min = clip(
                rng.normal(70 + 8 * stress_level, 35),
                0,
                240,
            )

            bedtime_hour = 23.2 + chronotype_shift + rng.normal(0, 0.55)
            if is_weekend:
                bedtime_hour += rng.normal(0.6, 0.35)
            bedtime_hour = clip(bedtime_hour, 20.0, 27.0)

            sleep_latency_min = (
                12
                + 3.0 * stress_level
                + (18 if caffeine_after_16 else 0)
                + 0.05 * screen_time_before_bed_min
                + rng.normal(0, 8)
            )
            sleep_latency_min = clip(sleep_latency_min, 3, 110)

            wake_lambda = 0.6 + 0.12 * stress_level + (
                0.4 if caffeine_after_16 else 0
            )
            wake_episodes = int(rng.poisson(wake_lambda))
            wake_episodes = min(wake_episodes, 8)

            sleep_duration_hours = (
                8.1
                - 0.17 * stress_level
                - 0.35 * int(caffeine_after_16)
                - 0.004 * screen_time_before_bed_min
                + 0.006 * exercise_minutes
                + rng.normal(0, 0.55)
            )
            if is_weekend:
                sleep_duration_hours += rng.normal(0.35, 0.25)
            sleep_duration_hours = clip(sleep_duration_hours, 4.0, 10.5)

            time_in_bed_hours = (
                sleep_duration_hours
                + sleep_latency_min / 60
                + wake_episodes * rng.uniform(0.05, 0.16)
            )
            time_in_bed_hours = clip(time_in_bed_hours, sleep_duration_hours, 12.0)

            wake_time_hour = bedtime_hour + time_in_bed_hours
            if wake_time_hour >= 24:
                wake_time_hour -= 24

            sleep_efficiency = sleep_duration_hours / time_in_bed_hours

            quality = (
                55
                + 6.0 * (sleep_duration_hours - 7)
                + 18.0 * (sleep_efficiency - 0.85)
                - 2.2 * max(stress_level - 5, 0)
                - 0.05 * screen_time_before_bed_min
                - 2.4 * wake_episodes
                - (5 if caffeine_after_16 else 0)
                + 0.035 * exercise_minutes
                + rng.normal(0, 6)
            )

            sleep_quality_score = int(round(clip(quality, 0, 100)))
            felt_rested_probability = 1 / (1 + np.exp(-(sleep_quality_score - 68) / 8))
            felt_rested = rng.random() < felt_rested_probability

            app_opened = rng.random() < (0.35 + 0.55 * engagement_tendency)
            sleep_log_completed = app_opened and rng.random() < (
                0.55 + 0.3 * engagement_tendency
            )
            insight_viewed = app_opened and rng.random() < (
                0.25 + 0.35 * engagement_tendency
            )
            coach_message_sent = insight_viewed and rng.random() < 0.65

            plan_completed_probability = 0.18 + 0.45 * engagement_tendency
            if previous_quality is not None and previous_quality < 60:
                plan_completed_probability += 0.12

            plan_completed = coach_message_sent and rng.random() < clip(
                plan_completed_probability,
                0,
                0.95,
            )

            rows.append(
                {
                    "user_id": user_id,
                    "date": current_date.isoformat(),
                    "bedtime_hour": round(bedtime_hour, 2),
                    "wake_time_hour": round(wake_time_hour, 2),
                    "sleep_duration_hours": round(sleep_duration_hours, 2),
                    "time_in_bed_hours": round(time_in_bed_hours, 2),
                    "sleep_latency_min": round(sleep_latency_min, 1),
                    "wake_episodes": wake_episodes,
                    "caffeine_after_16": int(caffeine_after_16),
                    "screen_time_before_bed_min": round(screen_time_before_bed_min, 1),
                    "stress_level": round(stress_level, 1),
                    "exercise_minutes": round(exercise_minutes, 1),
                    "sleep_efficiency": round(sleep_efficiency, 3),
                    "sleep_quality_score": sleep_quality_score,
                    "felt_rested": int(felt_rested),
                    "app_opened": int(app_opened),
                    "sleep_log_completed": int(sleep_log_completed),
                    "insight_viewed": int(insight_viewed),
                    "coach_message_sent": int(coach_message_sent),
                    "plan_completed": int(plan_completed),
                }
            )

            previous_quality = sleep_quality_score

    return pd.DataFrame(rows)


def save_synthetic_data(
    output_path: str | Path = "data/synthetic/sleep_app_data.csv",
) -> Path:
    """Generate and save synthetic sleep app data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_sleep_app_data()
    df.to_csv(output_path, index=False)

    return output_path


if __name__ == "__main__":
    path = save_synthetic_data()
    print(f"Synthetic data saved to: {path}")
