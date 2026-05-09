from __future__ import annotations

from typing import List

import pandas as pd


def _format_hour(hour_float: float) -> str:
    """Convert float hour to HH:MM format. Example: 23.5 -> 23:30."""
    hour = int(hour_float) % 24
    minute = int(round((hour_float - int(hour_float)) * 60))

    if minute == 60:
        hour = (hour + 1) % 24
        minute = 0

    return f"{hour:02d}:{minute:02d}"


def generate_sleep_coach_message(user_df: pd.DataFrame) -> str:
    """
    Generate safe, non-medical sleep coaching message.

    This is not a medical diagnosis. The goal is to explain behavioral
    patterns and suggest simple wellness actions.
    """
    if user_df.empty:
        return (
            "Not enough data yet. "
            "Track your sleep for several days to receive personalized insights."
        )

    data = user_df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date")

    recent = data.tail(7)
    previous = data.iloc[-14:-7] if len(data) >= 14 else pd.DataFrame()

    avg_duration = recent["sleep_duration_hours"].mean()
    avg_quality = recent["sleep_quality_score"].mean()
    avg_latency = recent["sleep_latency_min"].mean()
    avg_screen = recent["screen_time_before_bed_min"].mean()
    avg_stress = recent["stress_level"].mean()
    avg_wake = recent["wake_episodes"].mean()
    caffeine_days = int(recent["caffeine_after_16"].sum())
    bedtime_std = recent["bedtime_hour"].std()

    insights: List[str] = []
    actions: List[str] = []

    if not previous.empty:
        prev_quality = previous["sleep_quality_score"].mean()
        delta = avg_quality - prev_quality

        if delta >= 5:
            insights.append(
                f"Your sleep quality improved by about {delta:.1f} points "
                "compared with the previous week."
            )
        elif delta <= -5:
            insights.append(
                f"Your sleep quality decreased by about {abs(delta):.1f} points "
                "compared with the previous week."
            )
        else:
            insights.append(
                "Your sleep quality is relatively stable compared with the previous week."
            )

    insights.append(
        f"Your average sleep quality for the last 7 days is {avg_quality:.0f}/100."
    )

    if avg_duration < 7:
        insights.append(
            f"Your average sleep duration for the last 7 days is {avg_duration:.1f} hours."
        )
        actions.append(
            "Try to protect a slightly longer sleep window tonight. "
            "Even 20–30 extra minutes can help build consistency."
        )
    else:
        insights.append(
            f"Your average sleep duration for the last 7 days is {avg_duration:.1f} hours."
        )

    if pd.notna(bedtime_std) and bedtime_std > 0.75:
        actions.append(
            "Your bedtime varies a lot. "
            "For the next week, choose one realistic wake-up time and keep it stable."
        )
    elif len(recent) >= 4:
        median_bedtime = recent["bedtime_hour"].median()
        insights.append(f"Your bedtime is fairly consistent around {_format_hour(median_bedtime)}.")

    if avg_latency > 35:
        insights.append(
            f"It takes you about {avg_latency:.0f} minutes to fall asleep on average."
        )
        actions.append(
            "Create a 30-minute wind-down routine: dim lights, avoid work tasks, "
            "and keep the phone away from bed."
        )

    if avg_screen > 90:
        actions.append(
            "Screen time before bed is high. "
            "Try reducing it by 20 minutes for the next 3 nights and compare sleep quality."
        )

    if avg_stress > 7:
        actions.append(
            "Stress looks elevated. Add a short decompression habit before sleep: "
            "breathing, stretching, or a simple paper to-do list."
        )

    if caffeine_days >= 2:
        actions.append(
            f"You had caffeine after 16:00 on {caffeine_days} of the last 7 days. "
            "Try moving caffeine earlier for one week."
        )

    if avg_wake > 2:
        actions.append(
            "Night wake-ups are frequent. Keep the bedroom cool and dark, "
            "and avoid checking the phone when you wake up."
        )

    if not actions:
        actions.append(
            "Keep the current routine. The next improvement target is consistency: "
            "same wake-up time and similar bedtime."
        )

    disclaimer = (
        "This is a wellness-oriented coaching suggestion, not a medical diagnosis. "
        "If sleep problems are severe, persistent, or affect daily life, consult a qualified clinician."
    )

    message = "### Weekly sleep insight\n\n"
    message += "\n".join(f"- {item}" for item in insights[:4])
    message += "\n\n### Suggested next steps\n\n"
    message += "\n".join(f"- {item}" for item in actions[:4])
    message += f"\n\n_{disclaimer}_"

    return message
