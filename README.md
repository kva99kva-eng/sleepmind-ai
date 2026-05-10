# SleepMind AI

[![Python checks](https://github.com/kva99kva-eng/sleepmind-ai/actions/workflows/python-checks.yml/badge.svg)](https://github.com/kva99kva-eng/sleepmind-ai/actions/workflows/python-checks.yml)

**Sleep Product Analytics & AI Coaching Platform**

SleepMind AI is a portfolio project that combines sleep behavior analytics, product metrics, machine learning, threshold analysis and a rule-based AI coaching assistant.

The project is designed as a realistic MVP of a sleep-tech / digital health product.

## Executive Summary

This project demonstrates an end-to-end product analytics and ML workflow for a simulated sleep coaching app.

I generated a synthetic user-day dataset, designed product metrics, trained a model to predict whether a user is likely to feel rested, evaluated model behavior under class imbalance, and built a Streamlit dashboard with a safe rule-based sleep coach.

The strongest part of the project is the product framing: the model is not treated as a standalone ML task, but as part of a user-facing decision system where threshold choice, engagement metrics and recommendation safety matter.

Key analytical decisions:

- modeled a realistic sleep-tech product journey;
- separated product metrics from ML quality metrics;
- evaluated model behavior under class imbalance;
- compared thresholds using precision, recall and F1;
- avoided medical claims in the AI coach;
- documented limitations of synthetic data.

## Product Question

How can a sleep coaching app use behavioral and sleep-tracking data to understand user engagement, predict whether users feel rested, and provide safe non-medical coaching suggestions?

## Project Highlights

- Built an end-to-end sleep analytics MVP: data generation, product metrics, ML model, AI coach and dashboard.
- Simulated a sleep-tech product dataset with 22,500 user-day records.
- Designed product metrics: DAU rate, sleep log completion, insight view rate, coaching plan completion and sleep improvement rate.
- Trained a Random Forest model to predict `felt_rested`.
- Evaluated the model under class imbalance using balanced accuracy, ROC AUC, precision, recall, F1, threshold analysis and confusion matrices.
- Built a Streamlit dashboard for user sleep trends, product analytics, ML evaluation and coaching recommendations.
- Added a safe rule-based sleep coach with wellness-oriented recommendations and no medical diagnosis claims.

## Dataset

The MVP uses synthetic user-day sleep data.

Dataset size:

- 22,500 rows
- 250 users
- 90 days

Main columns include:

- `user_id`
- `date`
- `bedtime_hour`
- `wake_time_hour`
- `sleep_duration_hours`
- `time_in_bed_hours`
- `sleep_latency_min`
- `wake_episodes`
- `caffeine_after_16`
- `screen_time_before_bed_min`
- `stress_level`
- `exercise_minutes`
- `sleep_efficiency`
- `sleep_quality_score`
- `felt_rested`
- `app_opened`
- `sleep_log_completed`
- `insight_viewed`
- `coach_message_sent`
- `plan_completed`

## Product Analytics

The project calculates product and engagement metrics:

- DAU rate
- Sleep log completion rate
- Insight view rate
- Coaching plan completion rate
- Average sleep duration
- Average sleep quality
- Sleep improvement rate

These metrics imitate how a product analyst would evaluate a sleep coaching app.

## ML Task

The machine learning task is binary classification.

Target:

```text
felt_rested
```

The model predicts whether a user is likely to feel rested based on sleep and behavior features.

Features include:

- sleep duration
- sleep efficiency
- sleep latency
- wake episodes
- stress level
- screen time before bed
- caffeine after 16:00
- exercise minutes
- weekday / weekend
- sleep debt

## Model

Baseline model:

- Random Forest Classifier

Current model metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.6976 |
| Balanced accuracy | 0.6671 |
| F1 | 0.3212 |
| ROC AUC | 0.7434 |

The target is imbalanced, so balanced accuracy and ROC AUC are more informative than accuracy alone.

Top important features:

1. sleep efficiency
2. sleep duration hours
3. sleep debt hours
4. stress level
5. sleep latency minutes

## Threshold Analysis

The target variable `felt_rested` is imbalanced, so accuracy alone is not enough to evaluate the model.

The model was evaluated with two probability thresholds:

| Threshold | Accuracy | Balanced Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.6976 | 0.6671 | 0.2158 | 0.6277 | 0.3212 |
| 0.60 | 0.7922 | 0.6382 | 0.2580 | 0.4386 | 0.3249 |

The optimized threshold `0.60` slightly improves F1 and precision, but reduces recall.

For a sleep coaching product, the threshold choice depends on the product goal:

- lower threshold: find more potentially rested users, but with more false positives;
- higher threshold: make fewer but more confident positive predictions.

## AI Sleep Coach

The coaching assistant is rule-based in the MVP.

It analyzes the last 7 days of sleep data and generates:

- weekly sleep insight;
- trend comparison with the previous week;
- suggested next steps;
- safety disclaimer.

The coach does not provide medical diagnoses.

## Dashboard Screenshots

### User Sleep Dashboard

![User Sleep Dashboard](reports/figures/user_sleep_dashboard.png)

### Product Analytics

![Product Analytics](reports/figures/product_analytics.png)

### ML Model Metrics

![ML Model Metrics](reports/figures/ml_model_metrics.png)

### ML Threshold Analysis

![ML Model Threshold Analysis](reports/figures/ml_model_threshold_analysis.png)

### ML Confusion Matrices

![ML Model Confusion Matrices](reports/figures/ml_model_confusion_matrices.png)

## Project Structure

```text
sleepmind-ai/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── processed/
│   └── synthetic/
├── reports/
│   ├── figures/
│   ├── model_metrics.json
│   ├── evaluation_summary.json
│   └── threshold_metrics.csv
├── scripts/
│   └── run_pipeline.py
├── src/
│   ├── coach.py
│   ├── data_generation.py
│   ├── evaluation.py
│   ├── features.py
│   ├── metrics.py
│   └── model.py
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/kva99kva-eng/sleepmind-ai.git
cd sleepmind-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python scripts\run_pipeline.py
```

Run threshold evaluation:

```bash
python -m src.evaluation
```

Run the Streamlit dashboard:

```bash
streamlit run app\streamlit_app.py
```

If Streamlit is not recognized:

```bash
python -m streamlit run app\streamlit_app.py
```

## Dashboard Tabs

The Streamlit app contains four tabs:

1. User Sleep Dashboard — individual sleep trends and recent sleep logs.
2. Product Analytics — engagement metrics and user-level summaries.
3. ML Model — model quality metrics, feature importance, threshold analysis and confusion matrices.
4. AI Coach — personalized weekly sleep insight and recommendations.

## Limitations

This project is for portfolio and educational purposes.

Important limitations:

- The dataset is synthetic.
- The model is not validated on real clinical or wearable data.
- The sleep coach is rule-based.
- The project does not diagnose, treat or prevent sleep disorders.
- Recommendations are wellness-oriented and not medical advice.

## Future Work

Planned improvements:

- Add cohort analysis.
- Add retention curves.
- Add A/B test simulation.
- Add real wearable or sleep dataset.
- Add Sleep-EDF EEG-based sleep staging module.
- Improve model evaluation for imbalanced classification.
- Add tests for metrics and feature engineering.

## Portfolio Positioning

This project demonstrates:

- product analytics;
- sleep behavior analysis;
- feature engineering;
- machine learning;
- model evaluation;
- threshold analysis;
- Streamlit prototyping;
- safe AI assistant design;
- healthtech / sleeptech product thinking.

## Resume Summary

Built a sleep-tech product analytics MVP with synthetic user-day data, product metrics, Random Forest model, threshold analysis, Streamlit dashboard and safe rule-based sleep coaching logic.

## License

This project is licensed under the MIT License.
