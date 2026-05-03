# SleepMind AI — Product Case Study

## 1. Problem

Sleep tracking apps often show raw numbers, but users do not always understand what those numbers mean.

A user may see that they slept 6.2 hours, but still not know:

- whether their sleep routine is improving;
- which habits are associated with poor sleep quality;
- whether late caffeine or screen time affects their sleep;
- what action they should try tonight.

The product problem:

> How can we turn sleep tracking data into clear behavioral insights and simple coaching recommendations?

## 2. Product Hypothesis

Users who receive personalized sleep insights and simple coaching suggestions are more likely to:

- return to the app;
- complete sleep logs;
- view insights;
- follow coaching plans;
- improve sleep consistency over time.

## 3. Target Users

Potential target users:

- people who track sleep manually or with wearables;
- users interested in improving sleep habits;
- digital health / sleeptech product teams;
- analysts working with behavioral health data.

## 4. MVP Scope

The MVP includes four main components:

1. Synthetic sleep app data generation
2. Product analytics metrics
3. ML model for predicting whether a user felt rested
4. Rule-based AI sleep coach

The goal is not to build a medical product.

The goal is to demonstrate how sleep-related behavioral data can be analyzed and turned into product insights.

## 5. Dataset

The MVP uses synthetic user-day data.

Dataset size:

```text
250 users
90 days
22,500 rows
Each row represents one user on one day.

Main data groups:

Sleep behavior
bedtime
wake time
sleep duration
time in bed
sleep latency
wake episodes
sleep efficiency
sleep quality score
Lifestyle factors
stress level
screen time before bed
caffeine after 16:00
exercise minutes
Product events
app opened
sleep log completed
insight viewed
coach message sent
plan completed
Target variable
felt rested
6. Product Metrics

The project calculates the following product metrics:

MetricMeaning
DAU rateShare of users active on a given day
Sleep log completion rateShare of active users who completed a sleep log
Insight view rateShare of active users who viewed insights
Coaching plan completion rateShare of users who completed a suggested plan
Sleep improvement rateShare of users whose recent sleep quality improved

These metrics imitate how a product analyst would evaluate a sleep coaching app.

7. ML Task

The ML task is binary classification.

Target: felt_rested

The model predicts whether a user is likely to feel rested based on sleep and behavior features.

Features:

sleep duration
sleep efficiency
sleep latency
wake episodes
stress level
screen time before bed
caffeine after 16:00
exercise minutes
weekday
weekend flag
sleep debt
8. Model Results

Baseline model:

Random Forest Classifier

Current metrics:

Accuracy: 0.6976
Balanced accuracy: 0.6671
F1: 0.3212
ROC AUC: 0.7434

The target variable is imbalanced, so accuracy alone is not enough.

Balanced accuracy and ROC AUC are more useful in this project because the positive class felt_rested = 1 is less frequent.

9. Feature Importance

Top model features:

sleep efficiency
sleep duration
sleep debt
stress level
sleep latency

This result is consistent with the logic of the synthetic data generation process and with common sleep behavior assumptions.

The model mainly uses sleep quality-related and behavior-related features rather than random product engagement variables.

10. AI Sleep Coach

The AI sleep coach is rule-based in the MVP.

It analyzes the last 7 days of user data and generates:

weekly sleep insight;
comparison with the previous week;
personalized next steps;
safety disclaimer.

Example recommendation types:

protect a longer sleep window;
keep a stable wake-up time;
reduce screen time before bed;
move caffeine earlier;
use a wind-down routine.

The coach does not diagnose sleep disorders.

11. Safety and Limitations

This project is not a medical device.

Limitations:

synthetic dataset;
no clinical validation;
no real wearable data;
no diagnosis;
no treatment recommendations;
rule-based coaching logic;
baseline ML model only.

The project should be interpreted as a portfolio MVP for sleeptech / digital health analytics.

12. Next Steps

Possible next improvements:

Add retention curves
Add cohort analysis
Add A/B test simulation
Add model threshold tuning
Add confusion matrix visualization
Add screenshots to README
Add real sleep dataset
Add Sleep-EDF EEG-based sleep staging module
13. Portfolio Value

This project demonstrates:

product analytics thinking;
sleep behavior analysis;
feature engineering;
ML model evaluation;
imbalanced classification awareness;
dashboard prototyping;
safe AI assistant design;
healthtech / sleeptech product thinking.
14. Summary

SleepMind AI is a portfolio MVP that combines product analytics, machine learning, and AI-style coaching for sleep behavior data.

The strongest part of the project is not the model alone, but the full product-oriented pipeline:

data generation → product metrics → ML prediction → AI coach → dashboard

This makes the project more realistic than a single notebook and more relevant for junior roles in healthtech, sleeptech, neurotech, and product analytics.
