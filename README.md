# cap5771sp25-project

# Heart Disease Risk Prediction – CAP5771 Project

## Overview

This project is for CAP5771 (Spring 2025) and focuses on predicting heart disease risk using modifiable lifestyle factors like sleep, stress, and physical activity. We're using machine learning models to help identify early signs of heart health issues and eventually plan to build a simple dashboard for users to interact with.

---

## Milestone 1: Project Setup & EDA

In the first milestone, we:
- Collected and merged multiple datasets (lifestyle, sleep, heart health, and gym activity)
- Cleaned and preprocessed the data
- Explored the data with visualizations (correlation heatmaps, scatter plots, etc.)
- Identified patterns like high stress correlating with short sleep and poor activity levels

---

## Milestone 2: Feature Engineering, Selection, and Modeling

For this phase, we:
- Engineered two new features:
  - `heart_resilience_score` = (thalach / heart_rate) * (daily_steps / 1000)
  - `lifestyle_recovery_index` = (sleep_duration * calories_burned) / (stress_level + 1)
- Created a custom label called `heart_risk` based on thresholds from stress, sleep, BP, and other indicators
- Encoded categorical variables using LabelEncoder and one-hot encoding
- Trained three models: Logistic Regression, SVM, and Random Forest
- Random Forest performed the best, especially on the lifestyle-only dataset (F1 score ~0.976)

---

---

## Milestone 3: Evaluation, Tool Development, & Interpretation

In Milestone 3, we focused on model evaluation, building an interactive tool, and drawing actionable insights.

### Key Tasks:
- Evaluated model performance on test data using:
  - **Accuracy**, **Precision**, **Recall**, **F1 Score**.
- Compared **full-feature model** vs **lifestyle-only model**:
  - Full-feature Random Forest Accuracy: **99.9%**.
  - Lifestyle-only Random Forest Accuracy: **97.9%**.
- Identified top predictive features:
  - **Cholesterol**, **Systolic BP**, **Stress Level**, **Heart Resilience Score**.
- Developed a **Streamlit dashboard** to provide:
  - Real-time risk prediction and probability.
  - Visual comparison of user inputs vs ideal health values.
  - Input summary table for review.

---

## Streamlit Dashboard: How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

## How to Run

1. Clone the repo:
```bash
git clone https://github.com/<your-username>/cap5771sp25-project.git

pip install -r requirements.txt

jupyter notebook notebooks/Milestone2.ipynb
