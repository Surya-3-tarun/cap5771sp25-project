# cap5771sp25-project

## Sri Surya Tarun Narala
## UFID: 27268217

# Heart Disease Risk Prediction – CAP5771 Project

## Overview

This project predicts heart disease risk by analyzing both clinical and lifestyle factors—such as sleep, stress, physical activity, blood pressure, and cholesterol—using machine learning. The interactive Streamlit dashboard provides personalized, real-time insights to help users understand their heart health and make data-driven lifestyle improvements.

---

## Quick Start – How to Run the Dashboard
1. **Clone the repository**  
   git clone https://github.com/<your-username>/cap5771sp25-project.git
   cd cap5771sp25-project

2. **Install dependencies**  
    pip install -r requirements.txt

3. **Run the Streamlit app**  
    streamlit run app.py
   
Note: Ensure that the model files rf_model.pkl and scaler.pkl are located in the /model directory.

## Tool Demo Video
[Watch the Demo Video](https://drive.google.com/file/d/11vC_4P4XhsEFQS6n9E_j0dgJbq2a1yeu/view?usp=drive_link)

## Presentation Video
[Watch the Presentation](https://drive.google.com/file/d/1F-BtdTlTqvgqDTj6_Jlm_nxk6kkV7Gqe/view?usp=drive_link)


## PowerPoint Link
(https://docs.google.com/presentation/d/1GxwM6DXTHV5YtKdJQn9jzp2sgObon8wX/edit?usp=drive_link&ouid=102252239096845929472&rtpof=true&sd=true)

## Discussion PowerPoint Link
(https://docs.google.com/presentation/d/1w2LJ8Mi2AgH4VmvaxOfC_ARFTvHPJegH/edit?usp=drive_link&ouid=102252239096845929472&rtpof=true&sd=true)

## Project Milestones

### Milestone 1: Project Setup & EDA
- Collected and merged three datasets: lifestyle & sleep, heart health, and gym activity.
- Cleaned and preprocessed data (handled missing values, reduced outliers via IQR).
- Explored patterns with correlation heatmaps and scatter plots.
- Discovered high stress correlates with poor sleep and low activity.

### Milestone 2: Feature Engineering, Selection, and Modeling
- Engineered two new features:
  - `heart_resilience_score = (thalach / heart_rate) * (daily_steps / 1000)`
  - `lifestyle_recovery_index = (sleep_duration * calories_burned) / (stress_level + 1)`
- Created a binary target `heart_risk` using medically inspired thresholds.
- Encoded categorical variables (LabelEncoder & one-hot).
- Trained three models: Logistic Regression, SVM, Random Forest.
- Random Forest achieved the best performance (F1 ≈ **0.976**) using lifestyle-only features.

### Milestone 3: Evaluation, Tool Development, & Interpretation
- Evaluated all models on a hold-out test set with **Accuracy**, **Precision**, **Recall**, and **F1 Score**.
  - Full-feature Random Forest Accuracy: **99.9%**
  - Lifestyle-only Random Forest Accuracy: **97.9%**
- Identified top predictive features: **Cholesterol**, **Systolic BP**, **Stress Level**, **Heart Resilience Score**.
- Developed a **Streamlit dashboard** offering:
  - Real-time risk prediction and probability
  - Comparison chart of user metrics vs ideal benchmarks
  - Input summary table for full transparency

## Repository Info

- **Repository Name**: `cap5771sp25-project`
- **Collaborators Invited**: TAs, graders, and instructors are added and Gradescope linked.

## Final Note

This project was completed individually, covering data collection, analysis, feature engineering, model development, dashboard implementation, and report writing.

