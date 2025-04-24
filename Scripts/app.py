import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load full-feature model and scaler
model = joblib.load("model/rf_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Full feature list used for prediction
features = [
    'chol', 'trestbps', 'Water_Intake (liters)', 'thalach', 'age',
    'Sleep Duration', 'heart_resilience_score', 'Diastolic_BP',
    'Systolic_BP', 'Stress Level', 'gender', 'Physical Activity Level',
    'Heart Rate', 'lifestyle_recovery_index', 'Calories_Burned', 'BMI_y',
    'Quality of Sleep', 'Workout_Frequency (days/week)', 'experience_level_encoded'
]

st.set_page_config(page_title="Heart Risk Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")

def get_user_input():
    st.markdown("### Enter Your Details")

    chol = st.number_input("Cholesterol (mg/dL)", 100, 500, 210)
    trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 125)
    water = st.number_input("Water Intake (liters)", 0.0, 10.0, 2.5)
    thalach = st.slider("Max Heart Rate", 100, 220, 165)
    age = st.slider("Age", 20, 90, 45)
    sleep = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0)
    heart_rate = st.slider("Resting Heart Rate", 50, 140, 72)
    steps = st.slider("Daily Steps", 1000, 20000, 7000)
    diastolic = st.number_input("Diastolic BP", 50, 120, 82)
    systolic = st.number_input("Systolic BP", 90, 200, 130)
    stress = st.slider("Stress Level", 1, 10, 5)
    gender = st.radio("Gender", ["Male", "Female"])
    physical = st.slider("Physical Activity Level (1-5)", 1, 5, 4)
    calories = st.number_input("Calories Burned", 100, 1000, 450)
    bmi = st.number_input("BMI", 15, 40, 24)
    quality = st.slider("Quality of Sleep (1-5)", 1, 5, 4)
    workout_days = st.slider("Workout Days/Week", 0, 7, 3)
    experience = st.selectbox("Workout Experience", ["Beginner", "Intermediate", "Advanced"])

    experience_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    experience_level_encoded = experience_map[experience]
    gender_binary = 1 if gender == "Male" else 0

    heart_resilience_score = (thalach / heart_rate) * (steps / 1000)
    lifestyle_recovery_index = (sleep * calories) / (stress + 1)

    data = pd.DataFrame([[
        chol, trestbps, water, thalach, age,
        sleep, heart_resilience_score, diastolic,
        systolic, stress, gender_binary, physical,
        heart_rate, lifestyle_recovery_index, calories, bmi,
        quality, workout_days, experience_level_encoded
    ]], columns=features)

    return data

input_data = get_user_input()

if st.button("Predict Heart Risk"):
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction Result")
    st.success("Not at Risk" if prediction == 0 else "At Risk of Heart Disease")
    st.markdown(f"**Probability of Heart Risk:** `{proba * 100:.2f}%`")

    st.markdown("### Your Input Summary")
    st.dataframe(input_data)

    st.markdown("### You vs. Ideal Profile")
    compare_features = [
        "Sleep Duration", "Stress Level", "Calories_Burned",
        "Water_Intake (liters)", "Heart Rate", "BMI_y", "Physical Activity Level",
        "chol", "trestbps", "Systolic_BP", "Diastolic_BP"
    ]

    ideal_values = {
        "Sleep Duration": 8,
        "Stress Level": 3,
        "Calories_Burned": 600,
        "Water_Intake (liters)": 3.0,
        "Heart Rate": 65,
        "BMI_y": 22,
        "Physical Activity Level": 4,
        "chol": 180,
        "trestbps": 120,
        "Systolic_BP": 120,
        "Diastolic_BP": 80
    }

    user_values = {feature: input_data[feature].iloc[0] for feature in compare_features}

    compare_df = pd.DataFrame({
        "Feature": compare_features,
        "Ideal": [ideal_values[feat] for feat in compare_features],
        "You": [user_values[feat] for feat in compare_features]
    })

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    width = 0.35
    x = np.arange(len(compare_df["Feature"]))

    ax2.bar(x - width/2, compare_df["Ideal"], width, label="Ideal", color="skyblue")
    ax2.bar(x + width/2, compare_df["You"], width, label="You", color="orange")

    ax2.set_xticks(x)
    ax2.set_xticklabels(compare_df["Feature"], rotation=45, ha="right")
    ax2.set_ylabel("Value")
    ax2.set_title("Comparison: You vs Ideal")
    ax2.legend()
    st.pyplot(fig2)


