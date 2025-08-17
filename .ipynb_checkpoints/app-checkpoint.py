# app.py
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "final_score_model.pkl"
model = joblib.load(MODEL_PATH)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Student Final Score Predictor 🎓")
st.write("Predict the Final Score of a student based on Hours Studied.")

# User input
hours = st.number_input(
    "Enter Hours Studied:",
    min_value=0.0,
    max_value=24.0,
    step=0.5
)

# Predict button
if st.button("Predict"):
    prediction = model.predict([[hours]])  # must be 2D array
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

# Optional: show dataset
if st.checkbox("Show Dataset"):
    df = pd.read_csv(r"C:\Users\shrey\Students_Score_Prediction\data\Students Performance.csv")
    st.dataframe(df)

