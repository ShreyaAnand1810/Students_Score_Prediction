# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load trained model
MODEL_PATH = r"C:\Users\shrey\Students_Score_Prediction\final_score_model.pkl"
model = joblib.load(MODEL_PATH)

st.title("Student Final Score Predictor 🎓")

# User inputs
attendance = st.number_input("Attendance %:", 0, 100, 80)
hours = st.number_input("Hours Studied:", 0.0, 24.0, 4.0)

if st.button("Predict"):
    # Prediction
    input_features = [[ attendance, hours]]
    prediction = model.predict(input_features)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

    # Optional: Show error metrics on test set
    df = pd.read_csv(r"C:\Users\shrey\Students_Score_Prediction\data\Students Performance.csv")
    X = df[[ 'Attendance', 'Hours_Studied']]
    y = df['Final_Score']
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.write("### Model Performance on Dataset")
    st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
