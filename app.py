import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --- PREDICT FINAL SCORE (Pretrained Model) ---
st.title("Student Final Score Predictor 🎓")

# Load trained model (update the path if needed)
MODEL_PATH = r"C:\Users\shrey\Students_Score_Prediction\final_score_model.pkl"
model = joblib.load(MODEL_PATH)

st.header("🔮 Predict Final Score")
attendance = st.number_input("Attendance %:", 0, 100, 80)
hours = st.number_input("Hours Studied:", 0.0, 24.0, 4.0)

if st.button("Predict Final Score"):
    input_features = [[attendance, hours]]
    prediction = model.predict(input_features)[0]
    st.success(f"Predicted Final Score: {prediction:.2f}")

    # Optional: Show model performance
    df_eval = pd.read_csv(r"C:\Users\shrey\Students_Score_Prediction\data\Students Performance.csv")
    X_eval = df_eval[['Attendance', 'Hours_Studied']]
    y_eval = df_eval['Final_Score']
    y_pred_eval = model.predict(X_eval)

    mse = mean_squared_error(y_eval, y_pred_eval)
    mae = mean_absolute_error(y_eval, y_pred_eval)
    r2 = r2_score(y_eval, y_pred_eval)

    st.write("### Model Performance on Dataset")
    st.write(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
