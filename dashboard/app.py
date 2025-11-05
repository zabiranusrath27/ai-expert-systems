import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


import streamlit as st
from src.inference import predict

st.set_page_config(page_title="AI Expert System", layout="wide")
st.title("AI-Driven Expert System for Maintenance Decision Support")

col1, col2 = st.columns(2)
with col1:
    air = st.number_input("Air temperature [K]", 280.0, 400.0, 320.0)
    proc = st.number_input("Process temperature [K]", 280.0, 400.0, 330.0)
    torque = st.number_input("Torque [Nm]", 0.0, 120.0, 50.0)
    wear = st.number_input("Tool wear [min]", 0.0, 300.0, 120.0)
with col2:
    m_type = st.selectbox("Machine Type", ["L", "M", "H"])

if st.button("Predict"):
    data = {
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Type": m_type
    }
    res = predict(data, os.path.join(ROOT_DIR, "models/model_rf.pkl"))
    st.subheader("Result")
    st.write(f"Prediction: {'⚠️ Likely Failure' if res['prediction']==1 else '✅ OK'}")
    st.write(f"Failure Probability: {res['failure_probability']:.2f}")
    st.write(f"Expert Advice: {res['advice']}")
