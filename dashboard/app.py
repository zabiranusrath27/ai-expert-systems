# import sys
# import os
# from dotenv import load_dotenv

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)

# import streamlit as st
# from src.inference import predict
# from google import genai  # ✅ Gemini client

# # --- LOAD GEMINI API KEY ---
# load_dotenv()
# API_KEY = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=API_KEY)

# st.set_page_config(page_title="AI Expert System", layout="wide")
# st.title("AI-Driven Expert System for Maintenance Decision Support")

# col1, col2 = st.columns(2)
# with col1:
#     air = st.number_input("Air temperature [K]", 280.0, 400.0, 320.0)
#     proc = st.number_input("Process temperature [K]", 280.0, 400.0, 330.0)
#     rpm = st.number_input("Rotational speed [rpm]", 500.0, 3000.0, 1200.0)

# with col2:
#     torque = st.number_input("Torque [Nm]", 0.0, 120.0, 50.0)
#     wear = st.number_input("Tool wear [min]", 0.0, 300.0, 120.0)
#     m_type = st.selectbox("Machine Type", ["L", "M", "H"])

# if st.button("Predict"):
#     data = {
#         "Air temperature [K]": air,
#         "Process temperature [K]": proc,
#         "Rotational speed [rpm]": rpm,
#         "Torque [Nm]": torque,
#         "Tool wear [min]": wear,
#         "Type": m_type
#     }

#     res = predict(data, os.path.join(ROOT_DIR, "models/model_rf.pkl"))

#     prediction = res['prediction']
#     probability = res['failure_probability']

#     st.subheader("🔍 Model Output")
#     st.write(f"Prediction: **{prediction}**")
#     st.write(f"Failure Probability: {res['failure_probability']:.2f}")

#     # ------------------  GEMINI PROMPT  ------------------
#     prompt = f"""
#     You are an industrial maintenance expert.
#     Based on these machine conditions:

#     • Air Temp: {air} K
#     • Process Temp: {proc} K
#     • Speed: {rpm} rpm
#     • Torque: {torque} Nm
#     • Tool wear: {wear} minutes
#     • Machine type: {m_type}

#     The ML model output:
#     • Predicted Failure Type: {prediction}
#     • Failure probability: {probability:.2f}

#     Give a short, actionable maintenance recommendation.
#     Provide the response in bullet points.
#     """

#     ai_response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt
#     )

#     expert_advice = ai_response.text

#     st.subheader("🤖 Gemini Expert Advice")
#     st.info(expert_advice)

# import sys
# import os

# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, ROOT_DIR)

# import streamlit as st
# from src.inference import predict

# st.set_page_config(page_title="AI Expert System", layout="wide")
# st.title("AI-Driven Expert System for Maintenance Decision Support")

# col1, col2 = st.columns(2)
# with col1:
#     air = st.number_input("Air temperature [K]", 280.0, 400.0, 320.0)
#     proc = st.number_input("Process temperature [K]", 280.0, 400.0, 330.0)
#     rpm = st.number_input("Rotational speed [rpm]", 500.0, 3000.0, 1200.0)
# with col2:
#     torque = st.number_input("Torque [Nm]", 0.0, 120.0, 50.0)
#     wear = st.number_input("Tool wear [min]", 0.0, 300.0, 120.0)
#     m_type = st.selectbox("Machine Type", ["L", "M", "H"])

# if st.button("Predict"):
#     data = {
#         "Air temperature [K]": air,
#         "Process temperature [K]": proc,
#         "Rotational speed [rpm]": rpm,
#         "Torque [Nm]": torque,
#         "Tool wear [min]": wear,
#         "Type": m_type
#     }

#     res = predict(data, os.path.join(ROOT_DIR, "models/model_rf.pkl"))
#     st.subheader("Result")

#     st.write(f"Prediction: {'⚠️ Likely Failure' if res['prediction']==1 else '✅ OK'}")
#     st.write(f"Failure Probability: {res['failure_probability']:.2f}")
#     st.write(f"Expert Advice: {res['advice']}")
#     st.write(f"Rule Fired: {res['rule_fired']}")

import sys
import os
import streamlit as st
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)
from inference import predict

# ---- PATH FIX ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# ---- PAGE CONFIG ----
st.set_page_config(page_title="AI Expert System", layout="wide")
st.title("AI-Driven Expert System for Maintenance Decision Support")



# ---- INPUT SECTION ----


col1, col2 = st.columns(2)
with col1:
    air = st.number_input("Air temperature [K]", 280.0, 400.0, 320.0)
    proc = st.number_input("Process temperature [K]", 280.0, 400.0, 330.0)
    rpm = st.number_input("Rotational speed [rpm]", 500.0, 3000.0, 1200.0)

with col2:
    torque = st.number_input("Torque [Nm]", 0.0, 120.0, 50.0)
    wear = st.number_input("Tool wear [min]", 0.0, 300.0, 120.0)
    m_type = st.selectbox("Machine Type", ["L", "M", "H"])

# ---- MODEL SELECTION ----
# model_choice = st.radio("Select Model", ["Random Forest", "XGBoost"], horizontal=True)
# model_file = "models/model_rf.pkl" if model_choice == "Random Forest" else "models/model_xgb.pkl"

# ---- PREDICT BUTTON ----
if st.button("🔍 Predict"):
    data = {
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        "Type": m_type
    }

    res = predict(data)

    # ---- DISPLAY RESULTS ----
    st.subheader("📊 Prediction Results")
    st.markdown("---")

    # Probability visualization
    st.write(f"**Failure Probability:** {res['failure_probability']:.2f}")
    st.progress(min(max(res['failure_probability'], 0.0), 1.0))

    # Risk level color coding
    risk_color = {
        "✅ Low Risk": "green",
        "⚠️ Moderate Risk": "orange",
        "🚨 High Failure Risk": "red"
    }[res["risk_label"]]

    st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold'>{res['risk_label']}</span>", unsafe_allow_html=True)

    # Expert explanation
    st.markdown("Model Recommendation")
    st.info(res["expert_recommendation"])

    # Prediction outcome
    st.markdown("### 🔧 Model Decision")
    if res["prediction"] == 0:
        st.success("✅ Machine is operating normally.")
    else:
        st.error("⚠️ Likely Failure detected — maintenance action recommended.")

    
