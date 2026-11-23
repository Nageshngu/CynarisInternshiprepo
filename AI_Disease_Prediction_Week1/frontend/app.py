"""Week 9 Streamlit client for the FastAPI prediction service."""
import os
import requests
import streamlit as st

st.set_page_config(page_title="HealthBridge Symptom Checker", page_icon="🩺")
st.title("HealthBridge — Educational Symptom Checker")
st.warning("Research demonstration only. Not for diagnosis or treatment. For emergencies, seek immediate medical care.")
api_url = os.environ.get("HEALTHBRIDGE_API_URL", "http://127.0.0.1:8000").rstrip("/")
with st.form("symptoms"):
    age = st.number_input("Age", 0, 120, 35)
    gender = st.selectbox("Gender (dataset categories)", ["F", "M"])
    symptoms = {k: int(st.checkbox(k.replace("_", " "))) for k in ["Fever", "Cough", "Fatigue", "Breathlessness", "Chest_Pain", "Nausea", "Headache"]}
    bmi = st.number_input("BMI", 1.0, 100.0, 24.0)
    systolic = st.number_input("Systolic BP", 1.0, 300.0, 120.0)
    diastolic = st.number_input("Diastolic BP", 1.0, 200.0, 80.0)
    spo2 = st.number_input("SpO2 (%)", 0.0, 100.0, 98.0)
    glucose = st.number_input("Glucose (mg/dL)", 1.0, 1000.0, 95.0)
    hba1c = st.number_input("HbA1c (%)", 0.1, 25.0, 5.2)
    submitted = st.form_submit_button("Check symptoms")
if submitted:
    payload = {"Age": age, "Gender": gender, **symptoms, "BMI": bmi,
               "BP_Systolic": systolic, "BP_Diastolic": diastolic, "SpO2": spo2,
               "Glucose_mg_dL": glucose, "HbA1c": hba1c}
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.subheader("Model output (not a diagnosis)")
        st.write(result["predicted_condition"])
        st.dataframe(result["model_scores"], use_container_width=True)
        st.warning(result["safety_message"])
        st.caption(result["disclaimer"])
    except requests.RequestException as exc:
        st.error(f"Cannot contact prediction API: {exc}. Start FastAPI first.")
