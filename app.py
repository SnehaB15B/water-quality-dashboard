import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Water Quality Dashboard", page_icon="💧", layout="wide")

# -----------------------------
# Config
# -----------------------------
TOKEN = "p4YhF8J1abcD93kL5pqQnnCdg4h29k7x"
CSV_FILE = "sensor_data.csv"
SAFETY_MODEL_PATH = "models_output/water_safety_rf.joblib"
DISEASE_MODEL_PATH = "models_output/water_diseases_multi_rf.joblib"
SCALER_PATH = "models_output/scaler.joblib"


# -----------------------------
# Load ML models
# -----------------------------
@st.cache_resource
def load_models():
    safety_clf = joblib.load(SAFETY_MODEL_PATH)
    disease_clf = joblib.load(DISEASE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return safety_clf, disease_clf, scaler


safety_model, disease_model, scaler = load_models()


# -----------------------------
# Water Safety Function (ML)
# -----------------------------
def predict_ml(ph, turbidity, temp):
    # Prepare data as DataFrame for ML
    X = pd.DataFrame([[ph, turbidity, temp]], columns=['pH', 'Turbidity (NTU)', 'Temperature'])

    # Fill missing columns for ML
    all_features = scaler.mean_.shape[0]  # number of features used in scaler
    X_full = pd.DataFrame(np.zeros((1, all_features)), columns=[f"feat_{i}" for i in range(all_features)])
    X_scaled = scaler.transform(X_full)  # scaled input

    # Safety prediction
    safety_pred = safety_model.predict(X_scaled)[0]

    # Disease prediction
    disease_pred = disease_model.predict(X_scaled)[0]

    return safety_pred, disease_pred


# -----------------------------
# UI Header
# -----------------------------
st.title("💧 Smart Water Quality Dashboard")
st.markdown("Enter readings or fetch live sensor values and get ML predictions for water safety and diseases.")

# -----------------------------
# Input Section
# -----------------------------
st.subheader("📟 Enter Sensor Readings Manually")
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
with col2:
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)
with col3:
    temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1)

# -----------------------------
# ML Prediction
# -----------------------------
if st.button("Check Water Quality & Predict"):
    try:
        # Predict using ML
        safety_pred, disease_pred = predict_ml(ph, turbidity, temp)

        # Safety output
        if safety_pred == 1:
            st.success("✅ Water is SAFE for drinking (ML Prediction)")
        else:
            st.error("❌ Water is NOT SAFE (ML Prediction)")

        # Disease output
        diseases = ["Diarrhea", "Cholera", "Typhoid", "Gastroenteritis", "Chemical_Illness"]
        disease_dict = {d: disease_pred[i] for i, d in enumerate(diseases)}
        st.subheader("⚠️ Predicted Diseases Risk")
        for d, val in disease_dict.items():
            if val == 1:
                st.warning(f"- {d} risk detected")
            else:
                st.info(f"- {d}: no risk")

        # Send safety to Blynk
        try:
            requests.get(f"https://blynk.cloud/external/api/update?token={TOKEN}&v2={safety_pred}")
            st.info("📡 Safety prediction sent to Blynk")
        except:
            st.warning("⚠ Could not send to Blynk")

        # -----------------------------
        # Bar Chart Visualization
        # -----------------------------
        st.subheader("📊 Sensor Levels")
        sensors = ["pH", "Turbidity", "Temperature"]
        values = [ph, turbidity, temp]
        colors = ["#636EFA", "#EF553B", "#00CC96"]

        fig = go.Figure(go.Bar(x=sensors, y=values, marker_color=colors, text=values, textposition='auto'))
        fig.update_layout(title="Current Sensor Levels", yaxis_title="Value", xaxis_title="Sensor")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# -----------------------------
# Fetch Live Sensor from Blynk
# -----------------------------
st.divider()
st.subheader("📡 Fetch Live Sensor from Blynk (V1)")

if st.button("Fetch from Blynk"):
    try:
        value = requests.get(f"https://blynk.cloud/external/api/get?token={TOKEN}&v1").text
        st.info(f"🔹 Live Sensor Value: {value}")

        # Optionally parse CSV-like string if Blynk sends multiple readings
        # For example: "7.2,2.5,25"
        parts = value.split(",")
        if len(parts) == 3:
            ph, turbidity, temp = map(float, parts)
            st.success(f"Using fetched values: pH={ph}, Turbidity={turbidity}, Temp={temp}")

            # Auto-predict ML on fetched data
            safety_pred, disease_pred = predict_ml(ph, turbidity, temp)
            if safety_pred == 1:
                st.success("✅ Water is SAFE (ML Prediction)")
            else:
                st.error("❌ Water is NOT SAFE (ML Prediction)")

    except:
        st.error("❌ Failed to fetch from Blynk")
