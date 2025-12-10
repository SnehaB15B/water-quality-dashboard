import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import os
import requests
from collections import deque

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Smart Water Quality Dashboard", layout="wide")

# -----------------------------
# Model Paths
# -----------------------------
MODEL_DIR = "models_output"
SAFETY_MODEL_PATH = os.path.join(MODEL_DIR, "water_safety_rf.joblib")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "water_diseases_multi_rf.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

# -----------------------------
# Safe Secret Loader
# -----------------------------
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return ""

# -----------------------------
# Load ML Models
# -----------------------------
@st.cache_resource
def load_models():
    safety = joblib.load(SAFETY_MODEL_PATH)
    disease = joblib.load(DISEASE_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH) as f:
        feature_names = json.load(f)
    return safety, disease, scaler, feature_names

safety_model, disease_model, scaler, feature_names = load_models()

# -----------------------------
# Session Trend History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = {
        "pH": deque(maxlen=50),
        "Turbidity": deque(maxlen=50),
        "Temperature": deque(maxlen=50),
        "TDS": deque(maxlen=50)
    }

# -----------------------------
# Gauge Chart
# -----------------------------
def gauge(title, value, min_v, max_v):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={"axis": {"range": [min_v, max_v]}, "bar": {"color": "blue"}},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
    return fig

# -----------------------------
# Trend Chart
# -----------------------------
def plot_line(name, values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=list(values), mode="lines+markers"))
    fig.update_layout(title=name, height=250, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# -----------------------------
# Prediction Logic
# -----------------------------
def predict_quality(ph, turb, temp, tds):
    row = pd.Series({
        "pH": ph,
        "Turbidity (NTU)": turb,
        "Temperature (°C)": temp,
        "TDS (mg/L)": tds
    })

    # Safe feature ordering
    for col in feature_names:
        if col not in row:
            row[col] = 0
    row = row[feature_names]

    scaled = scaler.transform([row])

    safe_pred = safety_model.predict(scaled)[0]
    disease_pred = disease_model.predict(scaled)[0]

    diseases = ["Diarrhea", "Cholera", "Typhoid", "Gastroenteritis", "Chemical_Illness"]
    detected = [diseases[i] for i, v in enumerate(disease_pred) if v == 1]

    return safe_pred, detected

# -----------------------------
# Header
# -----------------------------
st.title("💧 Smart Water Quality Dashboard")
st.markdown("Manual and live IoT sensor inputs with ML predictions.")

# -----------------------------
# Manual Sensor Input
# -----------------------------
st.subheader("📟 Manual Sensor Input")

col1, col2, col3, col4 = st.columns(4)
with col1:
    ph = st.number_input("pH", 0.0, 14.0, 7.0, 0.1)
with col2:
    turb = st.number_input("Turbidity (NTU)", 0.0, 100.0, 2.0, 0.1)
with col3:
    temp = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0, 0.1)
with col4:
    tds = st.number_input("TDS (mg/L)", 0.0, 2000.0, 300.0, 10.0)

if st.button("🔍 Predict Water Quality"):
    safe, diseases = predict_quality(ph, turb, temp, tds)

    # Save to history
    st.session_state.history["pH"].append(ph)
    st.session_state.history["Turbidity"].append(turb)
    st.session_state.history["Temperature"].append(temp)
    st.session_state.history["TDS"].append(tds)

    # Gauges
    g1, g2, g3, g4 = st.columns(4)
    g1.plotly_chart(gauge("pH", ph, 0, 14))
    g2.plotly_chart(gauge("Turbidity", turb, 0, 100))
    g3.plotly_chart(gauge("Temperature °C", temp, 0, 50))
    g4.plotly_chart(gauge("TDS (mg/L)", tds, 0, 2000))

    st.subheader("📌 Water Safety")
    st.success("✅ SAFE" if safe == 1 else "🚨 NOT SAFE")


# -----------------------------
# Trends
# -----------------------------
st.divider()
st.subheader("📈 Live Trends")

t1, t2, t3, t4 = st.columns(4)
t1.plotly_chart(plot_line("Temperature", st.session_state.history["Temperature"]))
t2.plotly_chart(plot_line("TDS", st.session_state.history["TDS"]))
t3.plotly_chart(plot_line("Turbidity", st.session_state.history["Turbidity"]))
t4.plotly_chart(plot_line("pH", st.session_state.history["pH"]))

# -----------------------------
# Live Blynk Data
# -----------------------------
st.divider()
st.subheader("📡 Blynk Live Sensor Readings")

BLYNK_TOKEN = "4Y-YI7wxxHsSIrK3FCTZFDScQwq62XTH"


if st.button("📥 Fetch Live Data (Secret Token)"):
    if BLYNK_TOKEN == "":
        st.warning("No secret token found. Use manual token field below.")
    else:
        try:
            url = f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&v1&v2&v3&v4&format=json"
            res = requests.get(url, timeout=5).json()

            ph_val = float(res.get("v4"))
            turb_val = float(res.get("v3"))
            temp_val = float(res.get("v1"))
            tds_val = float(res.get("v2"))

            safe, diseases = predict_quality(ph_val, turb_val, temp_val, tds_val)

            # Save history
            st.session_state.history["pH"].append(ph_val)
            st.session_state.history["Turbidity"].append(turb_val)
            st.session_state.history["Temperature"].append(temp_val)
            st.session_state.history["TDS"].append(tds_val)

            g1, g2, g3, g4 = st.columns(4)
            g1.plotly_chart(gauge("pH", ph_val, 0, 14))
            g2.plotly_chart(gauge("Turbidity", turb_val, 0, 100))
            g3.plotly_chart(gauge("Temperature °C", temp_val, 0, 50))
            g4.plotly_chart(gauge("TDS (mg/L)", tds_val, 0, 2000))

            st.success("✅ SAFE" if safe == 1 else "🚨 NOT SAFE")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# -----------------------------
# Manual Token Fetch
# -----------------------------
token = st.text_input("Enter Blynk Token (optional):")

if st.button("📥 Fetch Live Data (Manual Token)"):
    if token.strip() == "":
        st.error("Please enter a valid token.")
    else:
        try:
            url = f"https://blynk.cloud/external/api/get?token={token}&v1&v2&v3&v4&format=json"
            res = requests.get(url, timeout=5).json()

            ph_val = float(res.get("v4"))
            turb_val = float(res.get("v3"))
            temp_val = float(res.get("v1"))
            tds_val = float(res.get("v2"))

            safe, diseases = predict_quality(ph_val, turb_val, temp_val, tds_val)

            # Save history
            st.session_state.history["pH"].append(ph_val)
            st.session_state.history["Turbidity"].append(turb_val)
            st.session_state.history["Temperature"].append(temp_val)
            st.session_state.history["TDS"].append(tds_val)

            g1, g2, g3, g4 = st.columns(4)
            g1.plotly_chart(gauge("pH", ph_val, 0, 14))
            g2.plotly_chart(gauge("Turbidity", turb_val, 0, 100))
            g3.plotly_chart(gauge("Temperature °C", temp_val, 0, 50))
            g4.plotly_chart(gauge("TDS (mg/L)", tds_val, 0, 2000))

            st.success("✅ SAFE" if safe == 1 else "🚨 NOT SAFE")

        except Exception as e:
            st.error(f"❌ Error: {e}")
