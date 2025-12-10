import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go
import os
import requests
from collections import deque
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Smart Water Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for Enhanced Frontend
# -----------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #00acc1;
        --success-color: #43a047;
        --danger-color: #e53935;
        --warning-color: #fb8c00;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e88e5 0%, #00acc1 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #1e88e5;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f5f5f5 0%, white 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1e88e5;
        margin: 2rem 0 1rem 0;
    }

    .section-header h2 {
        margin: 0;
        color: #1e88e5;
        font-size: 1.5rem;
    }

    /* Status badges */
    .status-safe {
        background: linear-gradient(135deg, #43a047 0%, #66bb6a 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(67,160,71,0.3);
        margin: 1rem 0;
    }

    .status-unsafe {
        background: linear-gradient(135deg, #e53935 0%, #ef5350 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(229,57,53,0.3);
        margin: 1rem 0;
    }

    /* Disease alert box */
    .disease-alert {
        background: #fff3e0;
        border-left: 5px solid #fb8c00;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Input section */
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e88e5 0%, #00acc1 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #1e88e5;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }

    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
    }

    /* CSV section */
    .csv-section {
        background: #e8f5e9;
        border-left: 4px solid #43a047;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Model Paths
# -----------------------------
MODEL_DIR = "models_output"
SAFETY_MODEL_PATH = os.path.join(MODEL_DIR, "water_safety_rf.joblib")
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "water_diseases_multi_rf.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

# -----------------------------
# CSV Storage Path
# -----------------------------
CSV_FILE = "blynk_water_data.csv"


# -----------------------------
# CSV Functions
# -----------------------------
def save_to_csv(timestamp, ph, turbidity, temperature, tds, is_safe):
    """Save sensor data to CSV file"""
    data = {
        'Timestamp': [timestamp],
        'pH': [ph],
        'Turbidity_NTU': [turbidity],
        'Temperature_C': [temperature],
        'TDS_mg_L': [tds],
        'Is_Safe': [is_safe]
    }

    df = pd.DataFrame(data)

    # Check if file exists
    if os.path.exists(CSV_FILE):
        # Append to existing file
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        # Create new file with headers
        df.to_csv(CSV_FILE, mode='w', header=True, index=False)


def load_csv_data():
    """Load data from CSV file"""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return None


def clear_csv_data():
    """Clear all data from CSV file"""
    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)
        return True
    return False


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
    # Determine color based on parameter and value
    if "pH" in title:
        color = "green" if 6.5 <= value <= 8.5 else "red"
    elif "Turbidity" in title:
        color = "green" if value < 5 else "orange" if value < 10 else "red"
    elif "Temperature" in title:
        color = "green" if 10 <= value <= 30 else "orange"
    elif "TDS" in title:
        color = "green" if value < 500 else "orange" if value < 1000 else "red"
    else:
        color = "blue"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": title, "font": {"size": 20, "color": "#1e88e5"}},
        delta={"reference": (max_v + min_v) / 2},
        gauge={
            "axis": {"range": [min_v, max_v]},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e0e0e0",
            "steps": [
                {"range": [min_v, (max_v + min_v) / 2], "color": "#e3f2fd"},
                {"range": [(max_v + min_v) / 2, max_v], "color": "#bbdefb"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": max_v * 0.9
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif"}
    )
    return fig


# -----------------------------
# Trend Chart
# -----------------------------
def plot_line(name, values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=list(values),
        mode="lines+markers",
        line=dict(color="#1e88e5", width=3),
        marker=dict(size=6, color="#00acc1"),
        fill='tozeroy',
        fillcolor='rgba(30, 136, 229, 0.1)'
    ))
    fig.update_layout(
        title={"text": name, "font": {"size": 18, "color": "#1e88e5"}},
        height=280,
        margin=dict(l=40, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            title="Reading #"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            title="Value"
        )
    )
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

    return safe_pred


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1>💧 Smart Water Quality Dashboard</h1>
    <p>Real-time monitoring and water quality predictions</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Info
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/water.png", width=80)
    st.markdown("### 📊 Dashboard Info")
    st.info("""
    This dashboard monitors water quality parameters and uses machine learning to predict:
    - Water safety status
    - Real-time parameter trends
    """)

    st.markdown("### 📈 Parameter Guidelines")
    st.markdown("""
    **pH**: 6.5 - 8.5 (Safe)  
    **Turbidity**: < 5 NTU (Good)  
    **Temperature**: 10-30°C (Optimal)  
    **TDS**: < 500 mg/L (Excellent)
    """)

    st.markdown("### 💾 Data Storage")
    csv_data = load_csv_data()
    if csv_data is not None:
        st.success(f"📁 {len(csv_data)} records stored")
    else:
        st.info("No data stored yet")

    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [WHO Guidelines](https://www.who.int)
    - [Water Quality Standards](https://www.epa.gov)
    """)

# -----------------------------
# Manual Sensor Input
# -----------------------------
st.markdown('<div class="section-header"><h2>📟 Manual Sensor Input</h2></div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ph = st.number_input("🧪 pH Level", 0.0, 14.0, 7.0, 0.1, help="Normal range: 6.5-8.5")
    with col2:
        turb = st.number_input("🌫️ Turbidity (NTU)", 0.0, 100.0, 2.0, 0.1, help="Lower is clearer")
    with col3:
        temp = st.number_input("🌡️ Temperature (°C)", 0.0, 100.0, 25.0, 0.1, help="Optimal: 10-30°C")
    with col4:
        tds = st.number_input("💧 TDS (mg/L)", 0.0, 2000.0, 300.0, 10.0, help="Total Dissolved Solids")

    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Predict Water Quality", key="predict_manual"):
    safe = predict_quality(ph, turb, temp, tds)

    # Save to history
    st.session_state.history["pH"].append(ph)
    st.session_state.history["Turbidity"].append(turb)
    st.session_state.history["Temperature"].append(temp)
    st.session_state.history["TDS"].append(tds)

    # Gauges with enhanced styling
    st.markdown('<div class="section-header"><h2>📊 Current Readings</h2></div>', unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(gauge("pH", ph, 0, 14), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(gauge("Turbidity", turb, 0, 100), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(gauge("Temperature °C", temp, 0, 50), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with g4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(gauge("TDS (mg/L)", tds, 0, 2000), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced status display
    st.markdown('<div class="section-header"><h2>🎯 Analysis Results</h2></div>', unsafe_allow_html=True)

    if safe == 1:
        st.markdown('<div class="status-safe">✅ WATER IS SAFE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-unsafe">🚨 WARNING: WATER NOT SAFE</div>', unsafe_allow_html=True)

# -----------------------------
# Trends
# -----------------------------
st.markdown('<div class="section-header"><h2>📈 Historical Trends</h2></div>', unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)
with t1:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_line("Temperature", st.session_state.history["Temperature"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with t2:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_line("TDS", st.session_state.history["TDS"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with t3:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_line("Turbidity", st.session_state.history["Turbidity"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with t4:
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(plot_line("pH", st.session_state.history["pH"]), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Live Blynk Data
# -----------------------------
st.markdown('<div class="section-header"><h2>📡 IoT Sensor Integration (Blynk)</h2></div>', unsafe_allow_html=True)

BLYNK_TOKEN = "4Y-YI7wxxHsSIrK3FCTZFDScQwq62XTH"

col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown("""
    <div class="info-box">
        <strong>🔗 Connect your IoT sensors</strong><br>
        Fetch real-time data from your Blynk-connected water quality sensors.
        Data will be automatically saved to CSV.
    </div>
    """, unsafe_allow_html=True)

if st.button("📥 Fetch Live Data (Secret Token)", key="fetch_secret"):
    if BLYNK_TOKEN == "":
        st.warning("⚠️ No secret token found. Use manual token field below.")
    else:
        with st.spinner("Fetching data from Blynk..."):
            try:
                url = f"https://blynk.cloud/external/api/get?token={BLYNK_TOKEN}&v1&v2&v3&v4&format=json"
                res = requests.get(url, timeout=5).json()

                ph_val = float(res.get("v4"))
                turb_val = float(res.get("v3"))
                temp_val = float(res.get("v1"))
                tds_val = float(res.get("v2"))

                safe = predict_quality(ph_val, turb_val, temp_val, tds_val)

                # Save to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_csv(timestamp, ph_val, turb_val, temp_val, tds_val, safe)

                # Save history
                st.session_state.history["pH"].append(ph_val)
                st.session_state.history["Turbidity"].append(turb_val)
                st.session_state.history["Temperature"].append(temp_val)
                st.session_state.history["TDS"].append(tds_val)

                st.success(f"✅ Data saved to CSV at {timestamp}")

                st.markdown('<div class="section-header"><h2>📊 Live Sensor Readings</h2></div>', unsafe_allow_html=True)

                g1, g2, g3, g4 = st.columns(4)
                with g1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("pH", ph_val, 0, 14), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("Turbidity", turb_val, 0, 100), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g3:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("Temperature °C", temp_val, 0, 50), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g4:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("TDS (mg/L)", tds_val, 0, 2000), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if safe == 1:
                    st.markdown('<div class="status-safe">✅ WATER IS SAFE FOR CONSUMPTION</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-unsafe">🚨 WARNING: WATER NOT SAFE</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")

# -----------------------------
# Manual Token Fetch
# -----------------------------
st.markdown("---")
st.markdown("### 🔑 Or Use Custom Token")

token = st.text_input("Enter Blynk Token:", type="password", help="Your private Blynk authentication token")

if st.button("📥 Fetch Live Data (Manual Token)", key="fetch_manual"):
    if token.strip() == "":
        st.error("⚠️ Please enter a valid token.")
    else:
        with st.spinner("Fetching data from Blynk..."):
            try:
                url = f"https://blynk.cloud/external/api/get?token={token}&v1&v2&v3&v4&format=json"
                res = requests.get(url, timeout=5).json()

                ph_val = float(res.get("v4"))
                turb_val = float(res.get("v3"))
                temp_val = float(res.get("v1"))
                tds_val = float(res.get("v2"))

                safe = predict_quality(ph_val, turb_val, temp_val, tds_val)

                # Save to CSV
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_csv(timestamp, ph_val, turb_val, temp_val, tds_val, safe)

                # Save history
                st.session_state.history["pH"].append(ph_val)
                st.session_state.history["Turbidity"].append(turb_val)
                st.session_state.history["Temperature"].append(temp_val)
                st.session_state.history["TDS"].append(tds_val)

                st.success(f"✅ Data saved to CSV at {timestamp}")

                st.markdown('<div class="section-header"><h2>📊 Live Sensor Readings</h2></div>', unsafe_allow_html=True)

                g1, g2, g3, g4 = st.columns(4)
                with g1:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("pH", ph_val, 0, 14), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("Turbidity", turb_val, 0, 100), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g3:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("Temperature °C", temp_val, 0, 50), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with g4:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.plotly_chart(gauge("TDS (mg/L)", tds_val, 0, 2000), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                if safe == 1:
                    st.markdown('<div class="status-safe">✅ WATER IS SAFE FOR CONSUMPTION</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-unsafe">🚨 WARNING: WATER NOT SAFE</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")

# -----------------------------
# CSV Data Management Section
# -----------------------------
st.markdown('<div class="section-header"><h2>💾 Stored Data Management</h2></div>', unsafe_allow_html=True)

csv_data = load_csv_data()

if csv_data is not None:
    st.markdown(f"""
    <div class="csv-section">
        <h3>📁 Data Storage Summary</h3>
        <p><strong>Total Records:</strong> {len(csv_data)}</p>
        <p><strong>File Location:</strong> {CSV_FILE}</p>
        <p><strong>Last Updated:</strong> {csv_data['Timestamp'].iloc[-1] if len(csv_data) > 0 else 'N/A'}</p>
    </div>
    """, unsafe_allow_html=True)

    # Display options
    col_opt1, col_opt2, col_opt3 = st.columns(3)

    with col_opt1:
        show_data = st.checkbox("📋 Show Data Table", value=False)

    with col_opt2:
        show_stats = st.checkbox("📊 Show Statistics", value=False)

    with col_opt3:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if clear_csv_data():
                st.success("✅ All data cleared successfully!")
                st.rerun()

    # Show data table
    if show_data:
        st.markdown("### 📋 Recorded Data")
        st.dataframe(csv_data, use_container_width=True, height=400)

        # Download button
        csv_download = csv_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_download,
            file_name=f"water_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # Show statistics
    if show_stats:
        st.markdown("### 📊 Data Statistics")

        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

        with stat_col1:
            st.metric("Avg pH", f"{csv_data['pH'].mean():.2f}")
            st.metric("Min pH", f"{csv_data['pH'].min():.2f}")
            st.metric("Max pH", f"{csv_data['pH'].max():.2f}")

        with stat_col2:
            st.metric("Avg Turbidity", f"{csv_data['Turbidity_NTU'].mean():.2f}")
            st.metric("Min Turbidity", f"{csv_data['Turbidity_NTU'].min():.2f}")
            st.metric("Max Turbidity", f"{csv_data['Turbidity_NTU'].max():.2f}")

        with stat_col3:
            st.metric("Avg Temperature", f"{csv_data['Temperature_C'].mean():.2f}°C")
            st.metric("Min Temperature", f"{csv_data['Temperature_C'].min():.2f}°C")
            st.metric("Max Temperature", f"{csv_data['Temperature_C'].max():.2f}°C")

        with stat_col4:
            st.metric("Avg TDS", f"{csv_data['TDS_mg_L'].mean():.2f}")
            st.metric("Min TDS", f"{csv_data['TDS_mg_L'].min():.2f}")
            st.metric("Max TDS", f"{csv_data['TDS_mg_L'].max():.2f}")

        # Safety statistics
        safe_count = (csv_data['Is_Safe'] == 1).sum()
        unsafe_count = (csv_data['Is_Safe'] == 0).sum()

        st.markdown("### 🎯 Safety Analysis")
        safety_col1, safety_col2 = st.columns(2)

        with safety_col1:
            st.metric("Safe Readings", safe_count, delta=f"{(safe_count / len(csv_data) * 100):.1f}%")

        with safety_col2:
            st.metric("Unsafe Readings", unsafe_count, delta=f"{(unsafe_count / len(csv_data) * 100):.1f}%",
                      delta_color="inverse")

else:
    st.markdown("""
    <div class="info-box">
        <h3>📭 No Data Stored Yet</h3>
        <p>Fetch data from Blynk sensors to start recording measurements.</p>
        <p>All Blynk data will be automatically saved to <strong>blynk_water_data.csv</strong></p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    <p>💧 Smart Water Quality Dashboard | Powered by Machine Learning & IoT</p>
    <p>Monitor, Predict, Protect - Ensuring safe water for everyone</p>
</div>
""", unsafe_allow_html=True)