# blynk_integration.py
import requests
import os
import joblib
import numpy as np
import pandas as pd
from utils.is_water_safe import simple_is_water_safe, advanced_is_water_safe

TOKEN = "4Y-YI7wxxHsSIrK3FCTZFDScQwq62XTH"  # replace with your token
BASE_URL = "https://blynk.cloud/external/api"

MODEL_DIR = "models_output"

def read_blynk(vpin):
    try:
        r = requests.get(f"{BASE_URL}/get?token={TOKEN}&{vpin}", timeout=5)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print("Blynk read error:", e)
        return None

def write_blynk(vpin, value):
    try:
        requests.get(f"{BASE_URL}/update?token={TOKEN}&{vpin}={value}", timeout=5)
    except Exception as e:
        print("Blynk write error:", e)

# Attempt to load models
use_ml = False
try:
    safety_model = joblib.load(os.path.join(MODEL_DIR, "water_safety_rf.joblib"))
    disease_model = joblib.load(os.path.join(MODEL_DIR, "water_diseases_multi_rf.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    feature_names = list(scaler.feature_names_in_)
    use_ml = True
    print("Loaded ML models.")
except Exception as e:
    print("ML models not loaded, will use rule-based fallback:", e)
    safety_model = None
    disease_model = None
    scaler = None
    feature_names = []

# Read usual sensor vpins (change according to your setup)
# Example mapping:
# v1 => pH
# v2 => Turbidity
# v3 => Temperature
# v4 => Flow
# v5 => Pump state
# v6 => Chlorine
# v7 => E.coli / Total Coliform
# v8 => Nitrates

raw = {}
for pin in ["v1","v2","v3","v4","v5","v6","v7","v8"]:
    val = read_blynk(pin)
    if val is not None and val != "":
        try:
            raw[pin] = float(val)
        except:
            raw[pin] = val

# Map readings to named inputs (adjust names to your sensors)
pH = raw.get("v1", None)
turbidity = raw.get("v2", None)
temperature = raw.get("v3", None)
flow = raw.get("v4", None)
pump = raw.get("v5", None)
chlorine = raw.get("v6", None)
ecoli = raw.get("v7", None)
nitrates = raw.get("v8", None)

# Try ML prediction if available
if use_ml:
    # Build a DataFrame aligning with model's features; missing features set to median-like defaults (0)
    row = {f: 0.0 for f in feature_names}
    # Attempt to fill common names if present
    mappings = {
        "pH": pH,
        "Turbidity (NTU)": turbidity,
        "Temperature": temperature,
        "Flow": flow,
        "Pump": pump,
        "Chlorine": chlorine,
        "Total Coliform (MPN/100ml)": ecoli,
        "E.coli": ecoli,
        "Nitrates": nitrates
    }
    for fname in feature_names:
        if fname in mappings and mappings[fname] is not None:
            row[fname] = mappings[fname]
    X = pd.DataFrame([row], columns=feature_names).fillna(0.0)
    X_scaled = scaler.transform(X)
    ml_safe = int(safety_model.predict(X_scaled)[0])
    ml_diseases = disease_model.predict(X_scaled)[0].tolist()
    print("ML safety:", ml_safe, "ML diseases:", ml_diseases)
else:
    # fallback simple rule
    if pH is None or turbidity is None or temperature is None:
        print("Missing sensors for simple rule fallback. Aborting.")
        ml_safe = None
        ml_diseases = [None, None, None, None, None]
    else:
        ml_safe = simple_is_water_safe(pH, turbidity, temperature)
        # No ML disease predictions available
        ml_diseases = [None, None, None, None, None]

# Advanced rule check
adv_row = {
    "pH": pH,
    "turbidity": turbidity,
    "chlorine": chlorine,
    "ecoli": ecoli,
    "nitrates": nitrates
}
adv_safe, adv_reasons = advanced_is_water_safe(adv_row)
print("Advanced rule:", adv_safe, adv_reasons)

# Send results back to Blynk
# v10 -> safety (ML or simple)
# v11 -> advanced rule safety
# v12..v16 -> disease flags (ML) in order: Diarrhea,Cholera,Typhoid,Gastroenteritis,Chemical_Illness
if ml_safe is not None:
    write_blynk("v10", int(ml_safe))
write_blynk("v11", int(adv_safe))
if ml_diseases:
    for i, pin in enumerate(["v12","v13","v14","v15","v16"]):
        val = ml_diseases[i]
        if val is None:
            write_blynk(pin, 0)
        else:
            write_blynk(pin, int(val))

print("Blynk update done.")
