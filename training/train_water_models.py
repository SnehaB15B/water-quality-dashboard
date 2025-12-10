import json
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Database for SIH Project.csv")
OUT_DIR = os.path.join(BASE_DIR, "..", "models_output")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
# Clean & fix column names
df.columns = (
    df.columns
    .str.replace("Â°", "°", regex=False)
    .str.replace("  ", " ", regex=False)  # fix double spaces
    .str.strip()
)

print("Cleaned Columns:", df.columns.tolist())

# Fix bad encodings (Â° → °)
df.columns = (
    df.columns
    .str.replace("Â°", "°", regex=False)
    .str.replace("  ", " ", regex=False)  # remove double spaces
    .str.strip()
)
print("CSV Columns:", df.columns.tolist())


if "Unnamed: 3" in df.columns:
    df = df.drop(columns=["Unnamed: 3"])

# -------------------------------
#  FORCE ONLY 4 INPUT FEATURES
# -------------------------------
REQUIRED_FEATURES = ["pH", "Temperature (°C)", "TDS (mg/L)", "Turbidity (NTU)"]



# Ensure all exist
for col in REQUIRED_FEATURES:
    if col not in df.columns:
        print(f"❌ ERROR: Missing required feature: {col}")

numeric_features = REQUIRED_FEATURES
print("FINAL MODEL FEATURES:", numeric_features)

# -------------------------------
#  SAFE LABEL
# -------------------------------
df["safe"] = 1
unsafe_cond = (
    (df["pH"].notna() & ~df["pH"].between(6.5, 8.5)) |
    (df["Turbidity (NTU)"] >= 5) |
    (df["TDS (mg/L)"] >= 1000)
)
df.loc[unsafe_cond, "safe"] = 0

# -------------------------------
#  DISEASE LABELS
# -------------------------------
df["Diarrhea"] = (df["Turbidity (NTU)"] >= 5).astype(int)
df["Cholera"] = (df["Turbidity (NTU)"] >= 5).astype(int)
df["Typhoid"] = (df["Turbidity (NTU)"] >= 5).astype(int)
df["Gastroenteritis"] = (df["pH"] < 6.5).astype(int)
df["Chemical_Illness"] = (df["TDS (mg/L)"] >= 1000).astype(int)

# -------------------------------
#  INPUT X and OUTPUT Y
# -------------------------------
X = df[numeric_features].fillna(df[numeric_features].median())
y_safe = df["safe"]
y_diseases = df[["Diarrhea","Cholera","Typhoid","Gastroenteritis","Chemical_Illness"]]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
#  SAFETY MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_safe, test_size=0.2, random_state=42
)

safety_clf = RandomForestClassifier(n_estimators=200, random_state=42)
safety_clf.fit(X_train, y_train)
test_pred = safety_clf.predict(X_test)

print("=== Safety Model ===")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# -------------------------------
#  MULTI-DISEASE MODEL
# -------------------------------
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_scaled, y_diseases.values, test_size=0.2, random_state=42
)

multi_clf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42)
)
multi_clf.fit(Xd_train, yd_train)

yd_pred = multi_clf.predict(Xd_test)

print("=== Disease Reports ===")
for i, col in enumerate(y_diseases.columns):
    print(f"\n--- {col} ---")
    print(classification_report(yd_test[:, i], yd_pred[:, i]))

# -------------------------------
#  SAVE MODELS + FEATURES
# -------------------------------
joblib.dump(safety_clf, os.path.join(OUT_DIR, "water_safety_rf.joblib"))
joblib.dump(multi_clf, os.path.join(OUT_DIR, "water_diseases_multi_rf.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
    json.dump(numeric_features, f)

print("\n✅ All models saved successfully with ONLY pH, Temperature, TDS, Turbidity!")
