# training/train_water_models.py
import pandas as pd
import numpy as np
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# PATH: dataset relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "Database for SIH Project.csv")
OUT_DIR = os.path.join(BASE_DIR, "..", "models_output")
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Load
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df = df.rename(columns=lambda x: x.strip())
if 'Unnamed: 3' in df.columns:
    df = df.drop(columns=['Unnamed: 3'])

# 2) Numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [c for c in numeric_features if c not in ('Completeness_Flag','synthetic_flag')]
print("Numeric features:", numeric_features)

# 3) Create 'safe' label (heuristic)
df['safe'] = 1
conditions_unsafe = (
    (df['pH'].notna() & ~df['pH'].between(6.5,8.5)) |
    (df['Turbidity (NTU)'].notna() & (df['Turbidity (NTU)'] >= 5)) |
    (df['Total Coliform (MPN/100ml)'].notna() & (df['Total Coliform (MPN/100ml)'] > 0)) |
    (df['BOD (mg/L)'].notna() & (df['BOD (mg/L)'] >= 3)) |
    (df['DO (mg/L)'].notna() & (df['DO (mg/L)'] <= 6))
)
df.loc[conditions_unsafe, 'safe'] = 0

# 4) Disease heuristics
df['Diarrhea'] = 0
df['Cholera'] = 0
df['Typhoid'] = 0
df['Gastroenteritis'] = 0
df['Chemical_Illness'] = 0

df.loc[df['Total Coliform (MPN/100ml)'].fillna(0) > 0, ['Diarrhea','Cholera','Typhoid']] = 1
df.loc[df['Turbidity (NTU)'].fillna(0) >= 5, 'Cholera'] = 1
df.loc[(df['BOD (mg/L)'].fillna(0) >= 3) | (df['DO (mg/L)'].fillna(999) <= 6), 'Gastroenteritis'] = 1
df.loc[df['COD (mg/L)'].fillna(0) >= 30, 'Chemical_Illness'] = 1

# 5) Prepare X and y
X = df[numeric_features].copy()
X = X.fillna(X.median())
y_safe = df['safe'].astype(int)
y_diseases = df[['Diarrhea','Cholera','Typhoid','Gastroenteritis','Chemical_Illness']].astype(int)

# 6) Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7) Safety model
X_train, X_test, ytrain_safe, ytest_safe = train_test_split(X_scaled, y_safe, test_size=0.2, random_state=42, stratify=y_safe)
safety_clf = RandomForestClassifier(n_estimators=200, random_state=42)
safety_clf.fit(X_train, ytrain_safe)
pred_safe = safety_clf.predict(X_test)

print("=== Safety model ===")
print("Accuracy:", accuracy_score(ytest_safe, pred_safe))
print(classification_report(ytest_safe, pred_safe))

# 8) Disease multi-label model
Xd_train, Xd_test, yd_train, yd_test = train_test_split(X_scaled, y_diseases.values, test_size=0.2, random_state=42)
multi_clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
multi_clf.fit(Xd_train, yd_train)
yd_pred = multi_clf.predict(Xd_test)

print("=== Disease models (per label) ===")
for i, col in enumerate(y_diseases.columns):
    print(f"--- {col} ---")
    print(classification_report(yd_test[:,i], yd_pred[:,i]))

# 9) Save models & scaler
joblib.dump(safety_clf, os.path.join(OUT_DIR, "water_safety_rf.joblib"))
joblib.dump(multi_clf, os.path.join(OUT_DIR, "water_diseases_multi_rf.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))
print("Saved models to:", OUT_DIR)

# 10) Diagnostics: feature importances & confusion matrix
importances = safety_clf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(9,4))
feat_imp.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances (Safety RF)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importances.png"))
plt.close()

cm = confusion_matrix(ytest_safe, pred_safe)
plt.figure(figsize=(4,4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix (Safety)")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_safety.png"))
plt.close()

print("Saved plots to:", OUT_DIR)
