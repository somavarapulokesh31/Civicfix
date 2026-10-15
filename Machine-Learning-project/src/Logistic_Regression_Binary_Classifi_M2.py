# ============================================================
# CIVICFIX - LOGISTIC REGRESSION (BINARY CLASSIFICATION)
# ============================================================
# Binary Logistic Regression predicts ONE of TWO classes.
#
# For CivicFix, we frame the problem as:
#   → Is a complaint RESOLVED (Closed) or NOT RESOLVED?
#
# Class mapping:
#   Closed      → 1  (Resolved)
#   Open        → 0  (Not Resolved)
#   In Progress → 0  (Not Resolved)
#
# The sigmoid function maps any value to [0, 1]:
#   σ(z) = 1 / (1 + e⁻ᶻ)
#
# If σ(z) ≥ 0.5 → Predict Closed (1)
# If σ(z) <  0.5 → Predict Not Resolved (0)
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, roc_curve, auc,
                                     precision_score, recall_score, f1_score)


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

MODEL_DIR  = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for Binary Logistic Regression.")
print("Shape:", df.shape)


# ============================================================
# 4. CREATE BINARY TARGET
# ============================================================
# Closed = 1 (Resolved)
# Open / In Progress = 0 (Not Resolved)

print("\nCreating binary target: Closed=1 | Not Resolved=0")

df["binary_status"] = (df["status"] == "Closed").astype(int)

print(df["binary_status"].value_counts())
print(f"  0 = Not Resolved : {(df['binary_status']==0).sum():,}")
print(f"  1 = Closed        : {(df['binary_status']==1).sum():,}")


# ============================================================
# 5. SELECT FEATURES
# ============================================================

FEATURE_COLS = [
    c for c in [
        "complaint_type", "agency", "borough",
        "location_type", "open_data_channel_type",
        "council_district", "latitude", "longitude"
    ]
    if c in df.columns
]

TARGET_COL = "binary_status"

df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

print(f"\nFeature columns: {FEATURE_COLS}")
print(f"Rows used      : {len(df):,}")


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

for col in FEATURE_COLS:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))


# ============================================================
# 7. SCALE FEATURES
# ============================================================

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

scaler = StandardScaler()
X      = scaler.fit_transform(X)


# ============================================================
# 8. SPLIT INTO TRAIN AND TEST SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 9. TRAIN BINARY LOGISTIC REGRESSION
# ============================================================
# binary classification: solver='lbfgs', no multi_class needed

print("\nTraining Binary Logistic Regression...")

model = LogisticRegression(
    solver       = "lbfgs",
    max_iter     = 1000,
    random_state = 42
)

model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]   # prob of class=1


# ============================================================
# 11. CALCULATE EVALUATION METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  BINARY LOGISTIC REGRESSION — RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Not Resolved (0)", "Closed (1)"]
))
print("Confusion Matrix:")
print(conf_mat)


# ============================================================
# 12. ROC CURVE
# ============================================================

print("Generating: ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="#2878e8", linewidth=2,
         label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
plt.title("Binary Logistic Regression — ROC Curve\n"
          "Predicting: Closed vs Not Resolved",
          fontsize=13)
plt.xlabel("False Positive Rate", fontsize=11)
plt.ylabel("True Positive Rate",  fontsize=11)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "binary_lr_roc_curve.png"))
plt.close()
print("  Saved: binary_lr_roc_curve.png")


# ============================================================
# 13. CONFUSION MATRIX CHART
# ============================================================

print("Generating: Confusion Matrix chart...")

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_mat,
    annot        = True,
    fmt          = "d",
    cmap         = "Blues",
    xticklabels  = ["Not Resolved", "Closed"],
    yticklabels  = ["Not Resolved", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"Binary Logistic Regression — Confusion Matrix\n"
    f"Accuracy: {accuracy*100:.2f}%",
    fontsize=12
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "binary_lr_confusion_matrix.png"))
plt.close()
print("  Saved: binary_lr_confusion_matrix.png")


# ============================================================
# 14. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : "Binary Logistic Regression",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2),
    "AUC_ROC"  : round(roc_auc,          4),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "binary_lr_results.csv"),
    index=False
)
print("Results CSV saved.")


# ============================================================
# 15. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(MODEL_DIR, "binary_logistic_regression.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Binary Logistic Regression Accuracy: {accuracy * 100:.2f}%")
print(f"ROC AUC Score                       : {roc_auc:.4f}")
