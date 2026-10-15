# ============================================================
# CIVICFIX - SCALING AND NORMALIZATION (PREPROCESSING MODULE 2)
# ============================================================
# Scales numerical features so they are all on a similar range.
# MinMax scaling: brings values to [0, 1]
# Standard scaling: mean=0, std=1 — best for most ML models
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_scaled.csv"

MODEL_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for scaling.")
print("Shape:", df.shape)


# ============================================================
# 4. SELECT NUMERICAL COLUMNS TO SCALE
# ============================================================

# We exclude the target column and ID columns from scaling
EXCLUDE_COLS = [
    "status", "status_encoded", "unique_key",
    "created_date", "closed_date"
]

# Get all numerical columns that are not excluded
numerical_cols = [
    c for c in df.select_dtypes(include=["int64", "float64"]).columns
    if c not in EXCLUDE_COLS
]

print(f"\nNumerical columns to scale ({len(numerical_cols)}):")
print(numerical_cols)


# ============================================================
# 5. APPLY STANDARD SCALER
# ============================================================
# StandardScaler is the most appropriate for logistic regression
# and most machine learning models

standard_scaler = StandardScaler()

df_scaled           = df.copy()
df_scaled[numerical_cols] = standard_scaler.fit_transform(df[numerical_cols])

print(f"\nStandard Scaler applied to {len(numerical_cols)} columns.")
print("  Mean = 0, Standard Deviation = 1")


# ============================================================
# 6. SAVE SCALER FOR USE IN PREDICTION
# ============================================================

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

with open(SCALER_PATH, "wb") as f:
    pickle.dump(standard_scaler, f)

print(f"\nScaler saved to:\n  {SCALER_PATH}")


# ============================================================
# 7. SHOW BEFORE AND AFTER SAMPLE
# ============================================================

print("\nSample values BEFORE scaling (latitude):")
if "latitude" in df.columns:
    print(df["latitude"].head(5).values)

print("\nSample values AFTER scaling (latitude):")
if "latitude" in df_scaled.columns:
    print(df_scaled["latitude"].head(5).values)


# ============================================================
# 8. SAVE SCALED DATASET
# ============================================================

df_scaled.to_csv(OUTPUT_PATH, index=False)

print(f"\nScaled dataset saved to:\n  {OUTPUT_PATH}")
print("Shape:", df_scaled.shape)
