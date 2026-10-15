# ============================================================
# CIVICFIX - FINAL PREPROCESSING PIPELINE
# ============================================================
# This script creates the final ML-ready dataset by:
# 1. Loading the clean dataset
# 2. Selecting only the most useful features
# 3. Encoding categorical columns
# 4. Scaling numerical columns
# 5. Separating features and target
# 6. Saving the final preprocessed dataset
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\final_preprocess_M2.csv"

MODEL_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Clean dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 4. SELECT MOST USEFUL FEATURE COLUMNS
# ============================================================
# Based on domain knowledge, these columns are the most
# relevant for predicting complaint status:
# - complaint_type : what kind of issue was reported
# - agency         : which department is handling it
# - borough        : which area of NYC
# - location_type  : where exactly (street, building etc)
# - open_data_channel_type : how was it reported (phone/online)
# - council_district : political district
# - latitude, longitude : geographic coordinates

FEATURE_COLUMNS = [
    c for c in [
        "complaint_type", "agency", "borough",
        "location_type", "open_data_channel_type",
        "council_district", "latitude", "longitude"
    ]
    if c in df.columns
]

TARGET_COLUMN = "status"

print(f"\nSelected {len(FEATURE_COLUMNS)} feature columns:")
for c in FEATURE_COLUMNS:
    print(f"  - {c}")
print(f"\nTarget column: {TARGET_COLUMN}")


# ============================================================
# 5. CREATE WORKING DATAFRAME WITH SELECTED COLUMNS
# ============================================================

df_final = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

print(f"\nWorking dataframe shape: {df_final.shape}")


# ============================================================
# 6. ENCODE CATEGORICAL FEATURE COLUMNS
# ============================================================

categorical_features = [
    c for c in FEATURE_COLUMNS
    if df_final[c].dtype == "object"
]

print(f"\nEncoding {len(categorical_features)} categorical feature columns...")

label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    df_final[col] = le.fit_transform(df_final[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded: {col}  ({len(le.classes_)} classes)")


# ============================================================
# 7. ENCODE TARGET COLUMN (STATUS)
# ============================================================

STATUS_MAP = {"Open": 0, "In Progress": 1, "Closed": 2}
df_final["status"] = df_final["status"].map(STATUS_MAP)
df_final.dropna(subset=["status"], inplace=True)
df_final["status"] = df_final["status"].astype(int)

print(f"\nTarget encoded: Open=0 | In Progress=1 | Closed=2")
print(df_final["status"].value_counts())


# ============================================================
# 8. SCALE NUMERICAL FEATURE COLUMNS
# ============================================================

numerical_features = [
    c for c in FEATURE_COLUMNS
    if df_final[c].dtype in ["int64", "float64"]
]

print(f"\nScaling {len(numerical_features)} numerical feature columns...")

scaler = StandardScaler()

if numerical_features:
    df_final[numerical_features] = scaler.fit_transform(df_final[numerical_features])
    print(f"  Scaled: {numerical_features}")


# ============================================================
# 9. SAVE PREPROCESSING OBJECTS FOR FLASK PREDICTION
# ============================================================

preprocessing_pipeline = {
    "feature_columns"     : FEATURE_COLUMNS,
    "categorical_features": categorical_features,
    "numerical_features"  : numerical_features,
    "label_encoders"      : label_encoders,
    "scaler"              : scaler,
    "status_map"          : STATUS_MAP,
    "status_map_inverse"  : {0: "Open", 1: "In Progress", 2: "Closed"}
}

PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessing_pipeline.pkl")

with open(PIPELINE_PATH, "wb") as f:
    pickle.dump(preprocessing_pipeline, f)

print(f"\nPreprocessing pipeline saved to:\n  {PIPELINE_PATH}")


# ============================================================
# 10. DISPLAY FINAL DATASET INFO
# ============================================================

print("\n" + "="*60)
print("FINAL PREPROCESSED DATASET")
print("="*60)
print(f"Rows    : {df_final.shape[0]:,}")
print(f"Columns : {df_final.shape[1]}")
print(f"Features: {FEATURE_COLUMNS}")
print(f"Target  : status (0=Open, 1=In Progress, 2=Closed)")
print("\nFirst 5 rows of final dataset:")
print(df_final.head())


# ============================================================
# 11. SAVE FINAL PREPROCESSED DATASET
# ============================================================

df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal preprocessed dataset saved to:\n  {OUTPUT_PATH}")
