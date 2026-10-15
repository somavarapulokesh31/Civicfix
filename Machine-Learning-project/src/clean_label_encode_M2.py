# ============================================================
# CIVICFIX - LABEL ENCODING (PREPROCESSING MODULE 2)
# ============================================================
# Converts categorical text columns into numerical labels.
# Example: "NYPD"->0, "HPD"->1, "DSNY"->2
# Also encodes the target: Open=0, In Progress=1, Closed=2
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_label_encoded.csv"

MODEL_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for label encoding.")
print("Shape:", df.shape)


# ============================================================
# 4. DEFINE COLUMNS TO LABEL ENCODE
# ============================================================

# These are high-cardinality categorical columns that
# are best handled with label encoding for tree-based models
LABEL_ENCODE_COLS = [
    c for c in ["agency", "complaint_type", "descriptor",
                "location_type", "city", "address_type"]
    if c in df.columns
]

print(f"\nColumns to label encode: {LABEL_ENCODE_COLS}")


# ============================================================
# 5. APPLY LABEL ENCODING TO EACH COLUMN
# ============================================================

# Store encoders so we can use them for prediction later
label_encoders = {}

for col in LABEL_ENCODE_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  Label encoded: {col}  ({len(le.classes_)} unique classes)")


# ============================================================
# 6. ENCODE TARGET COLUMN (STATUS)
# ============================================================

# Map status to numbers in a meaningful order:
# Open=0, In Progress=1, Closed=2
STATUS_MAPPING = {
    "Open"        : 0,
    "In Progress" : 1,
    "Closed"      : 2
}

df["status_encoded"] = df["status"].map(STATUS_MAPPING)

print(f"\nTarget column 'status' encoded:")
print("  Open=0  |  In Progress=1  |  Closed=2")
print(df["status_encoded"].value_counts())


# ============================================================
# 7. SAVE LABEL ENCODERS FOR USE IN PREDICTION
# ============================================================

ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoders, f)

print(f"\nLabel encoders saved to:\n  {ENCODER_PATH}")


# ============================================================
# 8. SAVE ENCODED DATASET
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nLabel-encoded dataset saved to:\n  {OUTPUT_PATH}")
print("Shape:", df.shape)
