# ============================================================
# CIVICFIX - ONE-HOT ENCODING (PREPROCESSING MODULE 2)
# ============================================================
# Converts low-cardinality categorical columns into
# binary (0/1) columns - one column per category value.
# Example: borough_BROOKLYN=1, borough_QUEENS=0 ...
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import os


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_one_hot_encoded.csv"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for one-hot encoding.")
print("Shape before OHE:", df.shape)


# ============================================================
# 4. DEFINE COLUMNS FOR ONE-HOT ENCODING
# ============================================================

# One-hot encoding is best for low-cardinality columns
# (fewer unique values = fewer new columns created)
OHE_COLUMNS = [
    c for c in ["borough", "open_data_channel_type", "address_type"]
    if c in df.columns
]

print(f"\nColumns to one-hot encode: {OHE_COLUMNS}")

# Show unique values for each column
for col in OHE_COLUMNS:
    print(f"  {col}: {df[col].nunique()} unique values → {df[col].unique().tolist()}")


# ============================================================
# 5. APPLY ONE-HOT ENCODING
# ============================================================

# pd.get_dummies creates a new binary column for each category
# drop_first=False keeps all categories for clarity
df_encoded = pd.get_dummies(
    df,
    columns=OHE_COLUMNS,
    drop_first=False
)

print(f"\nShape after OHE : {df_encoded.shape}")
print(f"New columns added: {df_encoded.shape[1] - df.shape[1]}")

# Show the new columns created
new_cols = [c for c in df_encoded.columns if c not in df.columns]
print("\nNew one-hot columns:")
for c in new_cols:
    print(f"  - {c}")


# ============================================================
# 6. SAVE ONE-HOT ENCODED DATASET
# ============================================================

df_encoded.to_csv(OUTPUT_PATH, index=False)

print(f"\nOne-hot encoded dataset saved to:\n  {OUTPUT_PATH}")
