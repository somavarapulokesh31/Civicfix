# ============================================================
# CIVICFIX - TARGET ENCODING (PREPROCESSING MODULE 2)
# ============================================================
# Target encoding replaces each category with the mean of
# the target variable for that category.
# Example: If "NYPD" complaints are 70% Closed, the agency
# column value for NYPD becomes 0.70
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

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_target_encoded.csv"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for target encoding.")
print("Shape:", df.shape)


# ============================================================
# 4. CREATE NUMERIC TARGET FOR ENCODING
# ============================================================

# We need a numeric version of status to calculate means
STATUS_NUMERIC = {"Open": 0, "In Progress": 1, "Closed": 2}
df["status_num"] = df["status"].map(STATUS_NUMERIC)

print("\nNumeric status created: Open=0 | In Progress=1 | Closed=2")


# ============================================================
# 5. APPLY TARGET ENCODING
# ============================================================

# Columns suitable for target encoding
TARGET_ENCODE_COLS = [
    c for c in ["agency", "complaint_type", "borough", "location_type"]
    if c in df.columns
]

print(f"\nApplying target encoding to: {TARGET_ENCODE_COLS}")

for col in TARGET_ENCODE_COLS:
    # Calculate mean status_num for each category
    target_means = df.groupby(col)["status_num"].mean()

    # Replace the original category with its mean target value
    df[col + "_target_enc"] = df[col].map(target_means)

    print(f"  {col}: target encoded "
          f"(range {target_means.min():.3f} to {target_means.max():.3f})")


# ============================================================
# 6. DROP TEMPORARY COLUMN
# ============================================================

df.drop(columns=["status_num"], inplace=True)


# ============================================================
# 7. SAVE OUTPUT DATASET
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nTarget encoded dataset saved to:\n  {OUTPUT_PATH}")
print("Shape:", df.shape)
