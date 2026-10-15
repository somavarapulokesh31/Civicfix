# ============================================================
# CIVICFIX - ORDINAL ENCODING (PREPROCESSING MODULE 2)
# ============================================================
# Ordinal encoding is used when categories have a meaningful
# natural order.
#
# CivicFix status order:
# Open < In Progress < Closed
#
# Original dataset may contain:
# Open
# In Progress
# Assigned
# Pending
# Closed
# Unspecified
#
# These are first standardised into 3 classes:
# Open = 0
# In Progress = 1
# Closed = 2
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_ordinal_encoded.csv"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for ordinal encoding.")
print("Shape:", df.shape)


# ============================================================
# 4. CHECK STATUS COLUMN
# ============================================================

if "status" not in df.columns:
    raise ValueError("The 'status' column was not found in the dataset.")

print("\nOriginal status values:")
print(df["status"].value_counts(dropna=False))


# ============================================================
# 5. STANDARDISE STATUS TO 3 CLASSES
# ============================================================
# Original statuses:
#
# Open          -> Open
# Unspecified   -> Open
# Assigned      -> In Progress
# Pending       -> In Progress
# In Progress   -> In Progress
# Closed        -> Closed
#
# Final classes:
#
# Open = 0
# In Progress = 1
# Closed = 2


def standardise_status(val):

    if val == "Closed":
        return "Closed"

    elif val in ["In Progress", "Assigned", "Pending"]:
        return "In Progress"

    elif val in ["Open", "Unspecified"]:
        return "Open"

    else:
        return "Open"


df["status"] = df["status"].apply(standardise_status)


# ============================================================
# 6. DISPLAY STANDARDISED STATUS
# ============================================================

print("\nStatus after standardisation:")
print(df["status"].value_counts())


# ============================================================
# 7. ORDINAL ENCODING
# ============================================================
# Natural order:
#
# Open < In Progress < Closed
#
# Therefore:
#
# Open = 0
# In Progress = 1
# Closed = 2


print("\nApplying ordinal encoding to 'status' column...")

print("Order: Open=0 < In Progress=1 < Closed=2")


ordinal_encoder = OrdinalEncoder(
    categories=[
        ["Open", "In Progress", "Closed"]
    ]
)


# ============================================================
# 8. CREATE ORDINAL ENCODED COLUMN
# ============================================================

df["status_ordinal"] = ordinal_encoder.fit_transform(
    df[["status"]]
)


# Convert from float values such as 0.0, 1.0, 2.0
# to integer values 0, 1, 2

df["status_ordinal"] = df["status_ordinal"].astype(int)


# ============================================================
# 9. SHOW ENCODING RESULT
# ============================================================

print("\nStatus ordinal encoding result:")

print(
    df[["status", "status_ordinal"]]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. DISPLAY ENCODING MAPPING
# ============================================================

print("\nEncoding Mapping:")
print("Open        -> 0")
print("In Progress -> 1")
print("Closed      -> 2")


# ============================================================
# 11. CHECK FOR UNKNOWN VALUES
# ============================================================

print("\nUnique standardized status values:")

print(df["status"].unique())


# ============================================================
# 12. SAVE OUTPUT DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 13. FINAL INFORMATION
# ============================================================

print(
    f"\nOrdinal encoded dataset saved to:\n"
    f"  {OUTPUT_PATH}"
)

print("Shape:", df.shape)

print(
    "\nOrdinal encoding completed successfully."
)