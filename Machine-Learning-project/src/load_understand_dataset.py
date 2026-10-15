"""
============================================================
CIVICFIX - LOAD AND UNDERSTAND DATASET
============================================================
This script loads the CivicFix dataset and displays
all basic information about it so that we can understand
what data we are working with before any processing.
============================================================
"""

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 2. DEFINE PROJECT PATH
# ============================================================
# We use pathlib so that this code works on ANY computer
# regardless of the folder where the project is saved.

# Project root resolved automatically

# ============================================================
# 3. DEFINE DATASET PATH
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

# ============================================================
# 4. LOAD DATASET
# ============================================================

print("Loading the CivicFix dataset...")
print(f"Dataset file : erm2-nwe9.csv")

df = pd.read_csv(DATASET_PATH, low_memory=False, encoding_errors="replace")

# Drop unnamed/empty columns that have no useful data
unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
df.drop(columns=unnamed_cols, inplace=True)

# Standardise column names: lowercase, strip spaces, replace space with _
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("Dataset loaded successfully!")

# ============================================================
# 5. DISPLAY FIRST FIVE ROWS
# ============================================================

print("\n" + "=" * 65)
print("  FIRST FIVE ROWS")
print("=" * 65)
print(df.head())

# ============================================================
# 6. DISPLAY DATASET SHAPE
# ============================================================

print("\n" + "=" * 65)
print("  DATASET SHAPE")
print("=" * 65)
print(f"  Total Rows    : {df.shape[0]:,}")
print(f"  Total Columns : {df.shape[1]}")

# ============================================================
# 7. DISPLAY COLUMN NAMES
# ============================================================

print("\n" + "=" * 65)
print("  COLUMN NAMES")
print("=" * 65)
for i, col in enumerate(df.columns, start=1):
    print(f"  {i:>2}. {col}")

# ============================================================
# 8. DISPLAY DATA TYPES
# ============================================================

print("\n" + "=" * 65)
print("  DATA TYPES")
print("=" * 65)
print(df.dtypes.to_string())

# ============================================================
# 9. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 65)
print("  MISSING VALUES")
print("=" * 65)

missing_count = df.isnull().sum()
missing_pct   = (missing_count / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count"      : missing_count,
    "Missing Percentage" : missing_pct
})

# Show only columns that actually have missing values
missing_report = missing_report[missing_report["Missing Count"] > 0]
missing_report = missing_report.sort_values("Missing Percentage", ascending=False)

if len(missing_report) == 0:
    print("  No missing values found in the dataset.")
else:
    print(missing_report.to_string())

# ============================================================
# 10. CHECK DUPLICATE VALUES
# ============================================================

print("\n" + "=" * 65)
print("  DUPLICATE ROWS")
print("=" * 65)
duplicate_count = df.duplicated().sum()
print(f"  Total Duplicate Rows : {duplicate_count}")

# ============================================================
# 11. DISPLAY STATISTICAL INFORMATION
# ============================================================

print("\n" + "=" * 65)
print("  NUMERICAL COLUMN STATISTICS")
print("=" * 65)

numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
if len(numerical_cols) > 0:
    print(df[numerical_cols].describe().to_string())
else:
    print("  No numerical columns found.")

print("\n" + "=" * 65)
print("  CATEGORICAL COLUMN STATISTICS")
print("=" * 65)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if len(categorical_cols) > 0:
    print(df[categorical_cols].describe().to_string())

# ============================================================
# 12. DATASET SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("  DATASET SUMMARY")
print("=" * 65)

print(f"\n  Total Records         : {len(df):,}")
print(f"  Total Columns         : {len(df.columns)}")
print(f"  Numerical Columns     : {len(numerical_cols)}")
print(f"  Categorical Columns   : {len(categorical_cols)}")
print(f"  Duplicate Rows        : {duplicate_count}")
print(f"  Columns with Nulls    : {len(missing_report)}")

print("\n  --- Status Column (Target Variable) ---")
if "status" in df.columns:
    print(df["status"].value_counts().to_string())

print("\n  --- Complaint Type (Top 10) ---")
if "complaint_type" in df.columns:
    print(df["complaint_type"].value_counts().head(10).to_string())

print("\n  --- Borough Distribution ---")
if "borough" in df.columns:
    print(df["borough"].value_counts().to_string())

print("\n  --- Agency Distribution ---")
if "agency" in df.columns:
    print(df["agency"].value_counts().head(10).to_string())

print("\n" + "=" * 65)
print("  DATASET UNDERSTANDING COMPLETE")
print("=" * 65)
