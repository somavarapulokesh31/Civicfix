# ============================================================
# CIVICFIX - IDENTIFY AND HANDLE MISSING VALUES
# ============================================================
# This script analyzes missing values in the CivicFix dataset,
# applies appropriate strategies to handle them, and saves
# the cleaned dataset for further processing.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_DATASET = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_missing_values.csv"

OUTPUT_PLOT = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\EDA_analysis\missing_values_chart.png"

# Create output directory if it does not exist
os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded successfully!")
print("Shape before cleaning:", df.shape)


# ============================================================
# 4. CHECK MISSING VALUES BEFORE CLEANING
# ============================================================

print("\n" + "="*60)
print("MISSING VALUES - BEFORE CLEANING")
print("="*60)

missing_count   = df.isnull().sum()
missing_percent = (missing_count / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing Count"   : missing_count,
    "Missing Percent" : missing_percent
})

# Keep only columns with at least 1 missing value
missing_report = missing_report[missing_report["Missing Count"] > 0]
missing_report = missing_report.sort_values("Missing Percent", ascending=False)

print(missing_report)
print(f"\nTotal columns with missing values: {len(missing_report)}")


# ============================================================
# 5. VISUALIZE MISSING VALUES
# ============================================================

if len(missing_report) > 0:
    plt.figure(figsize=(12, 6))
    plt.bar(
        missing_report.index,
        missing_report["Missing Percent"],
        color="tomato",
        edgecolor="black"
    )
    plt.title("Missing Value Percentage per Column", fontsize=14)
    plt.xlabel("Column Name", fontsize=12)
    plt.ylabel("Missing Percentage (%)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.close()
    print(f"\nMissing values chart saved to: {OUTPUT_PLOT}")


# ============================================================
# 6. DROP COLUMNS WITH MORE THAN 60% MISSING VALUES
# ============================================================

print("\n" + "="*60)
print("DROPPING COLUMNS WITH MORE THAN 60% MISSING VALUES")
print("="*60)

# Columns where more than 60% of data is missing are not useful
threshold = 60.0

columns_to_drop = missing_report[missing_report["Missing Percent"] > threshold].index.tolist()

print("Columns being dropped:")
for col in columns_to_drop:
    pct = missing_report.loc[col, "Missing Percent"]
    print(f"  - {col}  ({pct}% missing)")

df.drop(columns=columns_to_drop, inplace=True)
print(f"\nShape after dropping high-missing columns: {df.shape}")


# ============================================================
# 7. DROP UNNAMED / IRRELEVANT COLUMNS
# ============================================================

print("\n" + "="*60)
print("DROPPING UNNAMED COLUMNS")
print("="*60)

unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
print("Unnamed columns found:", unnamed_cols)

df.drop(columns=unnamed_cols, inplace=True)
print(f"Shape after dropping unnamed columns: {df.shape}")


# ============================================================
# 8. STANDARDISE STATUS COLUMN TO 3 CLASSES
# ============================================================
# The original dataset has: Closed, In Progress, Open,
# Assigned, Pending, Unspecified. 
# We map all to 3 meaningful classes only.
# ============================================================

print("\n" + "="*60)
print("STANDARDISING STATUS TO 3 CLASSES")
print("="*60)

print("Status before standardisation:")
print(df["status"].value_counts())

def standardise_status(value):
    if value == "Closed":
        return "Closed"
    elif value == "In Progress":
        return "In Progress"
    else:
        # Open, Assigned, Pending, Unspecified → all become Open
        return "Open"

df["status"] = df["status"].apply(standardise_status)

print("\nStatus after standardisation:")
print(df["status"].value_counts())


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================================

print("\n" + "="*60)
print("IDENTIFYING COLUMN TYPES")
print("="*60)

numerical_columns   = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

print(f"Numerical Columns  ({len(numerical_columns)}) : {numerical_columns}")
print(f"Categorical Columns({len(categorical_columns)}) : {categorical_columns}")


# ============================================================
# 10. FILL MISSING VALUES IN NUMERICAL COLUMNS WITH MEAN
# ============================================================

print("\n" + "="*60)
print("FILLING MISSING VALUES IN NUMERICAL COLUMNS (MEAN)")
print("="*60)

for col in numerical_columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"  {col}: filled {missing} missing values with mean = {mean_value:.2f}")


# ============================================================
# 11. FILL MISSING VALUES IN CATEGORICAL COLUMNS WITH MODE
# ============================================================

print("\n" + "="*60)
print("FILLING MISSING VALUES IN CATEGORICAL COLUMNS (MODE)")
print("="*60)

for col in categorical_columns:
    missing = df[col].isnull().sum()
    if missing > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"  {col}: filled {missing} missing values with mode = '{mode_value}'")


# ============================================================
# 12. CHECK MISSING VALUES AFTER CLEANING
# ============================================================

print("\n" + "="*60)
print("MISSING VALUES - AFTER CLEANING")
print("="*60)

remaining_missing = df.isnull().sum().sum()
print(f"Total remaining missing values: {remaining_missing}")

if remaining_missing == 0:
    print("All missing values have been handled successfully.")


# ============================================================
# 13. REMOVE DUPLICATE ROWS
# ============================================================

print("\n" + "="*60)
print("REMOVING DUPLICATE ROWS")
print("="*60)

before_dedup = len(df)
df.drop_duplicates(inplace=True)
after_dedup  = len(df)

print(f"Rows before removing duplicates : {before_dedup:,}")
print(f"Rows after  removing duplicates : {after_dedup:,}")
print(f"Duplicate rows removed          : {before_dedup - after_dedup}")


# ============================================================
# 14. FINAL SHAPE AFTER CLEANING
# ============================================================

print("\n" + "="*60)
print("FINAL DATASET SHAPE AFTER CLEANING")
print("="*60)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 15. SAVE CLEANED DATASET
# ============================================================

df.to_csv(OUTPUT_DATASET, index=False)

print("\n" + "="*60)
print("CLEANED DATASET SAVED")
print("="*60)
print(f"Saved to: {OUTPUT_DATASET}")
