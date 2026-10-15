# ============================================================
# CIVICFIX - CORRELATION MATRIX, HEATMAP AND BOXPLOTS
# ============================================================
# This script analyses numerical relationships in the dataset,
# generates a correlation heatmap, and draws boxplots
# to detect outliers in numerical columns.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\correlation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 4. SELECT NUMERICAL COLUMNS
# ============================================================

# Correlation analysis only makes sense on numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

print("\nNumerical columns available for correlation:")
print(numerical_columns)

# Use the most meaningful numerical columns
useful_num_cols = [
    c for c in [
        "latitude", "longitude",
        "council_district", "x_coordinate_state_plane",
        "y_coordinate_state_plane", "incident_zip"
    ]
    if c in numerical_columns
]

print(f"\nUsing {len(useful_num_cols)} numerical columns for analysis.")


# ============================================================
# 5. CALCULATE CORRELATION MATRIX
# ============================================================

print("\n" + "="*60)
print("CORRELATION MATRIX")
print("="*60)

if len(useful_num_cols) >= 2:
    correlation_matrix = df[useful_num_cols].corr()
    print(correlation_matrix)
else:
    print("Not enough numerical columns for correlation.")


# ============================================================
# 6. GENERATE CORRELATION HEATMAP
# ============================================================

print("\nGenerating: Correlation Heatmap...")

if len(useful_num_cols) >= 2:
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True
    )
    plt.title("Correlation Heatmap - CivicFix Numerical Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
    plt.close()
    print("  Saved: correlation_heatmap.png")


# ============================================================
# 7. GENERATE BOXPLOTS FOR NUMERICAL COLUMNS
# ============================================================

print("\nGenerating: Boxplots for numerical columns...")

for col in useful_num_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(
        x=df[col].dropna(),
        color="lightblue"
    )
    plt.title(f"Boxplot - {col}", fontsize=13)
    plt.xlabel(col, fontsize=11)
    plt.tight_layout()
    filename = f"boxplot_{col}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# 8. OUTLIER ANALYSIS USING IQR METHOD
# ============================================================

print("\n" + "="*60)
print("OUTLIER ANALYSIS (IQR Method)")
print("="*60)

for col in useful_num_cols:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    print(f"  {col:<35} | Outliers: {outlier_count:>5} "
          f"| Range: [{lower_bound:.2f}, {upper_bound:.2f}]")


# ============================================================
# 9. STATUS vs NUMERICAL FEATURES (grouped boxplot)
# ============================================================

print("\nGenerating: Status vs Latitude boxplot...")

if "latitude" in df.columns and "status" in df.columns:
    # Standardise status to 3 classes before plotting
    def std_status(v):
        if v == "Closed": return "Closed"
        if v in ["In Progress","Assigned","Pending"]: return "In Progress"
        return "Open"
    df["status"] = df["status"].apply(std_status)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        x="status",
        y="latitude",
        hue="status",
        data=df,
        palette={
            "Closed"      : "#4aa84c",
            "In Progress" : "#2878e8",
            "Open"        : "#f99a18"
        },
        legend=False
    )
    plt.title("Latitude Distribution by Complaint Status", fontsize=14)
    plt.xlabel("Status", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "status_vs_latitude.png"))
    plt.close()
    print("  Saved: status_vs_latitude.png")


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "="*60)
print("CORRELATION ANALYSIS COMPLETE")
print("="*60)
print(f"Output folder: {OUTPUT_DIR}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print(f"  - {f}")
