# ============================================================
# CIVICFIX - LINEAR REGRESSION WITH METRICS (MODULE 2)
# ============================================================
# Linear Regression is a supervised learning algorithm that
# predicts a continuous output by fitting a straight line
# through the data points.
#
# For CivicFix, we use it to predict:
#   → resolution_time_days (how many days to close a complaint)
#
# Evaluation metrics used:
#   MSE   - Mean Squared Error
#   RMSE  - Root Mean Squared Error
#   MAE   - Mean Absolute Error
#   R²    - Coefficient of Determination
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
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (mean_squared_error,
                                     mean_absolute_error,
                                     r2_score)
from sklearn.preprocessing   import LabelEncoder, StandardScaler


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

print("Dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 4. ENGINEER TARGET: RESOLUTION TIME (DAYS)
# ============================================================
# We compute how many days it took to resolve (close) a
# complaint using created_date and closed_date columns.
# Only rows where closed_date is available are used.

print("\nEngineering target: resolution_time_days...")

if "created_date" in df.columns and "closed_date" in df.columns:

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"]  = pd.to_datetime(df["closed_date"],  errors="coerce")

    df["resolution_time_days"] = (
        df["closed_date"] - df["created_date"]
    ).dt.days

    # Keep only rows with valid (non-negative) resolution times
    df = df[df["resolution_time_days"] > 0].copy()

    print(f"  Rows with valid resolution time: {len(df):,}")
    print(f"  Mean resolution time: {df['resolution_time_days'].mean():.1f} days")
    print(f"  Max  resolution time: {df['resolution_time_days'].max():.0f} days")

else:
    # Fallback: predict numeric council_district from coordinates
    print("  Date columns not found — using council_district as target.")
    df.dropna(subset=["council_district", "latitude", "longitude"], inplace=True)
    df["resolution_time_days"] = df["council_district"].astype(float)


# ============================================================
# 5. SELECT FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
    c for c in [
        "complaint_type", "agency", "borough",
        "location_type", "open_data_channel_type",
        "council_district", "latitude", "longitude"
    ]
    if c in df.columns
]

TARGET_COL = "resolution_time_days"

df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

# Force all feature columns to correct types
for col in FEATURE_COLS:
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

print(f"\nFeatures : {FEATURE_COLS}")
print(f"Target   : {TARGET_COL}")
print(f"Rows used: {len(df):,}")


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

label_encoders = {}

# ============================================================
# 7. SCALE FEATURES
# ============================================================

X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values.astype(float)

scaler = StandardScaler()
X      = scaler.fit_transform(X)


# ============================================================
# 8. SPLIT INTO TRAIN AND TEST SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 9. TRAIN LINEAR REGRESSION MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("Training complete!")


# ============================================================
# 10. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. CALCULATE EVALUATION METRICS
# ============================================================

mse   = mean_squared_error(y_test, y_pred)
rmse  = np.sqrt(mse)
mae   = mean_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("  LINEAR REGRESSION — RESULTS")
print("=" * 60)
print(f"  MSE   (Mean Squared Error)       : {mse:.4f}")
print(f"  RMSE  (Root Mean Squared Error)  : {rmse:.4f}")
print(f"  MAE   (Mean Absolute Error)      : {mae:.4f}")
print(f"  R²    (Coefficient of Det.)      : {r2:.4f}")
print("=" * 60)


# ============================================================
# 12. ACTUAL vs PREDICTED SCATTER PLOT
# ============================================================

print("\nGenerating: Actual vs Predicted scatter plot...")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color="steelblue", edgecolors="none", s=10)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", linewidth=2, label="Perfect Prediction"
)
plt.title(
    f"Linear Regression — Actual vs Predicted\nR² = {r2:.4f}",
    fontsize=13
)
plt.xlabel("Actual Values",    fontsize=11)
plt.ylabel("Predicted Values", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lr_actual_vs_predicted.png"))
plt.close()
print("  Saved: lr_actual_vs_predicted.png")


# ============================================================
# 13. RESIDUALS PLOT
# ============================================================

print("Generating: Residuals plot...")

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.3, color="darkorange", s=10)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.title("Linear Regression — Residuals Plot", fontsize=13)
plt.xlabel("Predicted Values", fontsize=11)
plt.ylabel("Residuals",        fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lr_residuals.png"))
plt.close()
print("  Saved: lr_residuals.png")


# ============================================================
# 14. FEATURE COEFFICIENTS CHART
# ============================================================

print("Generating: Feature Coefficients chart...")

coeff_df = pd.DataFrame({
    "Feature"     : FEATURE_COLS,
    "Coefficient" : model.coef_
}).sort_values("Coefficient", ascending=False)

plt.figure(figsize=(9, 5))
plt.barh(
    coeff_df["Feature"],
    coeff_df["Coefficient"],
    color=["#2878e8" if c >= 0 else "#e84848"
           for c in coeff_df["Coefficient"]]
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Linear Regression — Feature Coefficients", fontsize=13)
plt.xlabel("Coefficient Value", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lr_coefficients.png"))
plt.close()
print("  Saved: lr_coefficients.png")


# ============================================================
# 15. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([{
    "Model" : "Linear Regression",
    "MSE"   : round(mse,  4),
    "RMSE"  : round(rmse, 4),
    "MAE"   : round(mae,  4),
    "R2"    : round(r2,   4)
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "linear_regression_results.csv"),
    index=False
)
print("Results CSV saved.")


# ============================================================
# 16. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(MODEL_DIR, "linear_regression.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Linear Regression R²: {r2:.4f}")
