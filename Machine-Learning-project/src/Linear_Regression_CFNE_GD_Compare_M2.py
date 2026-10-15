# ============================================================
# CIVICFIX - LINEAR REGRESSION: CLOSED FORM vs GRADIENT DESCENT
# ============================================================
# This script compares two methods of training Linear Regression:
#
# 1. CLOSED FORM (Normal Equation):
#    Solves analytically: θ = (XᵀX)⁻¹ Xᵀy
#    - Exact solution in one step
#    - Slow for very large datasets
#
# 2. GRADIENT DESCENT:
#    Iteratively adjusts θ to minimize MSE loss
#    - Cost  J(θ) = (1/2m) Σ(ŷᵢ - yᵢ)²
#    - Update: θ = θ - α * ∇J(θ)
#    - α = learning rate
#
# Both methods are applied to the CivicFix dataset and
# their results are compared side-by-side.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import mean_squared_error, r2_score


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 4. PREPARE DATA (SAME AS LINEAR_REGRESSION_WITH_METRICS)
# ============================================================

if "created_date" in df.columns and "closed_date" in df.columns:
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"]  = pd.to_datetime(df["closed_date"],  errors="coerce")
    df["resolution_time_days"] = (df["closed_date"] - df["created_date"]).dt.days
    df = df[df["resolution_time_days"] > 0].copy()
    TARGET_COL = "resolution_time_days"
else:
    df.dropna(subset=["council_district", "latitude", "longitude"], inplace=True)
    df["resolution_time_days"] = df["council_district"].astype(float)
    TARGET_COL = "resolution_time_days"

FEATURE_COLS = [
    c for c in [
        "complaint_type", "agency", "borough",
        "location_type", "open_data_channel_type",
        "council_district", "latitude", "longitude"
    ]
    if c in df.columns
]

df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

for col in FEATURE_COLS:
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)

X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values.astype(float)

scaler = StandardScaler()
X      = scaler.fit_transform(X)

# Add bias term (column of ones for intercept)
X_b = np.c_[np.ones((X.shape[0], 1)), X]

X_train, X_test, y_train, y_test = train_test_split(
    X_b, y, test_size=0.20, random_state=42
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 5. METHOD 1: CLOSED FORM (NORMAL EQUATION)
# ============================================================
# θ = (XᵀX)⁻¹ Xᵀy
# Solved directly — no iteration needed

print("\n" + "=" * 60)
print("  METHOD 1: CLOSED FORM (Normal Equation)")
print("=" * 60)

# Use pseudo-inverse to handle singular matrices
theta_cf = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train

y_pred_cf  = X_test @ theta_cf
mse_cf     = mean_squared_error(y_test, y_pred_cf)
rmse_cf    = np.sqrt(mse_cf)
r2_cf      = r2_score(y_test, y_pred_cf)

print(f"  RMSE  : {rmse_cf:.4f}")
print(f"  R²    : {r2_cf:.4f}")
print(f"  Theta (first 5): {theta_cf[:5].round(4)}")


# ============================================================
# 6. METHOD 2: GRADIENT DESCENT
# ============================================================
# Iteratively minimizes cost function J(θ):
#
#   J(θ) = (1/2m) Σ (X·θ - y)²
#
# Update rule:
#   θ := θ - (α/m) · Xᵀ · (X·θ - y)

print("\n" + "=" * 60)
print("  METHOD 2: GRADIENT DESCENT")
print("=" * 60)

LEARNING_RATE = 0.01
N_ITERATIONS  = 500
m             = len(X_train)

# Initialise theta to zeros
theta_gd      = np.zeros(X_train.shape[1])
cost_history  = []

print(f"  Learning Rate : {LEARNING_RATE}")
print(f"  Iterations    : {N_ITERATIONS}")
print(f"  Training rows : {m:,}")

for iteration in range(N_ITERATIONS):

    # Predicted values
    predictions = X_train @ theta_gd

    # Error = predictions - actual
    errors = predictions - y_train

    # Cost function (MSE)
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    cost_history.append(cost)

    # Gradient = (1/m) Xᵀ · errors
    gradient = (1 / m) * (X_train.T @ errors)

    # Update theta
    theta_gd = theta_gd - (LEARNING_RATE * gradient)

    if iteration % 100 == 0:
        print(f"  Iteration {iteration:>4} | Cost: {cost:.4f}")

y_pred_gd = X_test @ theta_gd
mse_gd    = mean_squared_error(y_test, y_pred_gd)
rmse_gd   = np.sqrt(mse_gd)
r2_gd     = r2_score(y_test, y_pred_gd)

print(f"\n  Final Cost : {cost_history[-1]:.4f}")
print(f"  RMSE       : {rmse_gd:.4f}")
print(f"  R²         : {r2_gd:.4f}")


# ============================================================
# 7. COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("  COMPARISON: CLOSED FORM vs GRADIENT DESCENT")
print("=" * 60)
print(f"  {'Metric':<10} | {'Closed Form':>15} | {'Gradient Descent':>16}")
print(f"  {'-'*10}-+-{'-'*15}-+-{'-'*16}")
print(f"  {'RMSE':<10} | {rmse_cf:>15.4f} | {rmse_gd:>16.4f}")
print(f"  {'R²':<10} | {r2_cf:>15.4f} | {r2_gd:>16.4f}")
print("=" * 60)


# ============================================================
# 8. GRADIENT DESCENT COST CURVE
# ============================================================

print("\nGenerating: Gradient Descent cost curve...")

plt.figure(figsize=(9, 5))
plt.plot(
    range(N_ITERATIONS),
    cost_history,
    color="#2878e8",
    linewidth=2
)
plt.title(
    f"Gradient Descent — Cost (MSE) over {N_ITERATIONS} Iterations\n"
    f"Learning Rate = {LEARNING_RATE}",
    fontsize=13
)
plt.xlabel("Iteration", fontsize=11)
plt.ylabel("Cost (MSE)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gd_cost_curve.png"))
plt.close()
print("  Saved: gd_cost_curve.png")


# ============================================================
# 9. SIDE-BY-SIDE ACTUAL vs PREDICTED
# ============================================================

print("Generating: Side-by-side actual vs predicted comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_pred, title in zip(
    axes,
    [y_pred_cf, y_pred_gd],
    ["Closed Form (Normal Equation)", "Gradient Descent"]
):
    ax.scatter(y_test, y_pred, alpha=0.3, s=8, color="steelblue")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", linewidth=1.5, label="Perfect"
    )
    r2_val = r2_score(y_test, y_pred)
    ax.set_title(f"{title}\nR² = {r2_val:.4f}", fontsize=12)
    ax.set_xlabel("Actual",    fontsize=10)
    ax.set_ylabel("Predicted", fontsize=10)
    ax.legend()

plt.suptitle(
    "CivicFix — Closed Form vs Gradient Descent",
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cf_vs_gd_comparison.png"))
plt.close()
print("  Saved: cf_vs_gd_comparison.png")


# ============================================================
# 10. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([
    {"Method": "Closed Form (Normal Equation)",
     "RMSE": round(rmse_cf, 4), "R2": round(r2_cf, 4)},
    {"Method": "Gradient Descent",
     "RMSE": round(rmse_gd, 4), "R2": round(r2_gd, 4)},
])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "cf_vs_gd_results.csv"),
    index=False
)
print("Results CSV saved.")

print("\nClosed Form vs Gradient Descent comparison complete.")
