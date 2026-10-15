# ============================================================
# CIVICFIX - REGULARIZATION: RIDGE, LASSO, ELASTIC NET
# ============================================================
# Regularization prevents overfitting by adding a penalty
# term to the cost function.
#
# RIDGE (L2):
#   J(θ) = MSE + α Σθᵢ²
#   Shrinks coefficients but rarely to zero.
#
# LASSO (L1):
#   J(θ) = MSE + α Σ|θᵢ|
#   Can shrink coefficients exactly to zero (feature selection).
#
# ELASTIC NET (L1 + L2):
#   J(θ) = MSE + α₁ Σ|θᵢ| + α₂ Σθᵢ²
#   Combines both — useful when many correlated features exist.
#
# Applied to CivicFix resolution time prediction task.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import mean_squared_error, r2_score


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

MODEL_DIR  = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD AND PREPARE DATA (same pipeline as Linear Regression)
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded.")
print("Shape:", df.shape)

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
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

for col in FEATURE_COLS:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values.astype(float)

scaler = StandardScaler()
X      = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 4. TRAIN ALL THREE REGULARIZED MODELS
# ============================================================
# alpha controls regularization strength
# Higher alpha = more regularization = simpler model

ALPHA = 1.0

models = {
    "Ridge (L2)"      : Ridge(alpha=ALPHA),
    "Lasso (L1)"      : Lasso(alpha=ALPHA, max_iter=5000),
    "Elastic Net"     : ElasticNet(alpha=ALPHA, l1_ratio=0.5, max_iter=5000),
}

results = {}

print(f"\nAlpha (regularization strength) = {ALPHA}")
print("\n" + "=" * 60)
print("  REGULARIZATION — RESULTS")
print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)

    # Count zero coefficients (Lasso feature selection)
    n_zero = np.sum(np.abs(model.coef_) < 1e-4)

    results[name] = {
        "model"  : model,
        "y_pred" : y_pred,
        "rmse"   : rmse,
        "r2"     : r2,
        "n_zero" : n_zero,
        "coef"   : model.coef_
    }

    print(f"\n  {name}")
    print(f"    RMSE        : {rmse:.4f}")
    print(f"    R²          : {r2:.4f}")
    print(f"    Zero coefficients (shrunk to 0): {n_zero}/{len(model.coef_)}")

print("=" * 60)


# ============================================================
# 5. COEFFICIENT COMPARISON CHART
# ============================================================

print("\nGenerating: Coefficient comparison chart...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

colors = {"Ridge (L2)": "#2878e8", "Lasso (L1)": "#f99a18", "Elastic Net": "#4aa84c"}

for ax, (name, res) in zip(axes, results.items()):
    ax.barh(
        FEATURE_COLS,
        res["coef"],
        color=colors[name],
        edgecolor="black"
    )
    ax.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"{name}\nRMSE={res['rmse']:.3f}  R²={res['r2']:.3f}",
        fontsize=11
    )
    ax.set_xlabel("Coefficient", fontsize=10)

fig.suptitle(
    "CivicFix — Ridge vs Lasso vs Elastic Net Coefficients",
    fontsize=13
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "regularization_coefficients.png"))
plt.close()
print("  Saved: regularization_coefficients.png")


# ============================================================
# 6. ALPHA SENSITIVITY ANALYSIS (Ridge only)
# ============================================================

print("Generating: Alpha sensitivity analysis...")

alphas   = [0.001, 0.01, 0.1, 1, 10, 100]
rmse_alphas = []
r2_alphas   = []

for a in alphas:
    m = Ridge(alpha=a)
    m.fit(X_train, y_train)
    yp = m.predict(X_test)
    rmse_alphas.append(np.sqrt(mean_squared_error(y_test, yp)))
    r2_alphas.append(r2_score(y_test, yp))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].semilogx(alphas, rmse_alphas, marker="o", color="#2878e8", linewidth=2)
axes[0].set_title("Ridge — RMSE vs Alpha", fontsize=12)
axes[0].set_xlabel("Alpha (log scale)", fontsize=10)
axes[0].set_ylabel("RMSE", fontsize=10)

axes[1].semilogx(alphas, r2_alphas, marker="s", color="#f99a18", linewidth=2)
axes[1].set_title("Ridge — R² vs Alpha", fontsize=12)
axes[1].set_xlabel("Alpha (log scale)", fontsize=10)
axes[1].set_ylabel("R²", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "regularization_alpha_sensitivity.png"))
plt.close()
print("  Saved: regularization_alpha_sensitivity.png")


# ============================================================
# 7. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([
    {
        "Model"           : name,
        "Alpha"           : ALPHA,
        "RMSE"            : round(res["rmse"], 4),
        "R2"              : round(res["r2"],   4),
        "Zero_Coeff_Count": res["n_zero"]
    }
    for name, res in results.items()
])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "regularization_results.csv"),
    index=False
)
print("Results CSV saved.")


# ============================================================
# 8. SAVE BEST MODEL (RIDGE)
# ============================================================

MODEL_PATH = os.path.join(MODEL_DIR, "ridge_regression.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(results["Ridge (L2)"]["model"], f)

print(f"\nRidge model saved to: {MODEL_PATH}")
print("Regularization comparison complete.")
