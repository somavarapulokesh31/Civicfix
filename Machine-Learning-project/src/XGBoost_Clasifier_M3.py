# ============================================================
# CIVICFIX - XGBOOST CLASSIFIER (MODULE 3)
# ============================================================
# XGBoost (Extreme Gradient Boosting) is a highly optimized
# gradient boosting framework with regularization.
#
# Key improvements over standard Gradient Boosting:
#   - Regularization (L1 + L2) to prevent overfitting
#   - Tree pruning (max_delta_step, gamma)
#   - Parallel computation (faster training)
#   - Handles missing values natively
#   - Built-in cross-validation support
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Key parameters:
#   n_estimators    - number of trees
#   learning_rate   - shrinkage (eta)
#   max_depth       - tree depth
#   subsample       - row sampling per tree
#   colsample_bytree - column sampling per tree
#   reg_alpha       - L1 regularization
#   reg_lambda      - L2 regularization
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

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")
    print("Falling back to GradientBoostingClassifier for demonstration.\n")
    from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, precision_score,
                                     recall_score, f1_score)


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

MODEL_DIR  = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)
print("Dataset loaded.")
print("Shape:", df.shape)

def standardise_status(val):
    if val == "Closed":      return "Closed"
    if val == "In Progress": return "In Progress"
    return "Open"

df["status"] = df["status"].apply(standardise_status)

STATUS_MAP = {"Open": 0, "In Progress": 1, "Closed": 2}
df["status_encoded"] = df["status"].map(STATUS_MAP)

FEATURE_COLS = [
    c for c in [
        "complaint_type", "agency", "borough",
        "location_type", "open_data_channel_type",
        "council_district", "latitude", "longitude"
    ]
    if c in df.columns
]

TARGET_COL = "status_encoded"

df = df[FEATURE_COLS + [TARGET_COL]].dropna().copy()

for col in FEATURE_COLS:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 4. TRAIN XGBOOST MODEL
# ============================================================

if XGBOOST_AVAILABLE:
    print("\nTraining XGBoost Classifier...")
    print("  n_estimators=100 | learning_rate=0.1 | max_depth=5")
    print("  subsample=0.8 | colsample_bytree=0.8")
    print("  reg_alpha=0.1 (L1) | reg_lambda=1.0 (L2)")

    model = XGBClassifier(
        n_estimators      = 100,
        learning_rate     = 0.1,
        max_depth         = 5,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        objective         = "multi:softprob",
        num_class         = 3,
        eval_metric       = "mlogloss",
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        verbose               = 50
    )

else:
    # Fallback — sklearn GradientBoostingClassifier
    print("\nUsing GradientBoostingClassifier as XGBoost fallback...")
    model = GradientBoostingClassifier(
        n_estimators  = 100,
        learning_rate = 0.1,
        max_depth     = 5,
        subsample     = 0.8,
        random_state  = 42
    )
    model.fit(X_train, y_train)


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  XGBOOST — RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Open", "In Progress", "Closed"]
))


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% "
      f"(± {cv_scores.std()*100:.2f}%)")


# ============================================================
# 7. CONFUSION MATRIX CHART
# ============================================================

print("\nGenerating: Confusion Matrix...")

plt.figure(figsize=(7, 6))
sns.heatmap(
    conf_mat,
    annot        = True,
    fmt          = "d",
    cmap         = "YlOrBr",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"XGBoost — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgb_confusion_matrix.png"))
plt.close()
print("  Saved: xgb_confusion_matrix.png")


# ============================================================
# 8. FEATURE IMPORTANCE CHART
# ============================================================

print("Generating: Feature Importance chart...")

importances = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature"   : FEATURE_COLS,
    "Importance": importances
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(
    fi_df["Feature"],
    fi_df["Importance"],
    color="#e8760a",
    edgecolor="black"
)
plt.title("XGBoost — Feature Importances", fontsize=13)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "xgb_feature_importance.png"))
plt.close()
print("  Saved: xgb_feature_importance.png")


# ============================================================
# 9. REGULARIZATION COMPARISON (alpha vs accuracy)
# ============================================================

if XGBOOST_AVAILABLE:
    print("Generating: Regularization impact chart...")

    reg_alphas = [0, 0.01, 0.1, 1, 10]
    reg_acc    = []

    for ra in reg_alphas:
        xm = XGBClassifier(
            n_estimators = 100,
            learning_rate= 0.1,
            max_depth    = 5,
            reg_alpha    = ra,
            reg_lambda   = 1.0,
            objective    = "multi:softprob",
            num_class    = 3,
            eval_metric  = "mlogloss",
            random_state = 42,
            verbosity    = 0
        )
        xm.fit(X_train, y_train)
        reg_acc.append(xm.score(X_test, y_test) * 100)

    plt.figure(figsize=(8, 4))
    plt.semilogx([str(a) for a in reg_alphas], reg_acc,
                 marker="o", color="#e8760a", linewidth=2)
    plt.title("XGBoost — L1 Regularization (reg_alpha) vs Accuracy", fontsize=12)
    plt.xlabel("reg_alpha", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "xgb_regularization_impact.png"))
    plt.close()
    print("  Saved: xgb_regularization_impact.png")


# ============================================================
# 10. SAVE RESULTS AND MODEL
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : "XGBoost",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2),
    "CV_Mean"  : round(cv_scores.mean() * 100, 2),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "xgb_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
