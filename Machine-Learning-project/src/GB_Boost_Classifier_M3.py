# ============================================================
# CIVICFIX - GRADIENT BOOSTING CLASSIFIER (MODULE 3)
# ============================================================
# Gradient Boosting builds trees sequentially, where each
# new tree corrects the residual errors of the previous one.
#
# Unlike AdaBoost (which reweights samples), Gradient Boosting
# fits the next tree to the NEGATIVE GRADIENT of the loss.
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Key hyperparameters:
#   n_estimators  - number of boosting stages
#   learning_rate - shrinks each tree's contribution (step size)
#   max_depth     - depth of individual trees (usually 3-5)
#   subsample     - fraction of samples used per tree (stochastic GB)
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
from sklearn.ensemble        import GradientBoostingClassifier
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
# 4. HYPERPARAMETER COMBINATIONS
# ============================================================

print("\n" + "=" * 60)
print("  GRADIENT BOOSTING — HYPERPARAMETER SEARCH")
print("=" * 60)

configs = [
    {"n_estimators": 50,  "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
]

best_acc   = 0
best_cfg   = None
best_model = None

for cfg in configs:
    gb = GradientBoostingClassifier(
        n_estimators  = cfg["n_estimators"],
        learning_rate = cfg["learning_rate"],
        max_depth     = cfg["max_depth"],
        subsample     = 0.8,
        random_state  = 42
    )
    gb.fit(X_train, y_train)
    acc = gb.score(X_test, y_test)

    print(
        f"  n={cfg['n_estimators']:<4} lr={cfg['learning_rate']} "
        f"depth={cfg['max_depth']} → "
        f"Accuracy={acc*100:.2f}%"
    )

    if acc > best_acc:
        best_acc   = acc
        best_cfg   = cfg
        best_model = gb

print(f"\n  Best config: {best_cfg}  → Accuracy={best_acc*100:.2f}%")


# ============================================================
# 5. EVALUATE BEST MODEL
# ============================================================

model     = best_model
y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  GRADIENT BOOSTING — BEST MODEL RESULTS")
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
# 7. DEVIANCE (LOSS) CURVE
# ============================================================

print("\nGenerating: Deviance (training loss) curve...")

train_score = model.train_score_

plt.figure(figsize=(9, 4))
plt.plot(
    np.arange(len(train_score)) + 1,
    train_score,
    color="#2878e8", linewidth=2, label="Training Loss"
)
plt.title(
    f"Gradient Boosting — Training Loss Curve\n"
    f"({best_cfg['n_estimators']} trees, lr={best_cfg['learning_rate']}, "
    f"depth={best_cfg['max_depth']})",
    fontsize=13
)
plt.xlabel("Boosting Stage", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gb_loss_curve.png"))
plt.close()
print("  Saved: gb_loss_curve.png")


# ============================================================
# 8. CONFUSION MATRIX CHART
# ============================================================

print("Generating: Confusion Matrix...")

plt.figure(figsize=(7, 6))
sns.heatmap(
    conf_mat,
    annot        = True,
    fmt          = "d",
    cmap         = "Purples",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"Gradient Boosting — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gb_confusion_matrix.png"))
plt.close()
print("  Saved: gb_confusion_matrix.png")


# ============================================================
# 9. FEATURE IMPORTANCE CHART
# ============================================================

print("Generating: Feature Importance chart...")

fi_df = pd.DataFrame({
    "Feature"   : FEATURE_COLS,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(
    fi_df["Feature"],
    fi_df["Importance"],
    color="#7b3fa8",
    edgecolor="black"
)
plt.title("Gradient Boosting — Feature Importances", fontsize=13)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gb_feature_importance.png"))
plt.close()
print("  Saved: gb_feature_importance.png")


# ============================================================
# 10. SAVE RESULTS AND MODEL
# ============================================================

results_df = pd.DataFrame([{
    "Model"        : "Gradient Boosting",
    "n_estimators" : best_cfg["n_estimators"],
    "learning_rate": best_cfg["learning_rate"],
    "max_depth"    : best_cfg["max_depth"],
    "Accuracy"     : round(accuracy  * 100, 2),
    "Precision"    : round(precision * 100, 2),
    "Recall"       : round(recall    * 100, 2),
    "F1_Score"     : round(f1        * 100, 2),
    "CV_Mean"      : round(cv_scores.mean() * 100, 2),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "gb_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Gradient Boosting Accuracy: {accuracy * 100:.2f}%")
