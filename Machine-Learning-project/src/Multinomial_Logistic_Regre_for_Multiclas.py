# ============================================================
# CIVICFIX - MULTINOMIAL LOGISTIC REGRESSION
# ============================================================
# Multinomial (Multi-Class) Logistic Regression predicts
# ONE of THREE or more classes using the Softmax function.
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Softmax converts raw scores (logits) to probabilities:
#   P(class k) = e^(zk) / Σ e^(zj)
#
# The class with the highest probability is predicted.
#
# Solvers compared:
#   lbfgs        - default, works well for small-medium data
#   saga         - faster on large datasets, supports L1
#   newton-cg    - good for dense data
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
from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder, StandardScaler
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
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for Multinomial Logistic Regression.")
print("Shape:", df.shape)


# ============================================================
# 4. STANDARDISE STATUS TO 3 CLASSES
# ============================================================

def standardise_status(val):
    if val == "Closed":
        return "Closed"
    elif val in ["In Progress", "Assigned", "Pending"]:
        return "In Progress"
    else:
        return "Open"


df["status"] = df["status"].apply(standardise_status)

STATUS_MAP = {"Open": 0, "In Progress": 1, "Closed": 2}
df["status_encoded"] = df["status"].map(STATUS_MAP)

print("\nStatus distribution (3 classes):")
print(df["status"].value_counts())


# ============================================================
# 5. SELECT FEATURES
# ============================================================

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

print(f"\nFeature columns ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Rows used: {len(df):,}")


# ============================================================
# 6. ENCODE CATEGORICAL FEATURES
# ============================================================

for col in FEATURE_COLS:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))


# ============================================================
# 7. SCALE FEATURES
# ============================================================

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

scaler = StandardScaler()
X      = scaler.fit_transform(X)


# ============================================================
# 8. SPLIT INTO TRAIN AND TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y
)

print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 9. COMPARE MULTIPLE SOLVERS
# ============================================================

SOLVERS = ["lbfgs", "saga", "newton-cg"]

solver_results = {}

print("\n" + "=" * 60)
print("  COMPARING SOLVERS FOR MULTINOMIAL LOGISTIC REGRESSION")
print("=" * 60)

for solver in SOLVERS:
    model = LogisticRegression(
        multi_class  = "multinomial",
        solver       = solver,
        max_iter     = 2000,
        random_state = 42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

    solver_results[solver] = {
        "model" : model,
        "y_pred": y_pred,
        "acc"   : acc,
        "prec"  : prec,
        "rec"   : rec,
        "f1"    : f1
    }

    print(f"\n  Solver: {solver}")
    print(f"    Accuracy  : {acc  * 100:.2f}%")
    print(f"    Precision : {prec * 100:.2f}%")
    print(f"    Recall    : {rec  * 100:.2f}%")
    print(f"    F1-Score  : {f1   * 100:.2f}%")

print("=" * 60)


# ============================================================
# 10. SELECT BEST MODEL (lbfgs — standard choice)
# ============================================================

best_model  = solver_results["lbfgs"]["model"]
best_y_pred = solver_results["lbfgs"]["y_pred"]
best_acc    = solver_results["lbfgs"]["acc"]

print("\n\nDetailed report for lbfgs solver:")
print(classification_report(
    y_test, best_y_pred,
    target_names=["Open", "In Progress", "Closed"]
))


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("Generating: Confusion Matrix...")

conf_mat = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    conf_mat,
    annot        = True,
    fmt          = "d",
    cmap         = "Blues",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"Multinomial Logistic Regression — Confusion Matrix\n"
    f"Accuracy: {best_acc*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multinomial_lr_confusion_matrix.png"))
plt.close()
print("  Saved: multinomial_lr_confusion_matrix.png")


# ============================================================
# 12. SOLVER COMPARISON CHART
# ============================================================

print("Generating: Solver comparison chart...")

names   = list(solver_results.keys())
accs    = [solver_results[s]["acc"] * 100 for s in names]
f1s     = [solver_results[s]["f1"]  * 100 for s in names]

x = np.arange(len(names))
width = 0.35

plt.figure(figsize=(9, 5))
bars1 = plt.bar(x - width/2, accs, width, label="Accuracy",
                color="#2878e8", edgecolor="black")
bars2 = plt.bar(x + width/2, f1s,  width, label="F1-Score",
                color="#4aa84c", edgecolor="black")

for bar in bars1 + bars2:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{bar.get_height():.1f}%",
        ha="center", va="bottom", fontsize=9
    )

plt.title("Multinomial LR — Solver Comparison", fontsize=13)
plt.xlabel("Solver", fontsize=11)
plt.ylabel("Score (%)", fontsize=11)
plt.xticks(x, names)
plt.legend()
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multinomial_lr_solver_comparison.png"))
plt.close()
print("  Saved: multinomial_lr_solver_comparison.png")


# ============================================================
# 13. PROBABILITY DISTRIBUTION FOR SAMPLE PREDICTIONS
# ============================================================

print("Generating: Probability distribution chart...")

sample_proba = best_model.predict_proba(X_test[:5])

plt.figure(figsize=(9, 4))
width = 0.25
x_pos = np.arange(5)

for i, (label, color) in enumerate(
    zip(["Open", "In Progress", "Closed"],
        ["#f99a18", "#2878e8", "#4aa84c"])
):
    plt.bar(x_pos + i * width, sample_proba[:, i],
            width=width, label=label, color=color, edgecolor="black")

plt.title("Multinomial LR — Prediction Probabilities (5 Samples)", fontsize=12)
plt.xlabel("Sample Index", fontsize=10)
plt.ylabel("Probability",  fontsize=10)
plt.xticks(x_pos + width, [f"#{i}" for i in range(5)])
plt.legend(title="Predicted Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "multinomial_lr_probabilities.png"))
plt.close()
print("  Saved: multinomial_lr_probabilities.png")


# ============================================================
# 14. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([
    {
        "Solver"    : solver,
        "Accuracy"  : round(res["acc"]  * 100, 2),
        "Precision" : round(res["prec"] * 100, 2),
        "Recall"    : round(res["rec"]  * 100, 2),
        "F1_Score"  : round(res["f1"]   * 100, 2),
    }
    for solver, res in solver_results.items()
])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "multinomial_lr_results.csv"),
    index=False
)
print("Results CSV saved.")


# ============================================================
# 15. SAVE BEST MODEL
# ============================================================

MODEL_PATH = os.path.join(MODEL_DIR, "multinomial_logistic_regression.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

print(f"\nBest model (lbfgs) saved to: {MODEL_PATH}")
print(f"Multinomial LR Accuracy: {best_acc * 100:.2f}%")
