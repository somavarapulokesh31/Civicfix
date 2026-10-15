# ============================================================
# CIVICFIX - ADABOOST CLASSIFIER (MODULE 3)
# ============================================================
# AdaBoost (Adaptive Boosting) is an ensemble method that
# combines multiple weak learners (shallow trees) into a
# strong classifier.
#
# How it works:
#   1. Train a weak classifier on the data
#   2. Increase the weight of misclassified samples
#   3. Train the next classifier on reweighted data
#   4. Repeat N times — combine all classifiers by voting
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Base estimator: DecisionTreeClassifier (max_depth=1)
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
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.tree            import DecisionTreeClassifier
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
    if val == "Closed":
        return "Closed"
    elif val in ["In Progress", "Assigned", "Pending"]:
        return "In Progress"
    else:   # Open, Unspecified, or anything else
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
# 4. N_ESTIMATORS SWEEP
# ============================================================

print("\n" + "=" * 60)
print("  ADABOOST — N_ESTIMATORS SWEEP")
print("=" * 60)

estimator_counts = [10, 50, 100, 200]
est_acc          = []

base_estimator = DecisionTreeClassifier(max_depth=1)

for n in estimator_counts:
    ada = AdaBoostClassifier(
        estimator    = base_estimator,
        n_estimators = n,
        learning_rate = 1.0,
        random_state = 42
    )
    ada.fit(X_train, y_train)
    acc = ada.score(X_test, y_test)
    est_acc.append(acc)
    print(f"  n_estimators={n:<4}: Test Accuracy = {acc*100:.2f}%")


# ============================================================
# 5. TRAIN FINAL ADABOOST MODEL
# ============================================================

print("\nTraining final AdaBoost (100 estimators, learning_rate=1.0)...")

model = AdaBoostClassifier(
    estimator     = DecisionTreeClassifier(max_depth=1),
    n_estimators  = 100,
    learning_rate = 1.0,
    random_state  = 42
)

model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  ADABOOST — RESULTS")
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
# 7. LEARNING CURVE (STAGED SCORE)
# ============================================================

print("\nGenerating: AdaBoost staged accuracy chart...")

staged_scores = [
    accuracy_score(y_test, y_pred_staged)
    for y_pred_staged in model.staged_predict(X_test)
]

plt.figure(figsize=(9, 4))
plt.plot(
    range(1, len(staged_scores) + 1),
    [s * 100 for s in staged_scores],
    color="#2878e8", linewidth=2
)
plt.title("AdaBoost — Staged Test Accuracy over Estimators", fontsize=13)
plt.xlabel("Number of Estimators (Boosting Rounds)", fontsize=11)
plt.ylabel("Test Accuracy (%)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "adaboost_staged_accuracy.png"))
plt.close()
print("  Saved: adaboost_staged_accuracy.png")


# ============================================================
# 8. CONFUSION MATRIX CHART
# ============================================================

print("Generating: Confusion Matrix...")

plt.figure(figsize=(7, 6))
sns.heatmap(
    conf_mat,
    annot        = True,
    fmt          = "d",
    cmap         = "Oranges",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"AdaBoost — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "adaboost_confusion_matrix.png"))
plt.close()
print("  Saved: adaboost_confusion_matrix.png")


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
    color="#f99a18",
    edgecolor="black"
)
plt.title("AdaBoost — Feature Importances", fontsize=13)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "adaboost_feature_importance.png"))
plt.close()
print("  Saved: adaboost_feature_importance.png")


# ============================================================
# 10. SAVE RESULTS AND MODEL
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : "AdaBoost (100 estimators)",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2),
    "CV_Mean"  : round(cv_scores.mean() * 100, 2),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "adaboost_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "adaboost.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"AdaBoost Accuracy: {accuracy * 100:.2f}%")
