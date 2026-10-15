# ============================================================
# CIVICFIX - RANDOM FOREST CLASSIFIER (MODULE 3)
# ============================================================
# Random Forest builds many Decision Trees and combines
# their votes (bagging) to make a final prediction.
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Key advantages:
#   - Reduces overfitting compared to a single tree
#   - Handles missing values and noisy data well
#   - Provides feature importance scores
#
# Key hyperparameters:
#   n_estimators  - number of trees in the forest
#   max_depth     - max depth of each tree
#   max_features  - features to consider at each split
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
from sklearn.ensemble        import RandomForestClassifier
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
import sys
sys.path.insert(0, os.path.dirname(__file__))
from _shared_data_loader import load_civicfix

X, y, FEATURE_COLS = load_civicfix()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")


# ============================================================
# 4. N_ESTIMATORS SWEEP
# ============================================================

print("\n" + "=" * 60)
print("  RANDOM FOREST — N_ESTIMATORS SWEEP")
print("=" * 60)

estimator_counts = [10, 50, 100, 200]
est_acc          = []

for n in estimator_counts:
    rf = RandomForestClassifier(
        n_estimators = n,
        max_depth    = 15,
        random_state = 42,
        n_jobs       = -1
    )
    rf.fit(X_train, y_train)
    acc = rf.score(X_test, y_test)
    est_acc.append(acc)
    print(f"  n_estimators={n:<4}: Test Accuracy = {acc*100:.2f}%")


# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

print("\nTraining final Random Forest (100 trees, max_depth=15)...")

model = RandomForestClassifier(
    n_estimators      = 100,
    max_depth         = 15,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = "sqrt",
    random_state      = 42,
    n_jobs            = -1
)

model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  RANDOM FOREST — RESULTS")
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
    cmap         = "Blues",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"Random Forest — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_confusion_matrix.png"))
plt.close()
print("  Saved: rf_confusion_matrix.png")


# ============================================================
# 8. N_ESTIMATORS vs ACCURACY
# ============================================================

print("Generating: N_Estimators vs Accuracy chart...")

plt.figure(figsize=(8, 4))
plt.plot(
    estimator_counts, [a * 100 for a in est_acc],
    marker="o", color="#2878e8", linewidth=2
)
plt.title("Random Forest — N_Estimators vs Test Accuracy", fontsize=13)
plt.xlabel("Number of Trees", fontsize=11)
plt.ylabel("Test Accuracy (%)", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_estimators_vs_accuracy.png"))
plt.close()
print("  Saved: rf_estimators_vs_accuracy.png")


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
    color="#2878e8",
    edgecolor="black"
)
plt.title("Random Forest — Feature Importances", fontsize=13)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "rf_feature_importance.png"))
plt.close()
print("  Saved: rf_feature_importance.png")


# ============================================================
# 10. SAVE RESULTS AND MODEL
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : "Random Forest (100 trees)",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2),
    "CV_Mean"  : round(cv_scores.mean() * 100, 2),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "rf_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
