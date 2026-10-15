# ============================================================
# CIVICFIX - DECISION TREE CLASSIFIER (MODULE 3)
# ============================================================
# A Decision Tree splits the data into branches based on
# feature thresholds to reach leaf nodes containing a class.
#
# For CivicFix, we predict complaint status:
#   Open (0) | In Progress (1) | Closed (2)
#
# Key hyperparameters explored:
#   max_depth    - controls tree depth (prevents overfitting)
#   criterion    - "gini" (impurity) or "entropy" (info gain)
#   min_samples_split - min samples required to split a node
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
from sklearn.tree            import DecisionTreeClassifier, plot_tree
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
# 4. EXPLORE DIFFERENT MAX DEPTHS
# ============================================================

print("\n" + "=" * 60)
print("  DECISION TREE — MAX DEPTH ANALYSIS")
print("=" * 60)

depths       = [3, 5, 7, 10, 15, None]
depth_acc    = []
depth_train  = []

for d in depths:
    dt = DecisionTreeClassifier(
        max_depth    = d,
        criterion    = "gini",
        random_state = 42
    )
    dt.fit(X_train, y_train)
    train_acc = dt.score(X_train, y_train)
    test_acc  = dt.score(X_test,  y_test)
    depth_acc.append(test_acc)
    depth_train.append(train_acc)

    label = str(d) if d is not None else "None"
    print(f"  max_depth={label:<5}: Train={train_acc*100:.2f}%  "
          f"Test={test_acc*100:.2f}%")


# ============================================================
# 5. TRAIN FINAL MODEL (best depth = 10)
# ============================================================

BEST_DEPTH = 10

print(f"\nTraining final Decision Tree with max_depth={BEST_DEPTH}...")

model = DecisionTreeClassifier(
    max_depth         = BEST_DEPTH,
    criterion         = "gini",
    min_samples_split = 10,
    min_samples_leaf  = 5,
    random_state      = 42
)

model.fit(X_train, y_train)

y_pred    = model.predict(X_test)
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("  DECISION TREE — FINAL RESULTS")
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
print(f"5-Fold Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% "
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
    cmap         = "Greens",
    xticklabels  = ["Open", "In Progress", "Closed"],
    yticklabels  = ["Open", "In Progress", "Closed"],
    linewidths   = 0.5
)
plt.title(
    f"Decision Tree — Confusion Matrix\nAccuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dt_confusion_matrix.png"))
plt.close()
print("  Saved: dt_confusion_matrix.png")


# ============================================================
# 8. MAX DEPTH vs ACCURACY CHART
# ============================================================

print("Generating: Max Depth vs Accuracy chart...")

depth_labels = [str(d) if d is not None else "None" for d in depths]

plt.figure(figsize=(9, 5))
plt.plot(depth_labels, [a * 100 for a in depth_train],
         marker="o", label="Train Accuracy", color="#2878e8", linewidth=2)
plt.plot(depth_labels, [a * 100 for a in depth_acc],
         marker="s", label="Test Accuracy",  color="#f99a18", linewidth=2)
plt.axvline(x=str(BEST_DEPTH), color="red",
            linestyle="--", linewidth=1.5, label=f"Selected (depth={BEST_DEPTH})")
plt.title("Decision Tree — Max Depth vs Accuracy", fontsize=13)
plt.xlabel("Max Depth", fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dt_depth_vs_accuracy.png"))
plt.close()
print("  Saved: dt_depth_vs_accuracy.png")


# ============================================================
# 9. FEATURE IMPORTANCE CHART
# ============================================================

print("Generating: Feature Importance chart...")

fi_df = pd.DataFrame({
    "Feature"   : FEATURE_COLS,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 5))
plt.barh(fi_df["Feature"], fi_df["Importance"],
         color="#4aa84c", edgecolor="black")
plt.title("Decision Tree — Feature Importances", fontsize=13)
plt.xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dt_feature_importance.png"))
plt.close()
print("  Saved: dt_feature_importance.png")


# ============================================================
# 10. VISUALIZE SHALLOW TREE (depth=3 for readability)
# ============================================================

print("Generating: Decision Tree visualization (depth=3)...")

dt_viz = DecisionTreeClassifier(
    max_depth    = 3,
    criterion    = "gini",
    random_state = 42
)
dt_viz.fit(X_train, y_train)

plt.figure(figsize=(20, 8))
plot_tree(
    dt_viz,
    feature_names = FEATURE_COLS,
    class_names   = ["Open", "In Progress", "Closed"],
    filled        = True,
    rounded       = True,
    fontsize      = 9
)
plt.title("Decision Tree (depth=3) — CivicFix Status Prediction", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dt_tree_visualization.png"), dpi=150)
plt.close()
print("  Saved: dt_tree_visualization.png")


# ============================================================
# 11. SAVE RESULTS AND MODEL
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : f"Decision Tree (depth={BEST_DEPTH})",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2),
    "CV_Mean"  : round(cv_scores.mean() * 100, 2),
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "dt_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Decision Tree Accuracy: {accuracy * 100:.2f}%")
