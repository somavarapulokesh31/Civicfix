"""
============================================================
CIVICFIX - LOGISTIC REGRESSION MODEL
============================================================
Predicts complaint status: Open | In Progress | Closed
============================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, precision_score,
                                     recall_score, f1_score)
from _shared_data_loader import load_civicfix

MODEL_DIR  = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 3. LOAD DATA
# ============================================================
X, y, FEATURE_COLUMNS = load_civicfix()

# ============================================================
# 4. SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train):,}")
print(f"Testing  samples : {len(X_test):,}")

# ============================================================
# 5. TRAIN
# ============================================================
model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)
print("Training complete!")

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ============================================================
# 9. CALCULATE EVALUATION METRICS
# ============================================================

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
conf_mat  = confusion_matrix(y_test, y_pred)
cls_report = classification_report(
    y_test, y_pred,
    target_names=["Open", "In Progress", "Closed"]
)

# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("  LOGISTIC REGRESSION - RESULTS")
print("=" * 60)
print(f"  Accuracy  : {accuracy  * 100:.2f}%")
print(f"  Precision : {precision * 100:.2f}%")
print(f"  Recall    : {recall    * 100:.2f}%")
print(f"  F1-Score  : {f1        * 100:.2f}%")
print("\nClassification Report:")
print(cls_report)
print("Confusion Matrix:")
print(conf_mat)

# ============================================================
# 11. SAVE CONFUSION MATRIX CHART
# ============================================================

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_mat,
    annot    = True,
    fmt      = "d",
    cmap     = "Blues",
    xticklabels = ["Open", "In Progress", "Closed"],
    yticklabels = ["Open", "In Progress", "Closed"],
    linewidths  = 0.5
)
plt.title(
    f"Logistic Regression — Confusion Matrix\n"
    f"Accuracy: {accuracy*100:.2f}%",
    fontsize=13
)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label",    fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "lr_confusion_matrix.png"))
plt.close()
print("Confusion matrix chart saved.")

# ============================================================
# 12. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame([{
    "Model"    : "Logistic Regression",
    "Accuracy" : round(accuracy  * 100, 2),
    "Precision": round(precision * 100, 2),
    "Recall"   : round(recall    * 100, 2),
    "F1_Score" : round(f1        * 100, 2)
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "lr_results.csv"),
    index=False
)
print("Results CSV saved.")

# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression.pkl")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Logistic Regression Accuracy: {accuracy * 100:.2f}%")
