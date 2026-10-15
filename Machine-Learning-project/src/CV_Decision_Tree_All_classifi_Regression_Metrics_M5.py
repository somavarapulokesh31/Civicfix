# ============================================================
# CIVICFIX - CROSS VALIDATION: ALL CLASSIFIERS (MODULE 3)
# ============================================================
# Cross-Validation evaluates model performance more reliably
# than a single train-test split by testing the model on K
# different data folds.
#
# K-Fold CV splits data into K equal parts:
#   - Train on K-1 parts, test on 1 part
#   - Repeat K times, each fold used as test once
#   - Final score = mean of K scores
#
# This script runs 5-Fold CV on ALL classifiers and compares:
#   Logistic Regression, Decision Tree, Random Forest,
#   Gradient Boosting, AdaBoost
# Metrics: Accuracy, Precision, Recall, F1-Score
# ============================================================

import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing   import LabelEncoder
from sklearn.impute           import SimpleImputer
from sklearn.model_selection  import cross_val_score, StratifiedKFold
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                      GradientBoostingClassifier,
                                      AdaBoostClassifier)
from sklearn.metrics          import make_scorer, accuracy_score

BASE    = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS      = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT     = os.path.join(BASE, "outputs", "model_results")
os.makedirs(OUT, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DS, low_memory=False)
df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)

def std_status(v):
    if v == "Closed":      return 2
    if v in ["In Progress","Assigned","Pending"]: return 1
    return 0

df["target"] = df["status"].apply(std_status)

FEATURES = ["complaint_type","agency","borough","location_type",
            "open_data_channel_type","council_district","latitude","longitude"]
FEATURES = [f for f in FEATURES if f in df.columns]

df_f = df[FEATURES + ["target"]].copy()
for col in FEATURES:
    if df_f[col].dtype == "object" or str(df_f[col].dtype) in ("string","StringDtype"):
        df_f[col] = df_f[col].fillna("Unknown")
    else:
        df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

for col in FEATURES:
    if df_f[col].dtype == "object" or str(df_f[col].dtype) in ("string","StringDtype"):
        le = LabelEncoder()
        df_f[col] = le.fit_transform(df_f[col].astype(str))

df_f.dropna(subset=["target"], inplace=True)
imp = SimpleImputer(strategy="median")
X   = imp.fit_transform(df_f[FEATURES].values.astype(float))
y   = df_f["target"].values.astype(int)
print(f"  Rows: {len(X):,}  |  Features: {len(FEATURES)}")

# ── MODELS ───────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=500, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, max_depth=15,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                       max_depth=4, random_state=42),
    "AdaBoost"           : AdaBoostClassifier(n_estimators=100, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

print("\n5-Fold Cross-Validation Results:")
print("=" * 60)

for name, model in models.items():
    print(f"\n  {name}...")
    acc  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring="precision_weighted")
    rec  = cross_val_score(model, X, y, cv=cv, scoring="recall_weighted")
    f1   = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    results.append({
        "Model"       : name,
        "CV_Accuracy" : round(acc.mean()*100, 2),
        "CV_Acc_Std"  : round(acc.std()*100, 2),
        "CV_Precision": round(prec.mean()*100, 2),
        "CV_Recall"   : round(rec.mean()*100, 2),
        "CV_F1"       : round(f1.mean()*100, 2),
    })
    print(f"    Accuracy : {acc.mean()*100:.2f}% ± {acc.std()*100:.2f}%")
    print(f"    F1-Score : {f1.mean()*100:.2f}%")

# ── CV COMPARISON CHART ──────────────────────────────────────
print("\nGenerating: CV comparison chart...")
res_df = pd.DataFrame(results).sort_values("CV_Accuracy", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(res_df))
w = 0.18
colors = ["#16a34a","#2563eb","#f97316","#7c3aed"]
metrics = ["CV_Accuracy","CV_Precision","CV_Recall","CV_F1"]
mlabels = ["Accuracy","Precision","Recall","F1-Score"]

for i,(m,lbl,clr) in enumerate(zip(metrics,mlabels,colors)):
    bars = ax.bar(x + (i-1.5)*w, res_df[m], w, label=lbl, color=clr,
                  edgecolor="white", borderpad=0)
    for bar in bars:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{bar.get_height():.1f}", ha="center", fontsize=7.5)

# Error bars for accuracy std
ax.errorbar(x, res_df["CV_Accuracy"], yerr=res_df["CV_Acc_Std"],
            fmt="none", color="black", capsize=4, linewidth=1.5)

ax.set_xticks(x)
ax.set_xticklabels(res_df["Model"], fontsize=11)
ax.set_ylim(0, 115)
ax.set_ylabel("Score (%)")
ax.set_title("CivicFix — 5-Fold Cross-Validation: All Classifiers\n"
             "(Error bars = ± std deviation on Accuracy)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cv_all_classifiers.png"), dpi=100)
plt.close()
print("  Saved: cv_all_classifiers.png")

# ── SAVE CSV ─────────────────────────────────────────────────
res_df.to_csv(os.path.join(OUT, "cv_results.csv"), index=False)
print("\nCross-Validation complete.")
print(res_df[["Model","CV_Accuracy","CV_Acc_Std","CV_F1"]].to_string(index=False))
