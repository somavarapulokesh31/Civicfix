# ============================================================
# CIVICFIX - RUN ALL MODELS (generates all charts + CSVs)
# ============================================================
import os, sys, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics         import (accuracy_score, classification_report,
                                     confusion_matrix, precision_score,
                                     recall_score, f1_score)

BASE      = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS_PATH   = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT_DIR   = os.path.join(BASE, "outputs", "model_results")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. LOAD & PREPARE ────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DS_PATH, low_memory=False)
df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)

def std_status(v):
    if v == "Closed":      return "Closed"
    if v in ["In Progress","Assigned","Pending"]: return "In Progress"
    return "Open"

df["status"] = df["status"].apply(std_status)
STATUS_MAP   = {"Open": 0, "In Progress": 1, "Closed": 2}
df["target"] = df["status"].map(STATUS_MAP)

FEATURES = ["complaint_type","agency","borough","location_type",
            "open_data_channel_type","council_district","latitude","longitude"]
FEATURES = [f for f in FEATURES if f in df.columns]

df = df[FEATURES + ["target","status"]].copy()

# Fill missing
for col in FEATURES:
    if df[col].dtype == "object" or str(df[col].dtype) in ("string","StringDtype"):
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

df.dropna(subset=["target"], inplace=True)
df["target"] = df["target"].astype(int)

# Encode categories
label_encoders = {}
for col in FEATURES:
    if df[col].dtype == "object" or str(df[col].dtype) in ("string","StringDtype"):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df[FEATURES].values.astype(float)
y = df["target"].values.astype(int)

# Reliable NaN imputation using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

print(f"  Rows: {len(X):,}  |  Features: {FEATURES}")
print(f"  Classes: {dict(zip(*np.unique(y, return_counts=True)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

CLASS_NAMES = ["Open", "In Progress", "Closed"]

# ── 2. DEFINE MODELS ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
    "AdaBoost"           : AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42),
}

all_results = []

for name, model in models.items():
    slug = name.lower().replace(" ","_")
    print(f"\nTraining: {name}...")

    model.fit(X_train, y_train)
    y_pred    = model.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    prec      = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec       = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0)
    conf_mat  = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy={acc*100:.2f}%  F1={f1*100:.2f}%")

    all_results.append({
        "Model"    : name,
        "Accuracy" : round(acc*100, 2),
        "Precision": round(prec*100, 2),
        "Recall"   : round(rec*100, 2),
        "F1_Score" : round(f1*100, 2),
    })

    # Confusion Matrix
    plt.figure(figsize=(7, 5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
    plt.title(f"{name} — Confusion Matrix\nAccuracy: {acc*100:.2f}%", fontsize=13)
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{slug}_confusion_matrix.png"), dpi=100)
    plt.close()

    # Feature Importance (if available)
    fi = None
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "coef_"):
        fi = np.abs(model.coef_).mean(axis=0)

    if fi is not None and len(fi) == len(FEATURES):
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": fi})
        fi_df = fi_df.sort_values("Importance", ascending=True)
        plt.figure(figsize=(8, 5))
        plt.barh(fi_df["Feature"], fi_df["Importance"], color="#2563eb", edgecolor="black")
        plt.title(f"{name} — Feature Importances", fontsize=13)
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{slug}_feature_importance.png"), dpi=100)
        plt.close()

    # Save model
    with open(os.path.join(MODEL_DIR, f"{slug}.pkl"), "wb") as f:
        pickle.dump(model, f)

# ── 3. MODEL COMPARISON CHART ─────────────────────────────────
print("\nGenerating: Model comparison chart...")
res_df = pd.DataFrame(all_results)

x    = np.arange(len(res_df))
w    = 0.2
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w*1.5, res_df["Accuracy"],  w, label="Accuracy",  color="#16a34a", edgecolor="black")
ax.bar(x - w*0.5, res_df["Precision"], w, label="Precision", color="#2563eb", edgecolor="black")
ax.bar(x + w*0.5, res_df["Recall"],    w, label="Recall",    color="#f97316", edgecolor="black")
ax.bar(x + w*1.5, res_df["F1_Score"],  w, label="F1-Score",  color="#7c3aed", edgecolor="black")

ax.set_xticks(x)
ax.set_xticklabels(res_df["Model"], fontsize=11)
ax.set_ylim(0, 110)
ax.set_ylabel("Score (%)")
ax.set_title("CivicFix — ML Model Comparison (Accuracy / Precision / Recall / F1)", fontsize=13)
ax.legend()
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.4,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=100)
plt.close()
print("  Saved: model_comparison.png")

# ── 4. SAVE RESULTS CSV ────────────────────────────────────────
res_df.to_csv(os.path.join(OUT_DIR, "all_model_results.csv"), index=False)
print("  Saved: all_model_results.csv")

print(f"\n{'='*55}")
print("  ALL MODELS TRAINED — results in outputs/model_results/")
print(f"{'='*55}")
print(res_df.to_string(index=False))
