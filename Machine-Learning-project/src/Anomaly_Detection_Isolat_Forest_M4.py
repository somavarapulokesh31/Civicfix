# ============================================================
# CIVICFIX - ANOMALY DETECTION: ISOLATION FOREST (MODULE 4)
# ============================================================
# Anomaly Detection finds unusual/suspicious complaints that
# don't follow normal patterns in the dataset.
#
# ISOLATION FOREST:
#   - Builds random trees to isolate data points
#   - Anomalies are isolated in fewer splits (shorter path)
#   - Normal points require more splits to isolate
#   - contamination = expected % of anomalies (e.g. 5%)
#
# For CivicFix, anomalies could be:
#   - Complaints with very unusual location coordinates
#   - Rare complaint type + agency combinations
#   - Extreme resolution times
# ============================================================

import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing    import LabelEncoder
from sklearn.impute            import SimpleImputer
from sklearn.ensemble          import IsolationForest
from sklearn.decomposition     import PCA

BASE = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS   = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT  = os.path.join(BASE, "outputs", "model_results")
MDL  = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(MDL, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────
print("Loading CivicFix dataset...")
df = pd.read_csv(DS, low_memory=False)
df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)

FEATURES = ["complaint_type","agency","borough","location_type",
            "open_data_channel_type","council_district","latitude","longitude"]
FEATURES = [f for f in FEATURES if f in df.columns]

df_f = df[FEATURES].copy()
for col in FEATURES:
    if df_f[col].dtype == "object" or str(df_f[col].dtype) in ("string","StringDtype"):
        df_f[col] = df_f[col].fillna("Unknown")
        le = LabelEncoder()
        df_f[col] = le.fit_transform(df_f[col].astype(str))
    else:
        df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

imp = SimpleImputer(strategy="median")
X   = imp.fit_transform(df_f.values.astype(float))
print(f"  Rows: {len(X):,}  |  Features: {len(FEATURES)}")

# Sample for speed
np.random.seed(42)
N   = min(20000, len(X))
idx = np.random.choice(len(X), N, replace=False)
Xs  = X[idx]
print(f"  Sample: {N:,} rows")

# ── ISOLATION FOREST ─────────────────────────────────────────
CONTAMINATION = 0.05   # expect 5% anomalies
print(f"\nTraining Isolation Forest (contamination={CONTAMINATION})...")

iso = IsolationForest(
    n_estimators  = 200,
    contamination = CONTAMINATION,
    random_state  = 42,
    n_jobs        = -1
)
predictions  = iso.fit_predict(Xs)   # -1 = anomaly, 1 = normal
scores       = iso.score_samples(Xs)  # lower = more anomalous

n_anomalies = (predictions == -1).sum()
n_normal    = (predictions ==  1).sum()
anomaly_pct = n_anomalies / len(Xs) * 100

print(f"  Normal    : {n_normal:,}  ({100-anomaly_pct:.1f}%)")
print(f"  Anomalies : {n_anomalies:,}  ({anomaly_pct:.1f}%)")

# ── PCA 2D VISUALIZATION ─────────────────────────────────────
print("Saving: anomaly_detection_pca.png")
pca  = PCA(n_components=2, random_state=42)
X2d  = pca.fit_transform(Xs)

plt.figure(figsize=(10, 7))
mask_normal  = predictions == 1
mask_anomaly = predictions == -1

plt.scatter(X2d[mask_normal, 0],  X2d[mask_normal, 1],
            s=8, alpha=0.3, color="#2563eb", label=f"Normal ({n_normal:,})")
plt.scatter(X2d[mask_anomaly, 0], X2d[mask_anomaly, 1],
            s=20, alpha=0.7, color="#ef4444", label=f"Anomaly ({n_anomalies:,})", zorder=5)

plt.title(f"Isolation Forest — Anomaly Detection (CivicFix)\n"
          f"Contamination={CONTAMINATION}  |  {n_anomalies:,} anomalies detected ({anomaly_pct:.1f}%)",
          fontsize=13)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(fontsize=11, markerscale=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "anomaly_detection_pca.png"), dpi=100)
plt.close()

# ── ANOMALY SCORE DISTRIBUTION ───────────────────────────────
print("Saving: anomaly_score_distribution.png")
plt.figure(figsize=(10, 5))
plt.hist(scores[mask_normal],  bins=60, alpha=0.7, color="#2563eb", label="Normal",   density=True)
plt.hist(scores[mask_anomaly], bins=30, alpha=0.7, color="#ef4444", label="Anomaly",  density=True)
plt.axvline(np.percentile(scores, CONTAMINATION*100), color="black", linestyle="--", linewidth=2,
            label=f"Decision Threshold ({CONTAMINATION*100:.0f}th percentile)")
plt.title("Isolation Forest — Anomaly Score Distribution\n"
          "(Lower score = more anomalous)", fontsize=13)
plt.xlabel("Anomaly Score")
plt.ylabel("Density")
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "anomaly_score_distribution.png"), dpi=100)
plt.close()

# ── CONTAMINATION SENSITIVITY ────────────────────────────────
print("Saving: anomaly_contamination_sensitivity.png")
cont_vals     = [0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
anomaly_counts= []
for c in cont_vals:
    iso_tmp = IsolationForest(n_estimators=100, contamination=c,
                              random_state=42, n_jobs=-1)
    preds   = iso_tmp.fit_predict(Xs)
    anomaly_counts.append((preds == -1).sum())

plt.figure(figsize=(9, 4))
plt.plot([c*100 for c in cont_vals], anomaly_counts,
         marker="o", color="#ef4444", linewidth=2.5, markersize=8)
plt.axvline(CONTAMINATION*100, color="#16a34a", linestyle="--",
            linewidth=2, label=f"Selected: {CONTAMINATION*100:.0f}%")
plt.title("Isolation Forest — Contamination % vs Anomaly Count", fontsize=13)
plt.xlabel("Contamination (%)")
plt.ylabel("Number of Anomalies Detected")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "anomaly_contamination_sensitivity.png"), dpi=100)
plt.close()

# ── SAVE MODEL ───────────────────────────────────────────────
with open(os.path.join(MDL, "isolation_forest.pkl"), "wb") as f:
    pickle.dump({"model": iso, "imputer": imp, "pca": pca}, f)

print(f"\nIsolation Forest complete.")
print(f"  Anomalies detected: {n_anomalies:,} ({anomaly_pct:.1f}%)")
threshold = np.percentile(scores, CONTAMINATION*100)
print(f"  Decision threshold : {threshold:.4f}")
