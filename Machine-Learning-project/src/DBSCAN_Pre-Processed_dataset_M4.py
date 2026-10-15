# ============================================================
# CIVICFIX - DBSCAN ON PRE-PROCESSED DATASET (MODULE 4)
# ============================================================
# DBSCAN (Density-Based Spatial Clustering of Applications
# with Noise) finds clusters of arbitrary shape without
# needing to specify the number of clusters K.
#
# Key concepts:
#   eps          - maximum distance between two points to be
#                  considered neighbours
#   min_samples  - minimum points required to form a core point
#   Core point   - has >= min_samples neighbours within eps
#   Border point - within eps of a core point but not core
#   Noise (-1)   - not reachable from any core point
#
# Applied here to the PREPROCESSED dataset (encoded + scaled)
# ============================================================

import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster       import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute         import SimpleImputer
from sklearn.decomposition  import PCA
from sklearn.metrics        import silhouette_score

BASE       = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS_CLEAN   = os.path.join(BASE, "dataset", "final_preprocess_M2.csv")
DS_RAW     = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT        = os.path.join(BASE, "outputs", "model_results")
os.makedirs(OUT, exist_ok=True)

# ── LOAD PRE-PROCESSED DATASET ───────────────────────────────
print("Loading pre-processed dataset (final_preprocess_M2.csv)...")
try:
    df = pd.read_csv(DS_CLEAN, low_memory=False)
    df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)
    TARGET = "status"
    FEATS  = [c for c in df.columns if c != TARGET]
    df_f   = df[FEATS].copy()
    # Encode any remaining string columns
    from sklearn.preprocessing import LabelEncoder
    for col in FEATS:
        if df_f[col].dtype == "object" or str(df_f[col].dtype) in ("string","StringDtype"):
            le = LabelEncoder()
            df_f[col] = le.fit_transform(df_f[col].astype(str))
        else:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df_f.values.astype(float))
    print(f"  Loaded: {X.shape}")
except Exception as e:
    print(f"  Pre-processed not found ({e}), using raw dataset...")
    from sklearn.preprocessing import LabelEncoder
    df    = pd.read_csv(DS_RAW, low_memory=False)
    df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)
    FEATS = ["complaint_type","agency","borough","location_type","council_district","latitude","longitude"]
    FEATS = [f for f in FEATS if f in df.columns]
    df_f  = df[FEATS].copy()
    for col in FEATS:
        if df_f[col].dtype == "object" or str(df_f[col].dtype) in ("string","StringDtype"):
            df_f[col] = df_f[col].fillna("Unknown")
            le = LabelEncoder()
            df_f[col] = le.fit_transform(df_f[col].astype(str))
        else:
            df_f[col] = pd.to_numeric(df_f[col], errors="coerce")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df_f.values.astype(float))

# Scale
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample
np.random.seed(42)
N   = min(10000, len(X_scaled))
idx = np.random.choice(len(X_scaled), N, replace=False)
Xs  = X_scaled[idx]
print(f"  Sample: {N:,}")

# ── EPS SENSITIVITY ──────────────────────────────────────────
print("\nEps sensitivity analysis...")
eps_vals  = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
eps_results = []

for eps in eps_vals:
    db     = DBSCAN(eps=eps, min_samples=10, n_jobs=-1)
    labels = db.fit_predict(Xs)
    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise= (labels == -1).sum()
    eps_results.append({"eps": eps, "clusters": n_cls, "noise": n_noise})
    print(f"  eps={eps}  clusters={n_cls}  noise={n_noise}")

# ── TRAIN FINAL DBSCAN ───────────────────────────────────────
BEST_EPS = 0.8
print(f"\nTraining DBSCAN (eps={BEST_EPS}, min_samples=10)...")
db_final = DBSCAN(eps=BEST_EPS, min_samples=10, n_jobs=-1)
labels   = db_final.fit_predict(Xs)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"  Clusters found : {n_clusters}")
print(f"  Noise points   : {n_noise} ({n_noise/len(Xs)*100:.1f}%)")

# Silhouette (exclude noise)
mask_valid = labels != -1
if mask_valid.sum() > 100 and n_clusters > 1:
    sil = silhouette_score(Xs[mask_valid], labels[mask_valid], sample_size=5000)
    print(f"  Silhouette     : {sil:.4f}")
else:
    sil = 0

# ── PCA 2D VISUALIZATION ─────────────────────────────────────
print("Saving: dbscan_preprocessed_clusters.png")
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(Xs)

COLORS = ["#2563eb","#16a34a","#f97316","#7c3aed","#0891b2",
          "#be185d","#d97706","#059669","#dc2626","#4f46e5"]

plt.figure(figsize=(10, 7))
unique_labels = sorted(set(labels))
for lbl in unique_labels:
    mask = labels == lbl
    if lbl == -1:
        plt.scatter(X2d[mask,0], X2d[mask,1], s=5, alpha=0.2,
                    color="#94a3b8", label=f"Noise ({mask.sum()})")
    else:
        plt.scatter(X2d[mask,0], X2d[mask,1], s=10, alpha=0.5,
                    color=COLORS[lbl % len(COLORS)],
                    label=f"Cluster {lbl} ({mask.sum()})")

plt.title(f"DBSCAN on Pre-Processed Dataset (eps={BEST_EPS})\n"
          f"{n_clusters} clusters  |  {n_noise} noise points", fontsize=13)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(fontsize=9, markerscale=2, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dbscan_preprocessed_clusters.png"), dpi=100)
plt.close()

# ── EPS SENSITIVITY CHART ─────────────────────────────────────
print("Saving: dbscan_preprocessed_eps_sensitivity.png")
eps_df = pd.DataFrame(eps_results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(eps_df["eps"], eps_df["clusters"], marker="o", color="#2563eb", linewidth=2.5)
axes[0].axvline(BEST_EPS, color="#ef4444", linestyle="--", linewidth=2, label=f"Selected eps={BEST_EPS}")
axes[0].set_title("Eps vs Number of Clusters"); axes[0].set_xlabel("eps"); axes[0].set_ylabel("Clusters")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(eps_df["eps"], eps_df["noise"],    marker="s", color="#f97316", linewidth=2.5)
axes[1].axvline(BEST_EPS, color="#ef4444", linestyle="--", linewidth=2, label=f"Selected eps={BEST_EPS}")
axes[1].set_title("Eps vs Noise Points"); axes[1].set_xlabel("eps"); axes[1].set_ylabel("Noise Points")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.suptitle("DBSCAN (Pre-Processed) — Eps Sensitivity Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dbscan_preprocessed_eps_sensitivity.png"), dpi=100)
plt.close()

print(f"\nDBSCAN (Pre-Processed) complete. Clusters={n_clusters}, Noise={n_noise}")
