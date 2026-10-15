# ============================================================
# CIVICFIX - DBSCAN ON RAW DATASET (MODULE 4)
# ============================================================
# DBSCAN applied directly to the raw dataset using only
# geographical features (latitude, longitude) which are
# already numerical and don't need encoding.
#
# This is useful for finding geographic complaint hotspots —
# areas of NYC with unusual complaint density patterns.
#
# Difference from DBSCAN_Pre-Processed:
#   - Uses only lat/lng (no encoding needed)
#   - Directly interpretable as NYC map clusters
#   - Noise = isolated complaints far from any cluster
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster    import DBSCAN
from sklearn.impute      import SimpleImputer
from sklearn.metrics     import silhouette_score

BASE = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS   = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT  = os.path.join(BASE, "outputs", "model_results")
os.makedirs(OUT, exist_ok=True)

# ── LOAD RAW DATASET ─────────────────────────────────────────
print("Loading raw dataset...")
df = pd.read_csv(DS, low_memory=False)
df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)

# Use lat/lng for geographic clustering
GEO_FEATURES = ["latitude", "longitude"]
df_geo = df[GEO_FEATURES].copy()
df_geo["latitude"]  = pd.to_numeric(df_geo["latitude"],  errors="coerce")
df_geo["longitude"] = pd.to_numeric(df_geo["longitude"], errors="coerce")
df_geo.dropna(inplace=True)

# Also keep status for coloring analysis
df_status = df.loc[df_geo.index, "status"].copy()
def std(v):
    if v == "Closed": return "Closed"
    if v in ["In Progress","Assigned","Pending"]: return "In Progress"
    return "Open"
df_status = df_status.apply(std)

X_geo = df_geo.values
print(f"  Valid rows: {len(X_geo):,}")

# Sample
np.random.seed(42)
N   = min(15000, len(X_geo))
idx = np.random.choice(len(X_geo), N, replace=False)
Xs  = X_geo[idx]
ys  = df_status.iloc[idx].values

# ── EPS SENSITIVITY (geographic — degrees) ────────────────────
print("\nEps sensitivity (lat/lng degrees)...")
eps_vals = [0.005, 0.01, 0.02, 0.05, 0.1]
results  = []
for eps in eps_vals:
    db     = DBSCAN(eps=eps, min_samples=15, n_jobs=-1)
    labels = db.fit_predict(Xs)
    n_cls  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise= (labels == -1).sum()
    results.append({"eps": eps, "clusters": n_cls, "noise": n_noise})
    print(f"  eps={eps}  clusters={n_cls}  noise={n_noise}")

# ── FINAL MODEL ──────────────────────────────────────────────
BEST_EPS = 0.02
print(f"\nFinal DBSCAN (eps={BEST_EPS})...")
db_final = DBSCAN(eps=BEST_EPS, min_samples=15, n_jobs=-1)
labels   = db_final.fit_predict(Xs)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()
print(f"  Clusters : {n_clusters}")
print(f"  Noise    : {n_noise} ({n_noise/len(Xs)*100:.1f}%)")

# ── GEOGRAPHIC SCATTER PLOT ───────────────────────────────────
print("Saving: dbscan_raw_geographic.png")
COLORS = ["#2563eb","#16a34a","#f97316","#7c3aed","#0891b2",
          "#be185d","#d97706","#059669","#dc2626","#4f46e5",
          "#0d9488","#b45309","#9333ea","#0369a1","#15803d"]

plt.figure(figsize=(11, 8))
unique_labels = sorted(set(labels))
for lbl in unique_labels:
    mask = labels == lbl
    if lbl == -1:
        plt.scatter(Xs[mask,1], Xs[mask,0], s=3, alpha=0.15,
                    color="#94a3b8", label=f"Noise ({mask.sum()})", zorder=1)
    else:
        plt.scatter(Xs[mask,1], Xs[mask,0], s=8, alpha=0.5,
                    color=COLORS[lbl % len(COLORS)],
                    label=f"Cluster {lbl} ({mask.sum()})", zorder=2)

plt.title(f"DBSCAN on Raw Dataset — Geographic Complaint Clusters\n"
          f"NYC 311 | eps={BEST_EPS}° | {n_clusters} clusters | {n_noise} noise",
          fontsize=13)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(fontsize=8, markerscale=2, loc="upper right",
           bbox_to_anchor=(1.18, 1), borderaxespad=0)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dbscan_raw_geographic.png"), dpi=100, bbox_inches="tight")
plt.close()

# ── EPS CHART ────────────────────────────────────────────────
print("Saving: dbscan_raw_eps_sensitivity.png")
res_df = pd.DataFrame(results)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(res_df["eps"], res_df["clusters"], marker="o", color="#2563eb", linewidth=2.5)
axes[0].axvline(BEST_EPS, color="#ef4444", linestyle="--", linewidth=2)
axes[0].set_title("Eps vs Clusters (Raw Geographic)"); axes[0].grid(alpha=0.3)
axes[1].plot(res_df["eps"], res_df["noise"],    marker="s", color="#f97316", linewidth=2.5)
axes[1].axvline(BEST_EPS, color="#ef4444", linestyle="--", linewidth=2)
axes[1].set_title("Eps vs Noise Points");           axes[1].grid(alpha=0.3)
plt.suptitle("DBSCAN (Raw Dataset, Lat/Lng) — Eps Sensitivity", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dbscan_raw_eps_sensitivity.png"), dpi=100)
plt.close()

print(f"\nDBSCAN (Raw) complete. Clusters={n_clusters}, Noise={n_noise}")
