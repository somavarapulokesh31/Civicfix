# ============================================================
# CIVICFIX - PRINCIPAL COMPONENT ANALYSIS (MODULE 4)
# ============================================================
# PCA is a dimensionality reduction technique that transforms
# the original features into a smaller set of new features
# called Principal Components (PCs).
#
# Each PC is a linear combination of original features.
# PC1 captures the most variance, PC2 the second most, etc.
#
# For CivicFix (8 features → reduce to 2 or 3):
#   - Visualise high-dimensional data in 2D/3D
#   - Understand which features explain most variance
#   - Remove noise and redundant information
#   - Scree plot shows how many PCs to keep
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute         import SimpleImputer
from sklearn.decomposition  import PCA

BASE = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project"
DS   = os.path.join(BASE, "dataset", "erm2-nwe9.csv")
OUT  = os.path.join(BASE, "outputs", "model_results")
os.makedirs(OUT, exist_ok=True)

# ── LOAD DATA ────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DS, low_memory=False)
df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], inplace=True)

def std_status(v):
    if v == "Closed":      return "Closed"
    if v in ["In Progress","Assigned","Pending"]: return "In Progress"
    return "Open"
df["status_std"] = df["status"].apply(std_status)

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

imp    = SimpleImputer(strategy="median")
scaler = StandardScaler()
X      = imp.fit_transform(df_f.values.astype(float))
X      = scaler.fit_transform(X)
y_lbl  = df.loc[df_f.index[:len(X)], "status_std"].values

# Sample
np.random.seed(42)
N   = min(20000, len(X))
idx = np.random.choice(len(X), N, replace=False)
Xs  = X[idx]
ys  = y_lbl[idx] if len(y_lbl) >= len(idx) else np.array(["Unknown"]*N)
print(f"  Rows: {N:,}  |  Features: {len(FEATURES)}")

# ── FIT FULL PCA ─────────────────────────────────────────────
print("\nFitting PCA on all components...")
pca_full = PCA(random_state=42)
pca_full.fit(Xs)
var_ratio = pca_full.explained_variance_ratio_
cum_var   = np.cumsum(var_ratio)

print(f"  Variance explained per component:")
for i,(v,c) in enumerate(zip(var_ratio, cum_var)):
    print(f"    PC{i+1}: {v*100:.2f}%  (cumulative: {c*100:.2f}%)")

# n_components for 95% variance
n95 = np.argmax(cum_var >= 0.95) + 1
n90 = np.argmax(cum_var >= 0.90) + 1
print(f"\n  PCs needed for 90% variance: {n90}")
print(f"  PCs needed for 95% variance: {n95}")

# ── SCREE PLOT ───────────────────────────────────────────────
print("Saving: pca_scree_plot.png")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Individual variance
axes[0].bar(range(1, len(var_ratio)+1), var_ratio*100,
            color="#2563eb", edgecolor="white", alpha=0.8)
axes[0].set_title("PCA — Scree Plot (Individual Variance)", fontsize=13)
axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance Explained (%)")
axes[0].grid(axis="y", alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cum_var)+1), cum_var*100,
             marker="o", color="#16a34a", linewidth=2.5, markersize=8)
axes[1].axhline(90, color="#f97316", linestyle="--", linewidth=1.5, label="90%")
axes[1].axhline(95, color="#ef4444", linestyle="--", linewidth=1.5, label="95%")
axes[1].axvline(n90, color="#f97316", linestyle=":", linewidth=1.5)
axes[1].axvline(n95, color="#ef4444", linestyle=":", linewidth=1.5)
axes[1].set_title("PCA — Cumulative Variance Explained", fontsize=13)
axes[1].set_xlabel("Number of Components"); axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend(fontsize=11); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "pca_scree_plot.png"), dpi=100)
plt.close()

# ── PCA 2D SCATTER ───────────────────────────────────────────
print("Saving: pca_2d_scatter.png")
pca2 = PCA(n_components=2, random_state=42)
X2d  = pca2.fit_transform(Xs)

STATUS_COLORS = {"Closed":"#16a34a", "In Progress":"#2563eb", "Open":"#f97316", "Unknown":"#94a3b8"}
plt.figure(figsize=(10, 7))
for status, color in STATUS_COLORS.items():
    mask = ys == status
    if mask.sum() > 0:
        plt.scatter(X2d[mask,0], X2d[mask,1], s=8, alpha=0.4,
                    color=color, label=f"{status} ({mask.sum():,})")

plt.title(f"PCA — 2D Projection of CivicFix Features\n"
          f"PC1={pca2.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca2.explained_variance_ratio_[1]*100:.1f}%  "
          f"Total={pca2.explained_variance_ratio_.sum()*100:.1f}%", fontsize=12)
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(fontsize=11, markerscale=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pca_2d_scatter.png"), dpi=100)
plt.close()

# ── LOADING PLOT (feature contributions) ─────────────────────
print("Saving: pca_feature_loadings.png")
n_pcs_show = min(3, len(FEATURES), pca_full.n_components_)
loadings = pd.DataFrame(
    pca_full.components_[:n_pcs_show].T,
    index=FEATURES[:pca_full.components_.shape[1]],
    columns=[f"PC{i+1}" for i in range(n_pcs_show)]
)
fig, axes = plt.subplots(1, n_pcs_show, figsize=(5*n_pcs_show, 5), sharey=True)
if n_pcs_show == 1:
    axes = [axes]
colors_list = ["#2563eb","#16a34a","#f97316"]
for ax, col, color in zip(axes, loadings.columns, colors_list):
    vals = loadings[col].sort_values()
    ax.barh(vals.index, vals.values, color=color, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    pc_idx = int(col[2])-1
    variance_pct = var_ratio[pc_idx]*100 if pc_idx < len(var_ratio) else 0
    ax.set_title(f"{col} Loadings\n({variance_pct:.1f}% variance)", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
plt.suptitle("PCA — Feature Contributions to Principal Components", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pca_feature_loadings.png"), dpi=100)
plt.close()

print(f"\nPCA complete.")
print(f"  PCs for 90% variance: {n90}")
print(f"  PCs for 95% variance: {n95}")
