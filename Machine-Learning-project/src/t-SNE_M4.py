# ============================================================
# CIVICFIX - t-SNE VISUALIZATION (MODULE 4)
# ============================================================
# t-SNE (t-distributed Stochastic Neighbour Embedding) is a
# non-linear dimensionality reduction technique mainly used
# for visualising high-dimensional data in 2D or 3D.
#
# How it works:
#   1. Measures similarity between points in high dimensions
#      using a Gaussian probability distribution
#   2. Builds a similar probability distribution in 2D
#      using a Student t-distribution (heavier tails)
#   3. Minimises KL divergence between the two distributions
#
# Key parameters:
#   perplexity   - balance between local and global structure
#                  (typically 5-50, default 30)
#   n_iter       - number of optimisation iterations
#   learning_rate- step size for gradient descent
#
# Unlike PCA, t-SNE preserves LOCAL neighbourhood structure
# — nearby points in high dimensions stay nearby in 2D.
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
from sklearn.manifold       import TSNE

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
y_lbl  = df["status_std"].values[:len(X)]

# t-SNE is slow — use small sample
np.random.seed(42)
N   = min(3000, len(X))   # t-SNE O(N²) — keep small
idx = np.random.choice(len(X), N, replace=False)
Xs  = X[idx]
ys  = y_lbl[idx]
print(f"  Sample: {N:,} rows  |  Features: {len(FEATURES)}")

# ── PRE-REDUCE WITH PCA ──────────────────────────────────────
# Recommended: reduce to 50 dims with PCA first, then t-SNE
print("\nPre-reducing with PCA...")
n_pre = min(8, len(FEATURES), Xs.shape[1])
pca_pre = PCA(n_components=n_pre, random_state=42)
Xp = pca_pre.fit_transform(Xs)
print(f"  Variance retained: {pca_pre.explained_variance_ratio_.sum()*100:.1f}%")

# ── t-SNE with different perplexities ────────────────────────
PERPLEXITIES = [10, 30, 50]
STATUS_COLORS = {"Closed":"#16a34a", "In Progress":"#2563eb", "Open":"#f97316"}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
print("\nRunning t-SNE with different perplexities...")

for ax, perp in zip(axes, PERPLEXITIES):
    print(f"  Perplexity={perp}...")
    tsne = TSNE(
        n_components  = 2,
        perplexity    = perp,
        max_iter      = 1000,
        learning_rate = "auto",
        init          = "pca",
        random_state  = 42
    )
    X2d = tsne.fit_transform(Xp)

    for status, color in STATUS_COLORS.items():
        mask = ys == status
        if mask.sum() > 0:
            ax.scatter(X2d[mask,0], X2d[mask,1], s=8, alpha=0.5,
                       color=color, label=f"{status} ({mask.sum()})")

    ax.set_title(f"t-SNE  (perplexity={perp})", fontsize=12)
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.legend(fontsize=8, markerscale=2)
    ax.grid(alpha=0.3)

plt.suptitle(f"t-SNE Visualisation of CivicFix Complaints\n"
             f"(sample={N:,}, coloured by complaint status)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_perplexity_comparison.png"), dpi=100)
plt.close()
print("  Saved: tsne_perplexity_comparison.png")

# ── BEST t-SNE (perplexity=30) coloured by Agency ────────────
print("Running t-SNE coloured by agency...")
tsne_best = TSNE(n_components=2, perplexity=30, max_iter=1000,
                 learning_rate="auto", init="pca", random_state=42)
X2d_best = tsne_best.fit_transform(Xp)

# Get agency for sample
agency_col = df["agency"].values[:len(X)]
agencies_s = agency_col[idx]
top_agencies = pd.Series(agencies_s).value_counts().head(6).index.tolist()

AG_COLORS = ["#2563eb","#16a34a","#f97316","#7c3aed","#ef4444","#0891b2"]
plt.figure(figsize=(10, 7))
for ag, clr in zip(top_agencies, AG_COLORS):
    mask = agencies_s == ag
    if mask.sum() > 0:
        plt.scatter(X2d_best[mask,0], X2d_best[mask,1],
                    s=8, alpha=0.5, color=clr, label=f"{ag} ({mask.sum()})")
other = ~np.isin(agencies_s, top_agencies)
if other.sum() > 0:
    plt.scatter(X2d_best[other,0], X2d_best[other,1],
                s=5, alpha=0.2, color="#94a3b8", label=f"Other ({other.sum()})")

plt.title(f"t-SNE — CivicFix Complaints by Agency (perplexity=30)\n"
          f"sample={N:,}", fontsize=13)
plt.xlabel("t-SNE Dim 1"); plt.ylabel("t-SNE Dim 2")
plt.legend(fontsize=9, markerscale=2)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "tsne_by_agency.png"), dpi=100)
plt.close()
print("  Saved: tsne_by_agency.png")

print(f"\nt-SNE complete.")
