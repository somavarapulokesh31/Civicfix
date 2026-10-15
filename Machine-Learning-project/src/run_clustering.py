# ============================================================
# CIVICFIX - K-MEANS + HIERARCHICAL CLUSTERING (M4)
# ============================================================
import os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.cluster          import KMeans, AgglomerativeClustering
from sklearn.metrics          import silhouette_score
from sklearn.decomposition    import PCA
from scipy.cluster.hierarchy  import dendrogram, linkage

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

CLUSTER_FEATURES = ["complaint_type", "agency", "borough",
                    "location_type", "council_district", "latitude", "longitude"]
CLUSTER_FEATURES = [f for f in CLUSTER_FEATURES if f in df.columns]

df_c = df[CLUSTER_FEATURES].copy()

# Fill missing
for col in CLUSTER_FEATURES:
    if df_c[col].dtype == "object" or str(df_c[col].dtype) in ("string","StringDtype"):
        df_c[col] = df_c[col].fillna("Unknown")
    else:
        df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

# Encode categoricals
for col in CLUSTER_FEATURES:
    if df_c[col].dtype == "object" or str(df_c[col].dtype) in ("string","StringDtype"):
        le = LabelEncoder()
        df_c[col] = le.fit_transform(df_c[col].astype(str))

# Impute + scale
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(df_c.values.astype(float))
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sample for speed
np.random.seed(42)
N = min(15000, len(X_scaled))
idx = np.random.choice(len(X_scaled), N, replace=False)
Xs = X_scaled[idx]

print(f"  Rows for clustering: {N:,}  |  Features: {CLUSTER_FEATURES}")

# ── 2. K-MEANS: ELBOW + SILHOUETTE ──────────────────────────
print("\nRunning K-Means (K=2..8)...")
K_RANGE   = range(2, 9)
inertias  = []
sil_scores= []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    s = silhouette_score(Xs, labels, sample_size=5000, random_state=42)
    sil_scores.append(s)
    print(f"  K={k}  Inertia={km.inertia_:.0f}  Silhouette={s:.4f}")

best_k = list(K_RANGE)[np.argmax(sil_scores)]
print(f"  Best K = {best_k}")

# ── 3. ELBOW CURVE ───────────────────────────────────────────
print("Saving: kmeans_elbow_curve.png")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inertias, marker="o", color="#2563eb", linewidth=2.5, markersize=7)
axes[0].axvline(best_k, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Best K={best_k}")
axes[0].set_title("K-Means — Elbow Curve (Inertia vs K)", fontsize=13)
axes[0].set_xlabel("Number of Clusters (K)", fontsize=11)
axes[0].set_ylabel("Inertia", fontsize=11)
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].bar(list(K_RANGE), sil_scores,
            color=["#16a34a" if k==best_k else "#2563eb" for k in K_RANGE],
            edgecolor="white", width=0.6)
axes[1].axhline(max(sil_scores), color="#ef4444", linestyle="--", linewidth=1.5,
                label=f"Best={max(sil_scores):.3f} at K={best_k}")
axes[1].set_title("K-Means — Silhouette Scores", fontsize=13)
axes[1].set_xlabel("Number of Clusters (K)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].legend(); axes[1].grid(alpha=.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_elbow_silhouette.png"), dpi=100)
plt.close()

# ── 4. FINAL K-MEANS ─────────────────────────────────────────
print(f"Training final K-Means K={best_k}...")
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(Xs)

# ── 5. PCA CLUSTER VISUALIZATION ────────────────────────────
print("Saving: kmeans_cluster_visualization.png")
pca = PCA(n_components=2, random_state=42)
X2d = pca.fit_transform(Xs)

COLORS = ["#16a34a","#2563eb","#f97316","#ef4444","#7c3aed","#0891b2","#d97706","#be185d"]
plt.figure(figsize=(10, 7))
for c in range(best_k):
    mask = km_labels == c
    plt.scatter(X2d[mask,0], X2d[mask,1], s=8, alpha=0.5,
                color=COLORS[c%len(COLORS)], label=f"Cluster {c}")
cx = pca.transform(km_final.cluster_centers_)
plt.scatter(cx[:,0], cx[:,1], s=200, marker="X", c="black",
            zorder=5, label="Centroids")
plt.title(f"K-Means Clustering (K={best_k}) — PCA 2D View\n"
          f"Variance Explained: {pca.explained_variance_ratio_.sum()*100:.1f}%", fontsize=13)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend(markerscale=2, fontsize=9, loc="best")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_cluster_visualization.png"), dpi=100)
plt.close()

# ── 6. CLUSTER SIZE BAR ───────────────────────────────────────
print("Saving: kmeans_cluster_sizes.png")
uniq, cnts = np.unique(km_labels, return_counts=True)
plt.figure(figsize=(8, 4))
plt.bar([f"Cluster {c}" for c in uniq], cnts,
        color=[COLORS[c%len(COLORS)] for c in uniq], edgecolor="white")
for i,(u,v) in enumerate(zip(uniq,cnts)):
    plt.text(i, v+30, f"{v:,}", ha="center", fontsize=10)
plt.title(f"K-Means — Cluster Sizes (K={best_k})", fontsize=13)
plt.ylabel("Number of Complaints"); plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "kmeans_cluster_sizes.png"), dpi=100)
plt.close()

# Save model
with open(os.path.join(MODEL_DIR,"kmeans.pkl"),"wb") as f:
    pickle.dump({"model":km_final,"scaler":scaler,"imputer":imputer,"best_k":best_k}, f)
print(f"  K-Means saved. Best K={best_k}")

# ── 7. HIERARCHICAL: DENDROGRAM ──────────────────────────────
print("\nHierarchical Clustering...")
DEND_N = 300
idx_d  = np.random.choice(len(Xs), DEND_N, replace=False)
Xd     = Xs[idx_d]

print("Saving: hierarchical_dendrogram.png")
linked = linkage(Xd, method="ward")
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode="lastp", p=20,
           leaf_rotation=45, leaf_font_size=9, color_threshold=0)
plt.title(f"Agglomerative Clustering — Ward Linkage Dendrogram\n"
          f"(Sample: {DEND_N} complaints)", fontsize=13)
plt.xlabel("Sample Index", fontsize=11)
plt.ylabel("Euclidean Distance", fontsize=11)
plt.grid(alpha=.2, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hierarchical_dendrogram.png"), dpi=100)
plt.close()

# ── 8. HIERARCHICAL: LINKAGE COMPARISON ─────────────────────
print("Saving: hierarchical_linkage_comparison.png")
AGG_K = best_k
linkage_methods = ["ward", "complete", "average", "single"]
link_sil = {}
for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=AGG_K, linkage=method)
    lbl = agg.fit_predict(Xs)
    s   = silhouette_score(Xs, lbl, sample_size=5000, random_state=42)
    link_sil[method] = round(s, 4)
    print(f"  {method:<10} Silhouette={s:.4f}")

best_link = max(link_sil, key=link_sil.get)
plt.figure(figsize=(8, 4))
bars = plt.bar(list(link_sil.keys()), list(link_sil.values()),
               color=["#16a34a" if m==best_link else "#2563eb" for m in link_sil],
               edgecolor="white")
for bar, val in zip(bars, link_sil.values()):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
             f"{val:.4f}", ha="center", fontsize=10)
plt.title(f"Hierarchical — Linkage Method Comparison (K={AGG_K})", fontsize=13)
plt.xlabel("Linkage Method"); plt.ylabel("Silhouette Score")
plt.grid(alpha=.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hierarchical_linkage_comparison.png"), dpi=100)
plt.close()

# ── 9. HIERARCHICAL PCA VISUALIZATION ────────────────────────
print("Saving: hierarchical_cluster_visualization.png")
agg_final = AgglomerativeClustering(n_clusters=AGG_K, linkage=best_link)
agg_labels = agg_final.fit_predict(Xs)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, labels, title in zip(
    axes,
    [km_labels, agg_labels],
    [f"K-Means (K={best_k})", f"Agglomerative ({best_link}, K={AGG_K})"]
):
    for c in range(best_k):
        mask = labels == c
        ax.scatter(X2d[mask,0], X2d[mask,1], s=8, alpha=0.5,
                   color=COLORS[c%len(COLORS)], label=f"Cluster {c}")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(alpha=.3)

plt.suptitle("K-Means vs Agglomerative Clustering — PCA 2D", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "hierarchical_cluster_visualization.png"), dpi=100)
plt.close()

# Save hierarchical model info
with open(os.path.join(MODEL_DIR,"hierarchical.pkl"),"wb") as f:
    pickle.dump({"linkage_scores":link_sil,"best_linkage":best_link,"best_k":AGG_K}, f)
print(f"  Hierarchical saved. Best linkage={best_link}")

print("\n" + "="*55)
print("  CLUSTERING COMPLETE")
print(f"  K-Means best K       : {best_k}")
print(f"  Agglomerative linkage: {best_link}")
print("="*55)
