# ============================================================
# CIVICFIX - HIERARCHICAL CLUSTERING: AGGLOMERATIVE & DIVISIVE
# ============================================================
# Hierarchical clustering builds a tree (dendrogram) of
# nested clusters without needing to specify K in advance.
#
# TWO APPROACHES:
#
# AGGLOMERATIVE (Bottom-up):
#   - Start: each point is its own cluster
#   - Merge the two closest clusters step by step
#   - Stop when all points are in one cluster
#   - Linkage methods: ward, complete, average, single
#
# DIVISIVE (Top-down):
#   - Start: all points in one cluster
#   - Recursively split the largest cluster
#   - Implemented here using KMeans bisection
#
# For CivicFix, we cluster complaints to discover
# natural groupings by type, location, and agency.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster          import AgglomerativeClustering, KMeans
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import silhouette_score
from scipy.cluster.hierarchy  import dendrogram, linkage


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for Hierarchical Clustering.")
print("Shape:", df.shape)


# ============================================================
# 4. SELECT AND PREPARE FEATURES
# ============================================================

CLUSTER_COLS = [
    c for c in [
        "latitude", "longitude",
        "complaint_type", "agency",
        "borough", "council_district"
    ]
    if c in df.columns
]

df_cluster = df[CLUSTER_COLS].dropna().copy()

for col in CLUSTER_COLS:
    if df_cluster[col].dtype == "object":
        le = LabelEncoder()
        df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[CLUSTER_COLS])

# Use a small sample for dendrogram (hierarchical is memory-intensive)
DENDROGRAM_SAMPLE = 500
np.random.seed(42)
idx_dend = np.random.choice(len(X_scaled), DENDROGRAM_SAMPLE, replace=False)
X_dend   = X_scaled[idx_dend]

# Use a larger sample for clustering evaluation
CLUSTER_SAMPLE = 10000
if len(X_scaled) > CLUSTER_SAMPLE:
    idx_clust = np.random.choice(len(X_scaled), CLUSTER_SAMPLE, replace=False)
    X_cluster = X_scaled[idx_clust]
else:
    X_cluster = X_scaled

print(f"\nFeatures   : {CLUSTER_COLS}")
print(f"Total rows : {len(X_scaled):,}")
print(f"Dendrogram sample : {DENDROGRAM_SAMPLE}")
print(f"Clustering sample : {len(X_cluster):,}")


# ============================================================
# 5. DENDROGRAM — WARD LINKAGE
# ============================================================

print("\nGenerating: Dendrogram (Ward linkage)...")

linked = linkage(X_dend, method="ward")

plt.figure(figsize=(14, 6))
dendrogram(
    linked,
    truncate_mode  = "lastp",
    p              = 15,
    leaf_rotation  = 45,
    leaf_font_size = 10,
    color_threshold= 0
)
plt.title(
    f"Agglomerative Clustering — Dendrogram (Ward Linkage)\n"
    f"Sample: {DENDROGRAM_SAMPLE} CivicFix Complaints",
    fontsize=13
)
plt.xlabel("Sample Index", fontsize=11)
plt.ylabel("Euclidean Distance", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_dendrogram_ward.png"))
plt.close()
print("  Saved: hierarchical_dendrogram_ward.png")


# ============================================================
# 6. COMPARE LINKAGE METHODS
# ============================================================

print("\nComparing linkage methods (n_clusters=4)...")

linkage_methods = ["ward", "complete", "average", "single"]
linkage_scores  = {}

for method in linkage_methods:
    agg = AgglomerativeClustering(
        n_clusters = 4,
        linkage    = method
    )
    labels = agg.fit_predict(X_cluster)
    score  = silhouette_score(X_cluster, labels, sample_size=3000)
    linkage_scores[method] = score
    print(f"  Linkage={method:<10}: Silhouette Score = {score:.4f}")

best_linkage = max(linkage_scores, key=linkage_scores.get)
print(f"\n  Best linkage method: {best_linkage} "
      f"(score={linkage_scores[best_linkage]:.4f})")


# ============================================================
# 7. FIND OPTIMAL K FOR AGGLOMERATIVE CLUSTERING
# ============================================================

print("\nFinding optimal K for Agglomerative Clustering...")

K_RANGE  = range(2, 9)
agg_sil  = []

for k in K_RANGE:
    agg    = AgglomerativeClustering(n_clusters=k, linkage=best_linkage)
    labels = agg.fit_predict(X_cluster)
    score  = silhouette_score(X_cluster, labels, sample_size=3000)
    agg_sil.append(score)
    print(f"  K={k}: Silhouette = {score:.4f}")

best_k = list(K_RANGE)[np.argmax(agg_sil)]
print(f"\n  Best K: {best_k} (score={max(agg_sil):.4f})")


# ============================================================
# 8. AGGLOMERATIVE SILHOUETTE SCORE CHART
# ============================================================

print("\nGenerating: Silhouette Score chart...")

plt.figure(figsize=(9, 4))
plt.plot(
    list(K_RANGE), agg_sil,
    marker="o", color="#2878e8", linewidth=2
)
plt.axvline(x=best_k, color="red", linestyle="--", linewidth=1.5,
            label=f"Best K={best_k}")
plt.title(
    f"Agglomerative Clustering ({best_linkage} linkage) — Silhouette Scores",
    fontsize=13
)
plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Silhouette Score", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_silhouette_scores.png"))
plt.close()
print("  Saved: hierarchical_silhouette_scores.png")


# ============================================================
# 9. FINAL AGGLOMERATIVE MODEL WITH BEST K
# ============================================================

print(f"\nTraining final Agglomerative Clustering (K={best_k})...")

final_agg    = AgglomerativeClustering(n_clusters=best_k, linkage=best_linkage)
agg_labels   = final_agg.fit_predict(X_cluster)
agg_sil_best = silhouette_score(X_cluster, agg_labels)

print(f"  Final Silhouette Score: {agg_sil_best:.4f}")


# ============================================================
# 10. DIVISIVE CLUSTERING (BISECTING K-MEANS SIMULATION)
# ============================================================
# Simulates divisive clustering by recursively bisecting
# the largest cluster using K-Means with K=2

print("\nDivisive Clustering (Bisecting K-Means)...")

def bisecting_kmeans(X, n_clusters, random_state=42):
    """
    Recursive bisection of the largest cluster.
    Returns cluster labels for n_clusters clusters.
    """
    labels = np.zeros(len(X), dtype=int)

    for _ in range(n_clusters - 1):
        # Find the largest cluster
        unique, counts = np.unique(labels, return_counts=True)
        largest_cluster = unique[np.argmax(counts)]
        mask = labels == largest_cluster

        # Bisect it with K-Means
        km = KMeans(n_clusters=2, random_state=random_state, n_init=5)
        sub_labels = km.fit_predict(X[mask])

        # Assign new label to one of the two sub-clusters
        new_label = labels.max() + 1
        idx        = np.where(mask)[0]
        labels[idx[sub_labels == 1]] = new_label

    return labels


div_labels   = bisecting_kmeans(X_cluster, n_clusters=best_k)
div_sil      = silhouette_score(X_cluster, div_labels)

print(f"  Divisive Clustering (K={best_k}): Silhouette = {div_sil:.4f}")


# ============================================================
# 11. COMPARISON CHART: AGGLOMERATIVE vs DIVISIVE
# ============================================================

print("Generating: Agglomerative vs Divisive comparison chart...")

from sklearn.decomposition import PCA

pca   = PCA(n_components=2, random_state=42)
X_2d  = pca.fit_transform(X_cluster)

COLORS = ["#2878e8", "#f99a18", "#4aa84c", "#e84848",
          "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, labels, title, sil in zip(
    axes,
    [agg_labels, div_labels],
    [f"Agglomerative ({best_linkage}, K={best_k})",
     f"Divisive (Bisecting K-Means, K={best_k})"],
    [agg_sil_best, div_sil]
):
    for c in range(best_k):
        mask = labels == c
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=8, alpha=0.4,
            color=COLORS[c % len(COLORS)],
            label=f"Cluster {c}"
        )
    ax.set_title(f"{title}\nSilhouette = {sil:.4f}", fontsize=11)
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.legend(markerscale=2, fontsize=8)

plt.suptitle(
    "CivicFix — Agglomerative vs Divisive Hierarchical Clustering",
    fontsize=14
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_agg_vs_div.png"))
plt.close()
print("  Saved: hierarchical_agg_vs_div.png")


# ============================================================
# 12. LINKAGE METHOD COMPARISON CHART
# ============================================================

print("Generating: Linkage method comparison chart...")

plt.figure(figsize=(8, 4))
bars = plt.bar(
    list(linkage_scores.keys()),
    list(linkage_scores.values()),
    color=["#4aa84c" if m == best_linkage else "#2878e8"
           for m in linkage_scores.keys()],
    edgecolor="black"
)
for bar, score in zip(bars, linkage_scores.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f"{score:.4f}",
        ha="center", va="bottom", fontsize=9
    )
plt.title("Agglomerative — Linkage Method Comparison (K=4)", fontsize=13)
plt.xlabel("Linkage Method", fontsize=11)
plt.ylabel("Silhouette Score", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_linkage_comparison.png"))
plt.close()
print("  Saved: hierarchical_linkage_comparison.png")


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("  HIERARCHICAL CLUSTERING — SUMMARY")
print("=" * 60)
print(f"  Best Linkage              : {best_linkage}")
print(f"  Best K                    : {best_k}")
print(f"  Agglomerative Silhouette  : {agg_sil_best:.4f}")
print(f"  Divisive Silhouette       : {div_sil:.4f}")
print("=" * 60)

results_df = pd.DataFrame([{
    "Method"             : "Agglomerative",
    "Linkage"            : best_linkage,
    "Best_K"             : best_k,
    "Silhouette_Score"   : round(agg_sil_best, 4)
}, {
    "Method"             : "Divisive (Bisecting KMeans)",
    "Linkage"            : "N/A",
    "Best_K"             : best_k,
    "Silhouette_Score"   : round(div_sil, 4)
}])

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "hierarchical_results.csv"),
    index=False
)
print("Results CSV saved.")
print("Hierarchical clustering complete.")
