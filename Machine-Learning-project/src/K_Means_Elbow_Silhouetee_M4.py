# ============================================================
# CIVICFIX - K-MEANS CLUSTERING WITH ELBOW & SILHOUETTE (M4)
# ============================================================
# K-Means is an unsupervised learning algorithm that groups
# data points into K clusters based on similarity.
#
# How it works:
#   1. Randomly initialise K centroids
#   2. Assign each point to its nearest centroid
#   3. Recompute centroids as mean of their cluster
#   4. Repeat until centroids stop moving
#
# For CivicFix, we cluster complaints by:
#   - Location (latitude, longitude)
#   - Complaint type and borough
#
# Finding optimal K:
#   ELBOW METHOD   - Plot inertia vs K, find the "elbow"
#   SILHOUETTE SCORE - Measures cluster cohesion vs separation
#                      Best K = highest silhouette score
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.cluster     import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics     import silhouette_score


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

MODEL_DIR  = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"
OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\model_results"

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for K-Means Clustering.")
print("Shape:", df.shape)


# ============================================================
# 4. SELECT FEATURES FOR CLUSTERING
# ============================================================
# We use geographical + categorical features to cluster
# similar complaints together

CLUSTER_COLS = [
    c for c in [
        "latitude", "longitude",
        "complaint_type", "agency",
        "borough", "council_district"
    ]
    if c in df.columns
]

df_cluster = df[CLUSTER_COLS].dropna().copy()

print(f"\nClustering features: {CLUSTER_COLS}")
print(f"Rows used          : {len(df_cluster):,}")


# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

for col in CLUSTER_COLS:
    if df_cluster[col].dtype == "object":
        le = LabelEncoder()
        df_cluster[col] = le.fit_transform(df_cluster[col].astype(str))


# ============================================================
# 6. SCALE FEATURES
# ============================================================

scaler     = StandardScaler()
X_scaled   = scaler.fit_transform(df_cluster[CLUSTER_COLS])

# Use a sample for speed if dataset is large
MAX_SAMPLE = 20000
if len(X_scaled) > MAX_SAMPLE:
    np.random.seed(42)
    idx       = np.random.choice(len(X_scaled), MAX_SAMPLE, replace=False)
    X_scaled  = X_scaled[idx]
    print(f"Using {MAX_SAMPLE:,} random samples for clustering speed.")


# ============================================================
# 7. ELBOW METHOD — INERTIA vs K
# ============================================================

print("\nRunning Elbow Method (K = 2 to 10)...")

K_RANGE  = range(2, 11)
inertias = []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  K={k}: Inertia = {km.inertia_:.2f}")


# ============================================================
# 8. SILHOUETTE SCORE — CLUSTER QUALITY vs K
# ============================================================

print("\nCalculating Silhouette Scores...")

silhouette_scores = []

for k in K_RANGE:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels, sample_size=5000)
    silhouette_scores.append(score)
    print(f"  K={k}: Silhouette Score = {score:.4f}")

best_k = list(K_RANGE)[np.argmax(silhouette_scores)]
print(f"\n  Best K by Silhouette Score: K={best_k}")


# ============================================================
# 9. ELBOW CURVE CHART
# ============================================================

print("\nGenerating: Elbow Curve chart...")

plt.figure(figsize=(9, 5))
plt.plot(
    list(K_RANGE), inertias,
    marker="o", color="#2878e8", linewidth=2
)
plt.axvline(x=best_k, color="red", linestyle="--",
            linewidth=1.5, label=f"Best K={best_k}")
plt.title("K-Means — Elbow Method (Inertia vs K)", fontsize=13)
plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow_curve.png"))
plt.close()
print("  Saved: kmeans_elbow_curve.png")


# ============================================================
# 10. SILHOUETTE SCORE CHART
# ============================================================

print("Generating: Silhouette Score chart...")

plt.figure(figsize=(9, 5))
bars = plt.bar(
    list(K_RANGE), silhouette_scores,
    color=["#4aa84c" if k == best_k else "#2878e8" for k in K_RANGE],
    edgecolor="black"
)
plt.axhline(
    y=max(silhouette_scores),
    color="red", linestyle="--", linewidth=1.5,
    label=f"Best Score = {max(silhouette_scores):.4f} at K={best_k}"
)

for bar, score in zip(bars, silhouette_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.002,
        f"{score:.3f}",
        ha="center", va="bottom", fontsize=9
    )

plt.title("K-Means — Silhouette Scores per K", fontsize=13)
plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Silhouette Score", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_silhouette_scores.png"))
plt.close()
print("  Saved: kmeans_silhouette_scores.png")


# ============================================================
# 11. TRAIN FINAL K-MEANS WITH BEST K
# ============================================================

print(f"\nTraining final K-Means with K={best_k}...")

final_km = KMeans(
    n_clusters = best_k,
    random_state = 42,
    n_init = 10
)
final_labels = final_km.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, final_labels)
print(f"  Final Silhouette Score (K={best_k}): {final_silhouette:.4f}")


# ============================================================
# 12. CLUSTER VISUALIZATION (PCA to 2D)
# ============================================================

print("Generating: Cluster visualization (2D)...")

from sklearn.decomposition import PCA

pca        = PCA(n_components=2, random_state=42)
X_pca      = pca.fit_transform(X_scaled)
var_ratio  = pca.explained_variance_ratio_

COLORS = ["#2878e8", "#f99a18", "#4aa84c", "#e84848",
          "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71", "#3498db"]

plt.figure(figsize=(10, 7))
for c in range(best_k):
    mask = final_labels == c
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        s=10, alpha=0.4,
        color=COLORS[c % len(COLORS)],
        label=f"Cluster {c}"
    )

# Plot centroids projected to PCA space
centroids_pca = pca.transform(final_km.cluster_centers_)
plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1],
    s=200, marker="X", c="black", label="Centroids", zorder=5
)

plt.title(
    f"K-Means Clustering (K={best_k})\n"
    f"PCA Variance Explained: {var_ratio.sum()*100:.1f}%",
    fontsize=13
)
plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=11)
plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=11)
plt.legend(markerscale=2, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_cluster_visualization.png"))
plt.close()
print("  Saved: kmeans_cluster_visualization.png")


# ============================================================
# 13. CLUSTER SIZE DISTRIBUTION
# ============================================================

print("Generating: Cluster size distribution...")

unique, counts = np.unique(final_labels, return_counts=True)

plt.figure(figsize=(8, 4))
plt.bar(
    [f"Cluster {c}" for c in unique],
    counts,
    color=[COLORS[c % len(COLORS)] for c in unique],
    edgecolor="black"
)
plt.title(f"K-Means — Cluster Sizes (K={best_k})", fontsize=13)
plt.xlabel("Cluster", fontsize=11)
plt.ylabel("Number of Complaints", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_cluster_sizes.png"))
plt.close()
print("  Saved: kmeans_cluster_sizes.png")


# ============================================================
# 14. SAVE MODEL AND RESULTS
# ============================================================

results_df = pd.DataFrame({
    "K"               : list(K_RANGE),
    "Inertia"         : inertias,
    "Silhouette_Score": silhouette_scores
})
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "kmeans_results.csv"),
    index=False
)

MODEL_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": final_km, "scaler": scaler, "best_k": best_k}, f)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"K-Means best K={best_k} | Silhouette Score={final_silhouette:.4f}")
