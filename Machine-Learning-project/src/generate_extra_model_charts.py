"""Create the supplementary per-model plots used by the Visualizations page."""
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "model_results"
MODELS = ROOT / "models"
FEATURES = ["complaint_type", "agency", "borough", "location_type",
            "open_data_channel_type", "council_district", "latitude", "longitude"]
COLORS = ["#2563eb", "#16a34a", "#f97316", "#7c3aed", "#0891b2", "#be185d", "#d97706", "#059669"]


def save_metric_and_importance_charts():
    results = pd.read_csv(OUT / "all_model_results.csv")
    feature_names = FEATURES
    for _, row in results.iterrows():
        name = str(row["Model"])
        slug = name.lower().replace(" ", "_")
        scores = [float(row[col]) for col in ["Accuracy", "Precision", "Recall", "F1_Score"]]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(["Accuracy", "Precision", "Recall", "F1-Score"], scores,
                       color=["#16a34a", "#2563eb", "#f97316", "#7c3aed"])
        ax.set_xlim(0, 100); ax.set_xlabel("Score (%)"); ax.set_title(f"{name} — Test Metrics")
        ax.grid(axis="x", alpha=.2)
        for bar, value in zip(bars, scores):
            ax.text(value + 1, bar.get_y() + bar.get_height()/2, f"{value:.2f}%", va="center", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT / f"{slug}_metrics.png", dpi=120); plt.close(fig)

        try:
            with open(MODELS / f"{slug}.pkl", "rb") as f:
                model = pickle.load(f)
            values = getattr(model, "feature_importances_", None)
            if values is None and hasattr(model, "coef_"):
                values = np.abs(model.coef_).mean(axis=0)
            if values is None:
                continue
            names = feature_names[:len(values)]
            order = np.argsort(values)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(np.asarray(names)[order], np.asarray(values)[order], color="#16a34a")
            ax.set_title(f"{name} — Feature Importance")
            ax.set_xlabel("Importance")
            ax.grid(axis="x", alpha=.2)
            fig.tight_layout(); fig.savefig(OUT / f"{slug}_feature_importance.png", dpi=120); plt.close(fig)
        except Exception as exc:
            print(f"Skipped feature importance for {name}: {exc}")

    # Separate views of the same saved classifier evaluation support the
    # cross-validation/model-comparison section without inventing fold scores.
    for column, title, filename, color in [
        ("Accuracy", "Classifier Accuracy Comparison", "classifier_accuracy_comparison.png", "#2563eb"),
        ("F1_Score", "Classifier F1-Score Comparison", "classifier_f1_comparison.png", "#7c3aed"),
    ]:
        ordered = results.sort_values(column)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(ordered["Model"], ordered[column], color=color)
        ax.set_xlim(0, 100); ax.set_xlabel("Score (%)"); ax.set_title(title); ax.grid(axis="x", alpha=.2)
        for y, value in enumerate(ordered[column]):
            ax.text(value + 1, y, f"{value:.2f}%", va="center", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=120); plt.close(fig)


def save_dbscan_cluster_sizes():
    raw_path = ROOT / "dataset" / "erm2-nwe9.csv"
    raw = pd.read_csv(raw_path, low_memory=False)
    geo = raw[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    rng = np.random.RandomState(42)
    ids = rng.choice(len(geo), min(15000, len(geo)), replace=False)
    labels = DBSCAN(eps=.02, min_samples=15, n_jobs=-1).fit_predict(geo.values[ids])
    _save_cluster_counts(labels, "dbscan_raw_cluster_sizes.png", "DBSCAN Raw — Cluster Sizes")

    path = ROOT / "dataset" / "final_preprocess_M2.csv"
    data = pd.read_csv(path, low_memory=False).drop(columns=["status"], errors="ignore")
    for col in data:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = LabelEncoder().fit_transform(data[col].fillna("Unknown").astype(str))
    matrix = SimpleImputer(strategy="median").fit_transform(data.astype(float))
    matrix = StandardScaler().fit_transform(matrix)
    ids = np.random.RandomState(42).choice(len(matrix), min(10000, len(matrix)), replace=False)
    labels = DBSCAN(eps=.8, min_samples=10, n_jobs=-1).fit_predict(matrix[ids])
    _save_cluster_counts(labels, "dbscan_preprocessed_cluster_sizes.png", "DBSCAN Pre-Processed — Cluster Sizes")


def _save_cluster_counts(labels, filename, title):
    values, counts = np.unique(labels, return_counts=True)
    names = ["Noise" if value == -1 else f"Cluster {value}" for value in values]
    colors = ["#94a3b8" if value == -1 else COLORS[int(value) % len(COLORS)] for value in values]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(names, counts, color=colors)
    ax.set_title(title); ax.set_ylabel("Complaints"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.2)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / filename, dpi=120); plt.close(fig)


def save_tsne_status_plot():
    df = pd.read_csv(ROOT / "dataset" / "erm2-nwe9.csv", low_memory=False)
    features = [name for name in FEATURES if name in df.columns]
    frame = df[features].copy()
    for col in features:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = LabelEncoder().fit_transform(frame[col].fillna("Unknown").astype(str))
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    matrix = SimpleImputer(strategy="median").fit_transform(frame.astype(float))
    matrix = StandardScaler().fit_transform(matrix)
    ids = np.random.RandomState(42).choice(len(matrix), min(3000, len(matrix)), replace=False)
    labels = df.loc[ids, "status"].fillna("Open").replace({"Assigned": "In Progress", "Pending": "In Progress"}).astype(str).values
    coords = TSNE(n_components=2, perplexity=30, max_iter=1000, learning_rate="auto",
                  init="pca", random_state=42).fit_transform(matrix[ids])
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"Closed": "#16a34a", "In Progress": "#2563eb", "Open": "#f97316"}
    for status, color in palette.items():
        mask = labels == status
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=.45, color=color,
                       label=f"{status} ({mask.sum():,})")
    ax.set_title("t-SNE Embedding by Complaint Status (perplexity=30)")
    ax.set_xlabel("t-SNE Dimension 1"); ax.set_ylabel("t-SNE Dimension 2"); ax.legend(markerscale=2)
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT / "tsne_by_status.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    save_metric_and_importance_charts()
    save_dbscan_cluster_sizes()
    save_tsne_status_plot()
    print("Supplementary per-model charts created.")
