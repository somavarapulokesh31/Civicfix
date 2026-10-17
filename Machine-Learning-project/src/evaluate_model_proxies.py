"""Calculate clearly labeled downstream status-prediction proxy scores.

These are not native accuracy scores for clustering or dimensionality
reduction. A fixed logistic-regression evaluator predicts status from each
model's output representation on a held-out split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "erm2-nwe9.csv"
OUT = ROOT / "outputs" / "model_results" / "model_proxy_results.csv"
SEED = 42
SAMPLE_SIZE = 3000


def scores(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1_Score": round(f1 * 100, 2),
    }


def evaluate_representation(name, representation, y, train_idx, test_idx, rows):
    representation = np.asarray(representation)
    if representation.ndim == 1:
        representation = representation.reshape(-1, 1)
    evaluator = LogisticRegression(max_iter=500, class_weight="balanced")
    evaluator.fit(representation[train_idx], y[train_idx])
    rows.append({"Model": name, **scores(y[test_idx], evaluator.predict(representation[test_idx]))})


def main():
    df = pd.read_csv(DATA, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
    status = df["status"].fillna("Open").replace({"Assigned": "In Progress", "Pending": "In Progress"})
    y = LabelEncoder().fit_transform(status.astype(str))

    features = [c for c in ["complaint_type", "agency", "borough", "location_type",
                            "open_data_channel_type", "council_district", "latitude", "longitude"]
                if c in df.columns]
    X = df[features].copy()
    for col in features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].fillna("Unknown").astype(str)
            X[col] = LabelEncoder().fit_transform(X[col])
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)

    rng = np.random.RandomState(SEED)
    sample = rng.choice(len(X), min(SAMPLE_SIZE, len(X)), replace=False)
    X, y = X[sample], y[sample]
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.25, random_state=SEED, stratify=y
    )
    X_train = X[train_idx]
    rows = []

    km = KMeans(n_clusters=5, random_state=SEED, n_init=10).fit(X_train)
    evaluate_representation("K-Means Clustering", np.eye(5)[km.predict(X)], y, train_idx, test_idx, rows)

    agg_train = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(X_train)
    centers = np.vstack([X_train[agg_train == c].mean(axis=0) for c in np.unique(agg_train)])
    agg_labels = np.argmin(((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), axis=1)
    evaluate_representation("Hierarchical Clustering", np.eye(5)[agg_labels], y, train_idx, test_idx, rows)

    iso = IsolationForest(n_estimators=150, contamination=0.05, random_state=SEED, n_jobs=-1).fit(X_train)
    evaluate_representation("Anomaly Detection (Isolation Forest)", iso.decision_function(X)[:, None], y, train_idx, test_idx, rows)

    # DBSCAN cluster IDs are evaluated as categorical output features. Fit on
    # the full unlabeled sample, then evaluate status prediction on a holdout.
    raw_geo = df.loc[sample, ["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce")
    raw_geo = SimpleImputer(strategy="median").fit_transform(raw_geo)
    db_raw = DBSCAN(eps=0.02, min_samples=15).fit_predict(raw_geo)
    raw_labels = LabelEncoder().fit_transform(db_raw)
    evaluate_representation("DBSCAN (Raw Dataset)", np.eye(raw_labels.max() + 1)[raw_labels], y, train_idx, test_idx, rows)
    db_pre = DBSCAN(eps=0.8, min_samples=10).fit_predict(X)
    pre_labels = LabelEncoder().fit_transform(db_pre)
    evaluate_representation("DBSCAN (Pre-Processed)", np.eye(pre_labels.max() + 1)[pre_labels], y, train_idx, test_idx, rows)

    pca = PCA(n_components=2, random_state=SEED).fit_transform(X)
    evaluate_representation("PCA", pca, y, train_idx, test_idx, rows)

    # t-SNE is transductive: all sampled feature rows are embedded without
    # labels, then status prediction is evaluated on held-out labels.
    tsne = TSNE(n_components=2, perplexity=30, max_iter=750, random_state=SEED,
                init="pca", learning_rate="auto").fit_transform(X)
    evaluate_representation("t-SNE", tsne, y, train_idx, test_idx, rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Saved {len(rows)} proxy metric rows to {OUT}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
