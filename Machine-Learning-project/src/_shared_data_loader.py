# ============================================================
# CIVICFIX - SHARED DATA LOADER
# ============================================================
# Loads erm2-nwe9.csv, cleans, label-encodes, returns X, y
# ready for any sklearn model.
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

FEATURE_COLS = [
    "complaint_type",
    "agency",
    "borough",
    "location_type",
    "open_data_channel_type",
    "council_district",
    "latitude",
    "longitude",
]

STATUS_MAP = {"Open": 0, "In Progress": 1, "Closed": 2}


def load_civicfix(verbose=True):
    # ── 1. Load ──────────────────────────────────────────────
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    # Drop completely unnamed columns
    df.drop(
        columns=[c for c in df.columns if str(c).startswith("Unnamed")],
        inplace=True,
    )

    # ── 2. Standardise status → 3 classes ────────────────────
    def std_status(v):
        if v == "Closed":
            return "Closed"
        if v in ["In Progress", "Assigned", "Pending"]:
            return "In Progress"
        return "Open"          # Open, Unspecified, etc.

    df["status"] = df["status"].apply(std_status)
    df["status_encoded"] = df["status"].map(STATUS_MAP).astype("Int64")

    # ── 3. Keep only the columns we need ─────────────────────
    present_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df[present_features + ["status_encoded"]].copy()

    # ── 4. Drop rows where TARGET is missing ─────────────────
    df.dropna(subset=["status_encoded"], inplace=True)

    # ── 5. Fill missing feature values before encoding ───────
    for col in present_features:
        if df[col].dtype == "object" or str(df[col].dtype) in ("string", "StringDtype"):
            df[col] = df[col].fillna("Unknown")
        else:
            # numeric column — fill with median
            med = pd.to_numeric(df[col], errors="coerce").median()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med)

    # ── 6. Label-encode every categorical feature ────────────
    for col in present_features:
        if df[col].dtype == "object" or str(df[col].dtype) in ("string", "StringDtype"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # ── 7. Build X and y — drop any remaining NaN rows ──────
    X = df[present_features].values.astype(float)
    y = df["status_encoded"].values.astype(int)

    # Final safety: replace any remaining NaN/inf with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    if verbose:
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}
        print(f"Dataset loaded : {len(df):,} rows")
        print(f"Features       : {present_features}")
        print(f"Class dist     : {dist}  (0=Open, 1=InProgress, 2=Closed)")

    return X, y, present_features
