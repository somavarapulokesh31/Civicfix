# ============================================================
# CIVICFIX - EMBEDDING ENCODING (PREPROCESSING MODULE 2)
# ============================================================
# CivicFix dataset (erm2-nwe9.csv) has high-cardinality
# categorical columns like complaint_type (100+ types) and
# agency (30+ agencies).
#
# Label encoding loses the relationship between categories.
# One-hot encoding creates too many columns.
#
# EMBEDDING ENCODING (Mean Target Encoding):
#   Replace each category with its historical mean of
#   the numeric target (status: Open=0, InProgress=1, Closed=2)
#
#   Example:
#     NYPD complaints → mean status = 0.82 (mostly closed)
#     HPD complaints  → mean status = 1.34 (often in progress)
#
# This preserves the category's relationship with the target
# and keeps the data compact.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_embedding_encoded.csv"

OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\preprocessing"

MODEL_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\models"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

# Drop unnamed columns
unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
df.drop(columns=unnamed, inplace=True)

print("CivicFix dataset loaded.")
print("Shape:", df.shape)


# ============================================================
# 4. STANDARDISE STATUS TO 3 CLASSES
# ============================================================
# Original: Closed, In Progress, Open, Assigned, Pending, Unspecified
# Mapped to: Closed=2, In Progress=1, Open=0

print("\nOriginal status distribution:")
print(df["status"].value_counts())

def standardise_status(val):
    if val == "Closed":
        return "Closed"
    elif val in ["In Progress", "Assigned", "Pending"]:
        return "In Progress"
    else:       # Open, Unspecified
        return "Open"

df["status"] = df["status"].apply(standardise_status)

STATUS_MAP  = {"Open": 0, "In Progress": 1, "Closed": 2}
df["status_num"] = df["status"].map(STATUS_MAP)

print("\nStatus after standardisation (3 classes):")
print(df["status"].value_counts())


# ============================================================
# 5. COLUMNS TO EMBED
# ============================================================
# These are high-cardinality columns in the CivicFix dataset
# that benefit from mean target encoding:
#
#   complaint_type  - 100+ types (Noise, HEAT/HOT WATER, etc.)
#   agency          - 30+ agencies (NYPD, HPD, DSNY, DOT, etc.)
#   borough         - 5 NYC boroughs
#   location_type   - Street, Building, etc.

EMBED_COLS = [
    c for c in ["complaint_type", "agency", "borough", "location_type"]
    if c in df.columns
]

print(f"\nColumns to embed: {EMBED_COLS}")
for col in EMBED_COLS:
    print(f"  {col}: {df[col].nunique()} unique values")


# ============================================================
# 6. APPLY MEAN TARGET ENCODING
# ============================================================
# For each category, calculate:
#   _mean_status   → average resolution stage (0-2)
#   _closed_rate   → % of complaints that were Closed
#   _open_rate     → % of complaints that were Open
#   _volume        → how many complaints of that type exist

embedding_maps = {}
df_encoded     = df.copy()

print("\nApplying mean target embedding...")

for col in EMBED_COLS:

    grp = df.groupby(col)

    mean_status  = grp["status_num"].mean()
    closed_rate  = grp["status"].apply(lambda x: (x == "Closed").mean())
    open_rate    = grp["status"].apply(lambda x: (x == "Open").mean())
    volume       = grp.size()

    df_encoded[f"{col}_mean_status"]  = df[col].map(mean_status)
    df_encoded[f"{col}_closed_rate"]  = df[col].map(closed_rate)
    df_encoded[f"{col}_open_rate"]    = df[col].map(open_rate)
    df_encoded[f"{col}_volume"]       = df[col].map(volume)

    embedding_maps[col] = {
        "mean_status"  : mean_status.to_dict(),
        "closed_rate"  : closed_rate.to_dict(),
        "open_rate"    : open_rate.to_dict(),
        "volume"       : volume.to_dict(),
    }

    print(f"\n  {col}:")
    print(f"    Mean status range: {mean_status.min():.3f} – {mean_status.max():.3f}")
    print(f"    Closed rate range: {closed_rate.min():.3f} – {closed_rate.max():.3f}")


# ============================================================
# 7. SHOW TOP AGENCIES BY MEAN STATUS (INTERESTING INSIGHT)
# ============================================================

print("\n" + "="*60)
print("  AGENCY PERFORMANCE (mean status: 0=Open, 2=Closed)")
print("="*60)

if "agency" in EMBED_COLS:
    agency_perf = embedding_maps["agency"]["mean_status"]
    agency_df   = pd.DataFrame(
        list(agency_perf.items()),
        columns=["Agency", "Mean_Status"]
    ).sort_values("Mean_Status", ascending=False).head(10)
    print(agency_df.to_string(index=False))


# ============================================================
# 8. VISUALISE BOROUGH CLOSED RATES
# ============================================================

print("\nGenerating: Borough closed rate chart...")

if "borough" in EMBED_COLS:
    borough_closed = embedding_maps["borough"]["closed_rate"]
    b_df = pd.DataFrame(
        list(borough_closed.items()),
        columns=["Borough", "Closed_Rate"]
    ).sort_values("Closed_Rate", ascending=True)

    plt.figure(figsize=(9, 5))
    colors = ["#4aa84c" if r >= 0.55 else "#2878e8" for r in b_df["Closed_Rate"]]
    bars   = plt.barh(b_df["Borough"], b_df["Closed_Rate"] * 100,
                      color=colors, edgecolor="black")
    for bar, rate in zip(bars, b_df["Closed_Rate"]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{rate*100:.1f}%", va="center", fontsize=10)
    plt.title("CivicFix — Borough Complaint Closed Rate\n"
              "(Embedding: Mean Target Encoding)", fontsize=13)
    plt.xlabel("Closed Rate (%)", fontsize=11)
    plt.xlim(0, 80)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "embedding_borough_closed_rate.png"))
    plt.close()
    print("  Saved: embedding_borough_closed_rate.png")


# ============================================================
# 9. VISUALISE TOP COMPLAINT TYPES BY CLOSED RATE
# ============================================================

print("Generating: Top complaint types closed rate chart...")

if "complaint_type" in EMBED_COLS:
    ct_closed = embedding_maps["complaint_type"]["closed_rate"]
    ct_df = pd.DataFrame(
        list(ct_closed.items()),
        columns=["Complaint_Type", "Closed_Rate"]
    ).sort_values("Closed_Rate", ascending=False).head(12)

    plt.figure(figsize=(11, 6))
    plt.barh(ct_df["Complaint_Type"].str[:30],
             ct_df["Closed_Rate"] * 100,
             color="#2878e8", edgecolor="black")
    plt.title("CivicFix — Top 12 Complaint Types by Closed Rate\n"
              "(Embedding Feature)", fontsize=13)
    plt.xlabel("Closed Rate (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "embedding_complaint_type_closed_rate.png"))
    plt.close()
    print("  Saved: embedding_complaint_type_closed_rate.png")


# ============================================================
# 10. DROP TEMPORARY COLUMN
# ============================================================

df_encoded.drop(columns=["status_num"], inplace=True)


# ============================================================
# 11. SAVE EMBEDDING MAPS
# ============================================================

EMBED_PATH = os.path.join(MODEL_DIR, "embedding_maps.pkl")
with open(EMBED_PATH, "wb") as f:
    pickle.dump(embedding_maps, f)

print(f"\nEmbedding maps saved to:\n  {EMBED_PATH}")


# ============================================================
# 12. SAVE EMBEDDED DATASET
# ============================================================

print("\nShape before embedding:", df.shape)
print("Shape after  embedding:", df_encoded.shape)

df_encoded.to_csv(OUTPUT_PATH, index=False)

print(f"\nEmbedding-encoded dataset saved to:\n  {OUTPUT_PATH}")
print("\nEmbedding encoding complete.")
