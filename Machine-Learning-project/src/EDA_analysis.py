# ============================================================
# CIVICFIX - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# This script generates visual charts to understand
# the CivicFix dataset patterns and distributions.
# All charts are saved to the outputs/EDA_analysis folder.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_DIR = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\outputs\EDA_analysis"

# Create the output folder if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 3. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded for EDA.")
print("Shape:", df.shape)

# Set a consistent visual style for all charts
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# ============================================================
# 4. COMPLAINT TYPE DISTRIBUTION (TOP 15)
# ============================================================

print("\nGenerating: Complaint Type Distribution...")

complaint_counts = df["complaint_type"].value_counts().head(15)

plt.figure(figsize=(12, 7))
complaint_counts.sort_values().plot(
    kind="barh",
    color="steelblue",
    edgecolor="black"
)
plt.title("Top 15 Complaint Types in CivicFix", fontsize=15)
plt.xlabel("Number of Complaints", fontsize=12)
plt.ylabel("Complaint Type", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "complaint_type_distribution.png"))
plt.close()
print("  Saved: complaint_type_distribution.png")


# ============================================================
# 5. COMPLAINT STATUS DISTRIBUTION
# ============================================================

print("\nGenerating: Complaint Status Distribution...")

status_counts = df["status"].value_counts()

# Define colors for each status
status_colors = {
    "Closed"      : "#4aa84c",
    "In Progress" : "#2878e8",
    "Open"        : "#f99a18"
}
bar_colors = [status_colors.get(s, "gray") for s in status_counts.index]

plt.figure(figsize=(8, 5))
bars = plt.bar(
    status_counts.index,
    status_counts.values,
    color=bar_colors,
    edgecolor="black",
    width=0.5
)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 50,
        f"{int(height):,}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.title("Complaint Status Distribution", fontsize=15)
plt.xlabel("Status", fontsize=12)
plt.ylabel("Number of Complaints", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "status_distribution.png"))
plt.close()
print("  Saved: status_distribution.png")


# ============================================================
# 6. STATUS PIE CHART
# ============================================================

print("\nGenerating: Status Pie Chart...")

plt.figure(figsize=(7, 7))
plt.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    colors=bar_colors,
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
plt.title("Complaint Status - Pie Chart", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "status_pie_chart.png"))
plt.close()
print("  Saved: status_pie_chart.png")


# ============================================================
# 7. BOROUGH DISTRIBUTION
# ============================================================

print("\nGenerating: Borough Distribution...")

borough_counts = df["borough"].value_counts()

plt.figure(figsize=(10, 5))
borough_counts.plot(
    kind="bar",
    color="mediumpurple",
    edgecolor="black"
)
plt.title("Number of Complaints by Borough", fontsize=15)
plt.xlabel("Borough", fontsize=12)
plt.ylabel("Number of Complaints", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "borough_distribution.png"))
plt.close()
print("  Saved: borough_distribution.png")


# ============================================================
# 8. AGENCY DISTRIBUTION (TOP 10)
# ============================================================

print("\nGenerating: Agency Distribution...")

agency_counts = df["agency"].value_counts().head(10)

plt.figure(figsize=(10, 5))
agency_counts.sort_values().plot(
    kind="barh",
    color="darkorange",
    edgecolor="black"
)
plt.title("Top 10 Agencies Handling Complaints", fontsize=15)
plt.xlabel("Number of Complaints", fontsize=12)
plt.ylabel("Agency", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "agency_distribution.png"))
plt.close()
print("  Saved: agency_distribution.png")


# ============================================================
# 9. CHANNEL TYPE DISTRIBUTION
# ============================================================

print("\nGenerating: Channel Type Distribution...")

if "open_data_channel_type" in df.columns:
    channel_counts = df["open_data_channel_type"].value_counts()

    plt.figure(figsize=(8, 5))
    channel_counts.plot(
        kind="bar",
        color="teal",
        edgecolor="black"
    )
    plt.title("Complaints by Submission Channel", fontsize=15)
    plt.xlabel("Channel Type", fontsize=12)
    plt.ylabel("Number of Complaints", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "channel_distribution.png"))
    plt.close()
    print("  Saved: channel_distribution.png")


# ============================================================
# 10. COMPLAINT TYPE vs STATUS STACKED BAR
# ============================================================

print("\nGenerating: Complaint Type vs Status...")

top_types = df["complaint_type"].value_counts().head(8).index.tolist()
df_top    = df[df["complaint_type"].isin(top_types)]

pivot_table = df_top.groupby(
    ["complaint_type", "status"]
).size().unstack(fill_value=0)

pivot_table.plot(
    kind="bar",
    figsize=(14, 7),
    color=["#4aa84c", "#2878e8", "#f99a18"],
    edgecolor="black"
)
plt.title("Top 8 Complaint Types by Status", fontsize=15)
plt.xlabel("Complaint Type", fontsize=12)
plt.ylabel("Number of Complaints", fontsize=12)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Status")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "complaint_type_vs_status.png"))
plt.close()
print("  Saved: complaint_type_vs_status.png")


# ============================================================
# 11. MONTHLY COMPLAINT TREND
# ============================================================

print("\nGenerating: Monthly Complaint Trend...")

if "created_date" in df.columns:
    df["created_date_parsed"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )
    df["year_month"] = df["created_date_parsed"].dt.to_period("M").astype(str)

    monthly_counts = df.groupby("year_month").size().reset_index(name="count")
    monthly_counts = monthly_counts.sort_values("year_month")

    plt.figure(figsize=(12, 5))
    plt.plot(
        monthly_counts["year_month"],
        monthly_counts["count"],
        marker="o",
        color="#2878e8",
        linewidth=2
    )
    plt.title("Monthly Complaint Volume", fontsize=15)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Number of Complaints", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_trend.png"))
    plt.close()
    print("  Saved: monthly_trend.png")


# ============================================================
# 12. NUMERICAL COLUMN DISTRIBUTIONS
# ============================================================

print("\nGenerating: Numerical Column Distributions...")

numerical_cols = ["latitude", "longitude", "council_district"]
existing_num   = [c for c in numerical_cols if c in df.columns]

for col in existing_num:
    plt.figure(figsize=(8, 4))
    df[col].dropna().hist(
        bins=40,
        color="steelblue",
        edgecolor="black"
    )
    plt.title(f"Distribution of {col}", fontsize=13)
    plt.xlabel(col, fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    filename = f"distribution_{col}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  Saved: {filename}")


# ============================================================
# 13. SUMMARY
# ============================================================

print("\n" + "="*60)
print("EDA COMPLETE - ALL CHARTS SAVED")
print("="*60)
print(f"Output folder: {OUTPUT_DIR}")
print("Charts generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".png"):
        print(f"  - {f}")
