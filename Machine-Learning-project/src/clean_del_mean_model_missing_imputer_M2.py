# ============================================================
# CIVICFIX - HANDLE MISSING VALUES (PREPROCESSING MODULE 2)
# ============================================================
# Loads the original dataset, drops high-missing columns,
# fills remaining missing values, and saves cleaned output.
# ============================================================


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import pandas as pd


# ============================================================
# 2. DEFINE PATHS
# ============================================================

DATASET_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\erm2-nwe9.csv"

OUTPUT_PATH = r"C:\Users\somavarapu lokesh\OneDrive\Desktop\ML PROJECT\Machine-Learning-project\dataset\clean_missing_values.csv"


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Original dataset loaded.")
print("Shape before cleaning:", df.shape)


# ============================================================
# 4. DROP UNNAMED COLUMNS
# ============================================================

unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]

df.drop(columns=unnamed_cols, inplace=True)

print(f"Dropped {len(unnamed_cols)} unnamed columns.")


# ============================================================
# 5. DROP COLUMNS WITH MORE THAN 60% MISSING VALUES
# ============================================================

missing_percent = (df.isnull().sum() / len(df)) * 100

cols_to_drop = missing_percent[missing_percent > 60].index.tolist()

print(f"\nDropping {len(cols_to_drop)} columns with >60% missing values:")

for c in cols_to_drop:
    print(f"  - {c} ({missing_percent[c]:.1f}% missing)")

df.drop(columns=cols_to_drop, inplace=True)


# ============================================================
# 6. STANDARDISE STATUS TO 3 CLASSES
# ============================================================

def standardise_status(val):

    if val == "Closed":
        return "Closed"

    elif val == "In Progress":
        return "In Progress"

    else:
        return "Open"


df["status"] = df["status"].apply(standardise_status)

print("\nStatus standardised to: Closed | In Progress | Open")

print(df["status"].value_counts())


# ============================================================
# 7. FILL MISSING VALUES IN NUMERICAL COLUMNS WITH MEAN
# ============================================================

numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

filled_numerical = 0

for col in numerical_cols:

    if df[col].isnull().sum() > 0:

        mean_val = df[col].mean()

        df[col] = df[col].fillna(mean_val)

        filled_numerical += 1


print(
    f"\nFilled missing values in {filled_numerical} "
    "numerical columns with mean."
)


# ============================================================
# 8. FILL MISSING VALUES IN CATEGORICAL COLUMNS WITH MODE
# ============================================================

categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

filled_categorical = 0

for col in categorical_cols:

    if df[col].isnull().sum() > 0:

        mode_values = df[col].mode()

        if len(mode_values) > 0:

            mode_val = mode_values.iloc[0]

            df[col] = df[col].fillna(mode_val)

            filled_categorical += 1


print(
    f"Filled missing values in {filled_categorical} "
    "categorical columns with mode."
)


# ============================================================
# 9. REMOVE DUPLICATE ROWS
# ============================================================

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"\nDuplicate rows removed: {before - after}")


# ============================================================
# 10. FINAL SHAPE AND SAVE
# ============================================================

print("\nShape after cleaning:", df.shape)

print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# 11. SAVE CLEANED DATASET
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print(
    f"\nSaved cleaned dataset to:\n"
    f"  {OUTPUT_PATH}"
)