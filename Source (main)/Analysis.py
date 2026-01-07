# Statistical analysis & regression
# this first section of the analysis file will focus on simply mering the two clean datasets and preparing them for high quality analysis 

import pandas as pd
import os

# ==================================================
# 1. FILE PATH CONFIGURATION
# ==================================================

BASE_DIR = "Processed data"

ENERGY_PATH = os.path.join(BASE_DIR, "energy_clean.csv")
INDUSTRY_PATH = os.path.join(BASE_DIR, "industry_clean.csv")
MERGED_PATH = os.path.join(BASE_DIR, "merged_energy_industry.csv")

# ==================================================
# 2. LOAD CLEANED DATA
# ==================================================

def load_clean_data():
    print("Loading cleaned datasets...")

    energy_df = pd.read_csv(ENERGY_PATH)
    industry_df = pd.read_csv(INDUSTRY_PATH)

    print(f"Energy shape: {energy_df.shape}")
    print(f"Industry shape: {industry_df.shape}")

    return energy_df, industry_df

# ==================================================
# 3. BASIC STANDARDISATION
# ==================================================

def standardise_columns(energy_df, industry_df):
    """
    Ensures consistency in key merge columns.
    """

    energy_df.columns = energy_df.columns.str.lower().str.strip()
    industry_df.columns = industry_df.columns.str.lower().str.strip()

    energy_df["country"] = energy_df["country"].str.strip()
    industry_df["country"] = industry_df["country"].str.strip()

    energy_df["year"] = pd.to_numeric(energy_df["year"], errors="coerce")
    industry_df["year"] = pd.to_numeric(industry_df["year"], errors="coerce")

    return energy_df, industry_df

# ==================================================
# 4. PRE-MERGE VALIDATION
# ==================================================

def validate_inputs(energy_df, industry_df):
    required_energy = ["country", "year", "electricity_price"]
    required_industry = ["country", "year", "industry_value_added"]

    for col in required_energy:
        if col not in energy_df.columns:
            raise ValueError(f"Missing column in energy data: {col}")

    for col in required_industry:
        if col not in industry_df.columns:
            raise ValueError(f"Missing column in industry data: {col}")

    print("Column validation passed.")

    energy_df.dropna(subset=required_energy, inplace=True)
    industry_df.dropna(subset=required_industry, inplace=True)

    print("Rows with missing merge keys removed.")
    print(f"Energy rows remaining: {len(energy_df)}")
    print(f"Industry rows remaining: {len(industry_df)}")

    return energy_df, industry_df

# ==================================================
# 5. KEY OVERLAP DIAGNOSTICS
# ==================================================

def inspect_merge_coverage(energy_df, industry_df):
    """
    Examines overlap in country–year combinations.
    """

    energy_keys = set(zip(energy_df["country"], energy_df["year"]))
    industry_keys = set(zip(industry_df["country"], industry_df["year"]))

    overlap = energy_keys.intersection(industry_keys)

    print(f"Energy unique keys: {len(energy_keys)}")
    print(f"Industry unique keys: {len(industry_keys)}")
    print(f"Overlapping keys: {len(overlap)}")

# ==================================================
# 6. MERGE DATASETS
# ==================================================

def merge_datasets(energy_df, industry_df):
    print("Merging datasets on country and year...")

    merged_df = pd.merge(
        energy_df,
        industry_df,
        how="inner",
        on=["country", "year"]
    )

    print(f"Merged dataset shape: {merged_df.shape}")

    return merged_df

# ==================================================
# 7. POST-MERGE CLEANUP
# ==================================================

def clean_merged_data(merged_df):
    merged_df.sort_values(
        by=["country", "year"],
        inplace=True
    )

    merged_df.reset_index(drop=True, inplace=True)

    numeric_cols = [
        "electricity_price",
        "industry_value_added"
    ]

    merged_df[numeric_cols] = merged_df[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    return merged_df

# ==================================================
# 8. SAVE OUTPUT
# ==================================================

def save_merged_data(merged_df):
    merged_df.to_csv(MERGED_PATH, index=False)
    print(f"Merged dataset saved: {MERGED_PATH}")

# ==================================================
# 9. PIPELINE EXECUTION
# ==================================================

def run_merge_pipeline():
    energy_df, industry_df = load_clean_data()
    energy_df, industry_df = standardise_columns(energy_df, industry_df)
    energy_df, industry_df = validate_inputs(energy_df, industry_df)
    inspect_merge_coverage(energy_df, industry_df)
    merged_df = merge_datasets(energy_df, industry_df)
    merged_df = clean_merged_data(merged_df)
    save_merged_data(merged_df)

    print("Merge pipeline completed successfully.")

if __name__ == "__main__":
    run_merge_pipeline()

