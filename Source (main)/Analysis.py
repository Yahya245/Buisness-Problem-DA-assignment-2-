# Statistical analysis & regression
# this first section of the analysis file will focus on simply mering the two clean datasets and preparing them for high quality analysis 

import os
import pandas as pd


# ---------------------------------------------------
# Configuration
# ---------------------------------------------------

PROCESSED_DATA_DIR = "Processed data"
ENERGY_FILE = "energy_clean.csv"
INDUSTRY_FILE = "industry_clean.csv"
OUTPUT_FILE = "merged_energy_industry.csv"


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    return df


def validate_columns(df: pd.DataFrame, required_cols: list, df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ---------------------------------------------------
# Merge logic
# ---------------------------------------------------

def merge_energy_and_industry(
    energy_df: pd.DataFrame,
    industry_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge energy and industry datasets using:
    - energy.country  == industry.country_code
    - year
    """

    # ---- Validate structure ----
    validate_columns(
        energy_df,
        ["country", "year", "electricity_price"],
        "energy_clean"
    )

    validate_columns(
        industry_df,
        ["country_code", "year", "industry_value_added"],
        "industry_clean"
    )

    # ---- Ensure consistent dtypes ----
    energy_df["country"] = energy_df["country"].astype(str)
    industry_df["country_code"] = industry_df["country_code"].astype(str)

    energy_df["year"] = pd.to_numeric(energy_df["year"], errors="coerce")
    industry_df["year"] = pd.to_numeric(industry_df["year"], errors="coerce")

    # ---- Drop rows with missing join keys ----
    energy_df = energy_df.dropna(subset=["country", "year"])
    industry_df = industry_df.dropna(subset=["country_code", "year"])

    print(f"Energy rows before merge: {len(energy_df)}")
    print(f"Industry rows before merge: {len(industry_df)}")

    # ---- Perform merge ----
    merged_df = pd.merge(
        energy_df,
        industry_df,
        how="inner",
        left_on=["country", "year"],
        right_on=["country_code", "year"]
    )

    print(f"Merged rows after join: {len(merged_df)}")

    # ---- Clean up columns ----
    # Keep the industry country_code as the master identifier
    merged_df = merged_df.rename(columns={
    "country": "energy_country_code"
})

# Final column selection (guaranteed to exist)
    merged_df = merged_df[
    [
        "country_code",          # from industry_clean
        "year",
        "electricity_price",
        "industry_value_added"
    ]
]


    # ---- Reorder columns ----
    merged_df = merged_df[
        [
            "country_code",
            "year",
            "electricity_price",
            "industry_value_added"
        ]
    ]

    return merged_df


# ---------------------------------------------------
# Pipeline runner
# ---------------------------------------------------

def run_merge_pipeline():
    print("Loading cleaned datasets...")

    energy_path = os.path.join(PROCESSED_DATA_DIR, ENERGY_FILE)
    industry_path = os.path.join(PROCESSED_DATA_DIR, INDUSTRY_FILE)

    energy_df = load_csv(energy_path)
    industry_df = load_csv(industry_path)

    print("Energy columns:", list(energy_df.columns))
    print("Industry columns:", list(industry_df.columns))

    merged_df = merge_energy_and_industry(energy_df, industry_df)

    output_path = os.path.join(PROCESSED_DATA_DIR, OUTPUT_FILE)
    merged_df.to_csv(output_path, index=False)

    print(f"Merged dataset saved to: {output_path}")
    print("Merge pipeline completed successfully.")


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------

if __name__ == "__main__":
    run_merge_pipeline()

# ============================================================
# ANALYSIS.PY
# Section 1 & 2: Data Loading, Inspection & Descriptive Stats
# ============================================================

import pandas as pd
import os

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

MERGED_DATA_PATH = "Processed data/merged_energy_industry.csv"
OUTPUT_TABLES_DIR = "Outputs/tables"

# Ensure output directories exist
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)

# ------------------------------------------------------------
# SECTION 1: LOAD AND INSPECT MERGED DATA
# ------------------------------------------------------------
# Purpose:
# - Load merged dataset
# - Validate structure
# - Check for missing values
# - Ensure dataset is ready for analysis
# ------------------------------------------------------------

def load_merged_data() -> pd.DataFrame:
    """
    Loads the merged energy and industry dataset.
    """
    print("Loading merged dataset...")
    df = pd.read_csv(MERGED_DATA_PATH)
    print("Dataset loaded successfully.\n")
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Performs basic inspection of the dataset.
    """
    print("----- DATA INSPECTION -----")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}\n")

    print("Column names:")
    print(df.columns.tolist(), "\n")

    print("Data types:")
    print(df.dtypes, "\n")

    print("First 5 rows:")
    print(df.head(), "\n")

    print("Missing values per column:")
    print(df.isna().sum(), "\n")


def run_section_1():
    """
    Executes Section 1 of the analysis.
    """
    merged_df = load_merged_data()
    inspect_data(merged_df)
    return merged_df


# ------------------------------------------------------------
# SECTION 2: DESCRIPTIVE STATISTICS
# ------------------------------------------------------------
# Purpose:
# - Generate summary statistics
# - Analyse distributions of key variables
# - Save outputs for reporting
# ------------------------------------------------------------

def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates descriptive statistics for numerical variables.
    """
    print("Generating descriptive statistics...")
    desc_stats = df.describe()
    print(desc_stats, "\n")
    return desc_stats


def yearly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes yearly averages for electricity price
    and industry value added.
    """
    print("Calculating yearly aggregates...")
    yearly_stats = (
        df.groupby("year")[["electricity_price", "industry_value_added"]]
        .mean()
        .reset_index()
    )
    return yearly_stats


def country_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes country-level averages.
    """
    print("Calculating country-level averages...")
    country_stats = (
        df.groupby("country_code")[["electricity_price", "industry_value_added"]]
        .mean()
        .reset_index()
    )
    return country_stats


def save_table(df: pd.DataFrame, filename: str) -> None:
    """
    Saves a dataframe to the Outputs/tables directory.
    """
    path = os.path.join(OUTPUT_TABLES_DIR, filename)
    df.to_csv(path, index=False)
    print(f"Saved table: {path}")


def run_section_2(df: pd.DataFrame):
    """
    Executes Section 2 of the analysis.
    """
    # Overall descriptive statistics
    desc_stats = descriptive_statistics(df)
    save_table(desc_stats.reset_index(), "descriptive_statistics_overall.csv")

    # Yearly statistics
    yearly_stats = yearly_aggregates(df)
    save_table(yearly_stats, "yearly_averages.csv")

    # Country-level statistics
    country_stats = country_level_summary(df)
    save_table(country_stats, "country_averages.csv")


# ------------------------------------------------------------
# MAIN EXECUTION (SECTIONS 1 & 2 ONLY)
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Starting analysis: Sections 1 & 2\n")

    merged_data = run_section_1()
    run_section_2(merged_data)

    print("\nSections 1 & 2 completed successfully.")

