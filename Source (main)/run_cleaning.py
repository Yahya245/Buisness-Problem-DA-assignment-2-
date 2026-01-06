# run_cleaning.py

import pandas as pd
from Clean_data import clean_energy_data, clean_industry_data
import os

# -------------------
# Load data
# -------------------

def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV and handle common encoding issues and irregular lines."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1", engine="python", on_bad_lines="skip")
    return df


def load_energy_data() -> pd.DataFrame:
    path = "Raw data/Electricity datasets/Energy-dataset.csv"
    return load_csv(path)


def load_industry_data() -> pd.DataFrame:
    path = "Raw data/Industry GDP Datasets/Industry-GDP.csv"
    return load_csv(path)


# -------------------
# Save cleaned data
# -------------------

def save_clean_data(energy_df: pd.DataFrame, industry_df: pd.DataFrame):
    os.makedirs("Processed data", exist_ok=True)
    energy_df.to_csv("Processed data/energy_clean.csv", index=False)
    industry_df.to_csv("Processed data/industry_clean.csv", index=False)
    print("Cleaned datasets saved in 'Processed data/' folder.")

# -------------------
# Main pipeline
# -------------------

def run_cleaning_pipeline():
    print("Loading raw datasets...")

    energy_df = load_energy_data()
    industry_df = load_industry_data()

    # Debug prints
    print("Energy columns:")
    print(energy_df.columns)
    print("Energy raw shape:", energy_df.shape)

    print("Industry columns:")
    print(industry_df.columns)
    print("Industry raw shape:", industry_df.shape)

    print("Cleaning energy data...")
    energy_clean = clean_energy_data(energy_df)

    print("Cleaning industry data...")
    industry_clean = clean_industry_data(industry_df)

    save_clean_data(energy_clean, industry_clean)

    print("Cleaning pipeline completed successfully.")

# -------------------
# Run if main
# -------------------

if __name__ == "__main__":
    run_cleaning_pipeline()

