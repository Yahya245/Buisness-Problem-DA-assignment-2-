# run_cleaning.py
# This part of the code is essential for the processed data being produced.

import pandas as pd
from Clean_data import clean_energy_data, clean_industry_data, save_clean
import os

RAW_PATH = "Raw data/"

def load_csv(file_path, skiprows=0, encoding='utf-8'):
    try:
        df = pd.read_csv(file_path, skiprows=skiprows, encoding=encoding, engine="python", on_bad_lines="skip")
        return df
    except UnicodeDecodeError:
        print(f"UnicodeDecodeError with {encoding}, retrying with latin1...")
        df = pd.read_csv(file_path, skiprows=skiprows, encoding="latin1", engine="python", on_bad_lines="skip")
        return df

def load_energy_data():
    path = os.path.join(RAW_PATH, "Electricity datasets/Energy-dataset.csv")
    df = load_csv(path)
    print("Energy columns:", df.columns)
    print("Energy raw shape:", df.shape)
    return df

def load_industry_data():
    # Replace this with the exact path to your downloaded Industry-GDP.csv
    path = os.path.join(RAW_PATH, "Industry GDP datasets/Industry-GDP.csv")
    df = load_csv(path, skiprows=4, encoding='latin1')
    print("Industry columns:", df.columns)
    print("Industry raw shape:", df.shape)
    return df

def run_cleaning_pipeline():
    print("Loading raw datasets...")
    energy_df = load_energy_data()
    industry_df = load_industry_data()

    print("Cleaning energy data...")
    energy_clean = clean_energy_data(energy_df)
    save_clean(energy_clean, "energy_clean.csv")

    print("Cleaning industry data...")
    industry_clean = clean_industry_data(industry_df)
    save_clean(industry_clean, "industry_clean.csv")

    print("Cleaning pipeline completed successfully.")

if __name__ == "__main__":
    run_cleaning_pipeline()






