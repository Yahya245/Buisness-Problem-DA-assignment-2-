# Clean_data.py

import pandas as pd
import os

PROCESSED_PATH = "Processed data/"

# Ensure folder exists
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Utility functions ---
def standardise_columns(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def convert_dtypes(df):
    for col in df.columns:
        if "year" in col or col.isdigit():
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif "value_added" in col or "price" in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def drop_missing_key_rows(df, keys):
    return df.dropna(subset=keys)

# --- Energy cleaning ---
def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_columns(df)
    # Identify columns
    country_col = [c for c in df.columns if "country" in c.lower()][0]
    year_col = [c for c in df.columns if "time" in c.lower() or "year" in c.lower()][0]
    price_col = [c for c in df.columns if "electricity" in c.lower()][0]

    df = df[[country_col, year_col, price_col]]
    df = df.rename(columns={
        country_col: "country",
        year_col: "year",
        price_col: "electricity_price"
    })
    df = convert_dtypes(df)
    df = drop_missing_key_rows(df, ["country", "year", "electricity_price"])
    return df

# --- Industry cleaning ---
def clean_industry_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardise_columns(df)
    id_vars = ["country_name", "country_code", "indicator_name", "indicator_code"]
    value_vars = [c for c in df.columns if c not in id_vars and "unnamed" not in c]

    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="year",
        value_name="industry_value_added"
    )

    df_long = df_long.rename(columns={"country_name": "country"})
    df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
    df_long['industry_value_added'] = pd.to_numeric(df_long['industry_value_added'], errors='coerce')

    df_long = drop_missing_key_rows(df_long, ["country", "year", "industry_value_added"])
    return df_long

# --- Saving ---
def save_clean(df, name):
    path = os.path.join(PROCESSED_PATH, name)
    df.to_csv(path, index=False)
    print(f"Cleaned dataset saved: {path}")











