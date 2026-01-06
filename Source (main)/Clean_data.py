# Data cleaning & reshaping logic

# Clean_data.py

import pandas as pd

# -------------------
# Helper functions
# -------------------

def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, and replace spaces in column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    return df

def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert year/price/value columns to numeric types."""
    for col in df.columns:
        if col not in ["country", "year"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def drop_missing_key_rows(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Drop rows missing critical key values."""
    return df.dropna(subset=keys)

# -------------------
# Energy dataset
# -------------------

def clean_energy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean electricity price dataset.
    """
    df = standardise_columns(df)

    # Map actual column names from World Bank CSV
    # Adjust this if your column names differ
    df = df.rename(columns={
        'country_name': 'country',
        'time': 'year',
        'getting_electricity:_price_of_electricity_(us_cents_per_kwh)_(db16-20_methodology)_[ic.elc.pri.kh.db1619]': 'electricity_price'
    })

    df = df[['country', 'year', 'electricity_price']]

    df = convert_dtypes(df)
    df = drop_missing_key_rows(df, ["country", "year", "electricity_price"])

    return df

# -------------------
# Industry dataset
# -------------------

def clean_industry_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean industry value added dataset (wide → long format).
    """
    df = standardise_columns(df)

    # Keep only country + year columns (years usually start from column index 4)
    cols_to_keep = [df.columns[0]] + list(df.columns[4:])
    df = df[cols_to_keep]

    # Melt wide → long
    df = df.melt(id_vars=[df.columns[0]],
                 var_name="year",
                 value_name="industry_value_added")

    # Rename first column to 'country'
    df = df.rename(columns={df.columns[0]: "country"})

    df = convert_dtypes(df)
    df = drop_missing_key_rows(df, ["country", "year", "industry_value_added"])

    return df

