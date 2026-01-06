# Function to load CSV files
""" This part of code handles loading and basic validation of raw datatsets used int he business problem data analysis project"""

import os
import pandas as pd

#Generic CSV loader function
def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame with basic validation.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


    df = pd.read_csv(
    file_path,
    encoding="latin1",
    engine="python",
    on_bad_lines="skip"
)

    if df.empty:
        raise ValueError("The loaded dataset is empty.")

    return df

#loading the Energy dataset
def load_energy_data() -> pd.DataFrame:
    """
    Load the electricity price dataset.

    Returns
    -------
    pd.DataFrame
        Electricity price data by country and year.
    """

    path = os.path.join("Raw data","Electricity datasets","Energy-dataset.csv")
    df = load_csv(path)

    return df
#loading the GDP dataset
def load_industry_data() -> pd.DataFrame:
    """
    Load the industry value added dataset.

    Returns
    -------
    pd.DataFrame
        Industry value added (% of GDP) by country and year.
    """

    path = os.path.join("Raw data","Industry GDP Datasets", "Industry-GDP.csv")
    df = load_csv(path)

    return df

def preview_data(df: pd.DataFrame, rows: int = 5) -> None:
    """Quickly inspect dataset structure and missing values."""
    print(df.head(rows))
    print("\nMissing values:\n", df.isna().sum())




