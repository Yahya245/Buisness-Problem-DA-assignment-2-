# Statistical analysis & regression
# this first section of the analysis file will focus on simply mering the two clean datasets and preparing them for high quality analysis 

import os
import pandas as pd
import seaborn as sns



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

# Analysis.py
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ensure output folders exist
os.makedirs("Outputs/tables", exist_ok=True)
os.makedirs("Outputs/figures", exist_ok=True)

# -----------------------------
# SECTION 1: DATA INSPECTION
# -----------------------------
def run_section_1(df: pd.DataFrame):
    print("\n----- SECTION 1: DATA INSPECTION -----\n")
    
    # Shape and columns
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}\n")
    
    # Data types
    print("Data types:")
    print(df.dtypes, "\n")
    
    # First 5 rows
    print("First 5 rows:")
    print(df.head(), "\n")
    
    # Missing values
    missing = df.isna().sum()
    print("Missing values per column:")
    print(missing, "\n")
    
    # Descriptive statistics
    desc = df.describe()
    print("Descriptive statistics:")
    print(desc, "\n")
    
    # Save table
    desc.to_csv("Outputs/tables/descriptive_statistics_overall.csv")
    
    return df  # Return df so we can pass it to section 2

# -----------------------------
# SECTION 2: YEARLY & COUNTRY AGGREGATES
# -----------------------------
def run_section_2(df: pd.DataFrame):
    print("\n----- SECTION 2: YEARLY & COUNTRY AGGREGATES -----\n")
    
    # Yearly averages
    yearly_avg = df.groupby("year")[["electricity_price", "industry_value_added"]].mean().reset_index()
    print("Yearly averages:")
    print(yearly_avg.head(), "\n")
    yearly_avg.to_csv("Outputs/tables/yearly_averages.csv", index=False)
    
    # Country-level averages
    country_avg = df.groupby("country_code")[["electricity_price", "industry_value_added"]].mean().reset_index()
    print("Country-level averages:")
    print(country_avg.head(), "\n")
    country_avg.to_csv("Outputs/tables/country_averages.csv", index=False)
    
    return df

# -----------------------------
# SECTION 3: PLOTTING ANALYSIS
# -----------------------------
def run_section_3(df: pd.DataFrame):
    print("\n----- SECTION 3: VISUAL ANALYSIS -----\n")
    
    # ---- Electricity price over years ----
    plt.figure(figsize=(10,6))
    df.groupby("year")["electricity_price"].mean().plot()
    plt.title("Average Electricity Price Over Years")
    plt.xlabel("Year")
    plt.ylabel("Electricity Price")
    plt.tight_layout()
    plt.savefig("Outputs/figures/electricity_price_over_years.png")
    plt.close()
    
    # ---- Industry value added over years ----
    plt.figure(figsize=(10,6))
    df.groupby("year")["industry_value_added"].mean().plot(color="orange")
    plt.title("Average Industry Value Added Over Years")
    plt.xlabel("Year")
    plt.ylabel("Industry Value Added")
    plt.tight_layout()
    plt.savefig("Outputs/figures/industry_value_added_over_years.png")
    plt.close()
    
    # ---- Scatter plot: electricity vs industry ----
def run_section_3(df: pd.DataFrame):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    sns.regplot(
        data=df,
        x="electricity_price",
        y="industry_value_added",
        scatter_kws={"alpha": 0.5, "s": 20},
        line_kws={"color": "red"}
    )
    plt.title("Electricity Price vs Industry Value Added")
    plt.xlabel("Electricity Price")
    plt.ylabel("Industry Value Added")
    plt.tight_layout()
    plt.savefig("Outputs/figures/electricity_vs_industry.png")
    plt.close()
    
print("Plots saved to Outputs/figures/")

    # -----------------------------
# ADDITIONAL PLOTS: Histogram & Bar Chart
# -----------------------------
def run_section_3_extra(df: pd.DataFrame):
    print("\n----- SECTION 3 EXTRA: HISTOGRAM & BAR CHART -----\n")

    # ---- Histogram of electricity prices ----
    plt.figure(figsize=(10,6))
    df["electricity_price"].hist(bins=30, color="green", edgecolor="black")
    plt.title("Histogram of Electricity Prices")
    plt.xlabel("Electricity Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("Outputs/figures/electricity_price_histogram.png")
    plt.close()

    # ---- Bar chart: top 10 countries by average industry value added ----
    top_countries = (
        df.groupby("country_code")["industry_value_added"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    plt.figure(figsize=(12,6))
    top_countries.plot(kind="bar", color="purple")
    plt.title("Top 10 Countries by Average Industry Value Added")
    plt.xlabel("Country Code")
    plt.ylabel("Average Industry Value Added")
    plt.tight_layout()
    plt.savefig("Outputs/figures/top10_industry_value_bar.png")
    plt.close()

    print("Extra plots saved to Outputs/figures/")


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("Loading merged dataset...")
    merged_data = pd.read_csv("Processed data/merged_energy_industry.csv")
    print("Dataset loaded successfully.\n")
    
    merged_data = run_section_1(merged_data)
    merged_data = run_section_2(merged_data)
    run_section_3(merged_data)
    run_section_3_extra(merged_data)

    # ---- CONTROL FLAGS ----
    RUN_SECTION_4 = False   # ← THIS stops regeneration
    RUN_SECTION_5 = False

    if RUN_SECTION_4:
        run_section_4(merged_data)

    if RUN_SECTION_5:
        run_section_5(merged_data)

    print("\nAnalysis pipeline completed successfully.")


# =========================
# SECTION 4: CORRELATION & RELATIONSHIPS
# =========================

def run_section_4(df: pd.DataFrame):
    """
    Section 4 analysis: Correlations and regression insights
    Saves correlation tables and scatter plots with regression lines to Outputs/figures
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    import os

    print("\n----- SECTION 4: CORRELATION & RELATIONSHIPS -----")

    # Ensure output folder exists
    fig_dir = "Outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Compute correlation
    corr = df[["electricity_price", "industry_value_added"]].corr()
    print("Correlation matrix:")
    print(corr)
    
    # Save correlation table
    corr.to_csv("Outputs/tables/correlation_table.csv", index=True)


    # Pearson correlation test
    pearson_coef, p_value = pearsonr(df["electricity_price"], df["industry_value_added"])
    print(f"\nPearson correlation coefficient: {pearson_coef:.3f}, p-value: {p_value:.5f}")

    # Scatter plot with regression line
    plt.figure(figsize=(8,6))
    sns.regplot(
        data=df,
        x="electricity_price",
        y="industry_value_added",
        scatter_kws={'s':20, 'alpha':0.6},
        line_kws={'color':'red'}
    )
    plt.title("Electricity Price vs Industry Value Added")
    plt.xlabel("Electricity Price (US cents per kWh)")
    plt.ylabel("Industry Value Added")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/scatter_electricity_vs_industry.png")
    plt.close()
    print(f"Saved scatter plot: {fig_dir}/scatter_electricity_vs_industry.png")

# =========================
# SECTION 5: TIME SERIES / YEARLY TRENDS
# =========================

def run_section_5(df: pd.DataFrame):
    """
    Section 5 analysis: Trends over years
    Generates line plots for electricity price and industry value added over time
    Saves figures to Outputs/figures
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    print("\n----- SECTION 5: TIME SERIES / YEARLY TRENDS -----")

    fig_dir = "Outputs/figures"
    os.makedirs(fig_dir, exist_ok=True)

    # Aggregate by year
    yearly_avg = df.groupby("year")[["electricity_price", "industry_value_added"]].mean().reset_index()
    print("Yearly averages head:")
    print(yearly_avg.head())

    # Line plot: electricity price over time
    plt.figure(figsize=(10,6))
    sns.lineplot(data=yearly_avg, x="year", y="electricity_price", marker="o")
    plt.title("Average Electricity Price Over Years")
    plt.ylabel("Electricity Price (US cents per kWh)")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_electricity_price_over_years.png")
    plt.close()
    print(f"Saved line plot: {fig_dir}/line_electricity_price_over_years.png")

    # Line plot: industry value added over time
    plt.figure(figsize=(10,6))
    sns.lineplot(data=yearly_avg, x="year", y="industry_value_added", marker="o", color="orange")
    plt.title("Average Industry Value Added Over Years")
    plt.ylabel("Industry Value Added")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/line_industry_value_over_years.png")
    plt.close()
    print(f"Saved line plot: {fig_dir}/line_industry_value_over_years.png")

    # Histogram of electricity prices
    plt.figure(figsize=(8,6))
    sns.histplot(df["electricity_price"], bins=30, kde=True, color="green")
    plt.title("Distribution of Electricity Prices")
    plt.xlabel("Electricity Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/hist_electricity_price.png")
    plt.close()
    print(f"Saved histogram: {fig_dir}/hist_electricity_price.png")

    # Bar chart: mean industry value added per year
    plt.figure(figsize=(10,6))
    sns.barplot(data=yearly_avg, x="year", y="industry_value_added", palette="Blues_d")
    plt.title("Mean Industry Value Added per Year")
    plt.ylabel("Industry Value Added")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/bar_industry_value_per_year.png")
    plt.close()
    print(f"Saved bar chart: {fig_dir}/bar_industry_value_per_year.png")







