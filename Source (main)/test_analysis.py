#Unit tests
# This part of the code is going to be running unit tests to ensure the Analysis functions are working as expected.

# ============================================================
# test_analysis.py
# Unit tests for Analysis.py
# ============================================================

import os
import pandas as pd
import numpy as np
import pytest
from scipy.stats import pearsonr

# Import your analysis functions
from Analysis import (
    run_section_1,
    run_section_2,
    run_section_3,
    run_section_3_extra,
    run_section_4,
    run_section_5
)

# -----------------------------
# FIXTURES
# -----------------------------
@pytest.fixture
def sample_df():
    """Small sample dataset for testing analysis functions"""
    return pd.DataFrame({
        "country": ["US", "UK", "DE", "FR"],
        "country_code": ["US", "UK", "DE", "FR"],
        "year": [2020, 2021, 2022, 2022],
        "electricity_price": [10.5, 11.0, 9.8, 12.0],
        "industry_value_added": [500, 520, 510, 530]
    })

@pytest.fixture
def empty_df():
    """Empty dataframe edge case"""
    return pd.DataFrame(columns=["country","country_code","year","electricity_price","industry_value_added"])

# -----------------------------
# HELPER FUNCTION
# -----------------------------
def ensure_output_dirs():
    os.makedirs("Outputs/tables", exist_ok=True)
    os.makedirs("Outputs/figures", exist_ok=True)

# -----------------------------
# SECTION 1 TESTS
# -----------------------------
def test_section_1_runs(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    df_out = run_section_1(sample_df)
    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape[0] == 4
    assert os.path.exists("Outputs/tables/descriptive_statistics_overall.csv")

def test_section_1_empty(empty_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    df_out = run_section_1(empty_df)
    assert isinstance(df_out, pd.DataFrame)
    assert os.path.exists("Outputs/tables/descriptive_statistics_overall.csv")

# -----------------------------
# SECTION 2 TESTS
# -----------------------------
def test_section_2_runs(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    df_out = run_section_2(sample_df)
    assert os.path.exists("Outputs/tables/yearly_averages.csv")
    assert os.path.exists("Outputs/tables/country_averages.csv")
    yearly = pd.read_csv("Outputs/tables/yearly_averages.csv")
    assert "electricity_price" in yearly.columns
    assert len(yearly) == 3  # years 2020,2021,2022

def test_section_2_empty(empty_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    df_out = run_section_2(empty_df)
    assert os.path.exists("Outputs/tables/yearly_averages.csv")
    assert os.path.exists("Outputs/tables/country_averages.csv")

# -----------------------------
# SECTION 3 TESTS
# -----------------------------
def test_section_3_scatter_and_extra(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    run_section_3(sample_df)
    run_section_3_extra(sample_df)
    scatter = "Outputs/figures/electricity_vs_industry.png"
    hist = "Outputs/figures/electricity_price_histogram.png"
    bar = "Outputs/figures/top10_industry_value_bar.png"
    for f in [scatter, hist, bar]:
        assert os.path.exists(f)

def test_section_3_empty(empty_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    run_section_3(empty_df)
    run_section_3_extra(empty_df)
    scatter = "Outputs/figures/electricity_vs_industry.png"
    hist = "Outputs/figures/electricity_price_histogram.png"
    bar = "Outputs/figures/top10_industry_value_bar.png"
    for f in [scatter, hist, bar]:
        assert os.path.exists(f)

# -----------------------------
# SECTION 4 TESTS
# -----------------------------
def test_section_4_corr_values(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    run_section_4(sample_df)
    corr_file = "Outputs/tables/correlation_table.csv"
    scatter_file = "Outputs/figures/scatter_electricity_vs_industry.png"
    assert os.path.exists(corr_file)
    assert os.path.exists(scatter_file)
    corr_df = pd.read_csv(corr_file, index_col=0)
    pearson_coef, _ = pearsonr(sample_df["electricity_price"], sample_df["industry_value_added"])
    assert np.isclose(corr_df.loc["electricity_price","industry_value_added"], pearson_coef, atol=1e-6)

def test_section_4_empty(empty_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    # Should not crash on empty df
    run_section_4(empty_df)
    corr_file = "Outputs/tables/correlation_table.csv"
    scatter_file = "Outputs/figures/scatter_electricity_vs_industry.png"
    assert os.path.exists(corr_file)
    assert os.path.exists(scatter_file)

# -----------------------------
# SECTION 5 TESTS
# -----------------------------
def test_section_5_runs(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    run_section_5(sample_df)
    files = [
        "Outputs/figures/line_electricity_price_over_years.png",
        "Outputs/figures/line_industry_value_over_years.png",
        "Outputs/figures/hist_electricity_price.png",
        "Outputs/figures/bar_industry_value_per_year.png"
    ]
    for f in files:
        assert os.path.exists(f)

def test_section_5_empty(empty_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    run_section_5(empty_df)
    files = [
        "Outputs/figures/line_electricity_price_over_years.png",
        "Outputs/figures/line_industry_value_over_years.png",
        "Outputs/figures/hist_electricity_price.png",
        "Outputs/figures/bar_industry_value_per_year.png"
    ]
    for f in files:
        assert os.path.exists(f)

# -----------------------------
# EDGE CASES
# -----------------------------
def test_nan_values(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    ensure_output_dirs()
    df = pd.DataFrame({
        "country": ["US", "UK", "DE"],
        "country_code": ["US", "UK", "DE"],
        "year": [2020, 2021, 2022],
        "electricity_price": [10.5, None, 9.8],
        "industry_value_added": [500, 520, None]
    })
    run_section_1(df)
    run_section_2(df)
    run_section_3(df)
    run_section_3_extra(df)
    run_section_4(df)
    run_section_5(df)
    # Check files exist even with NaNs
    assert os.path.exists("Outputs/tables/descriptive_statistics_overall.csv")
    assert os.path.exists("Outputs/tables/correlation_table.csv")
    assert os.path.exists("Outputs/figures/electricity_vs_industry.png")
    assert os.path.exists("Outputs/figures/electricity_price_histogram.png")
    assert os.path.exists("Outputs/figures/top10_industry_value_bar.png")





