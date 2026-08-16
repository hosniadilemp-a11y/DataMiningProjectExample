"""
Unit tests for Data Acquisition and Validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.utils import DATA_RAW, TARGET_COL, NUMERICAL_COLS, CATEGORICAL_COLS, BOOLEAN_COLS

def test_raw_data_exists():
    """Verify raw dataset file is present."""
    assert DATA_RAW.exists(), f"Raw data file missing at {DATA_RAW}"

def test_raw_data_schema():
    """Verify dataset columns, shape, and zero missing values."""
    df = pd.read_csv(DATA_RAW)
    assert len(df) > 0, "Dataset is empty"
    assert df.shape[1] == 27, f"Expected 27 columns, got {df.shape[1]}"
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' missing"
    assert df.isnull().sum().sum() == 0, "Raw dataset contains missing values"

def test_target_binary():
    """Verify target column is binary (0 or 1)."""
    df = pd.read_csv(DATA_RAW)
    unique_targets = set(df[TARGET_COL].unique())
    assert unique_targets == {0, 1}, f"Unexpected target classes: {unique_targets}"
