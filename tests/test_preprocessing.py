"""
Unit tests for Preprocessing & Feature Engineering module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import add_engineered_features, prepare_data

def test_feature_engineering():
    """Verify engineered feature additions."""
    sample_df = pd.DataFrame({
        "cholesterol_total": [200.0, 300.0],
        "hdl": [50.0, 30.0],
        "resting_bp_systolic": [130, 150],
        "resting_bp_diastolic": [80, 90]
    })
    feat_df = add_engineered_features(sample_df)
    
    assert "cholesterol_hdl_ratio" in feat_df.columns
    assert "pulse_pressure" in feat_df.columns
    assert "mean_arterial_pressure" in feat_df.columns
    
    assert np.isclose(feat_df["cholesterol_hdl_ratio"].iloc[0], 4.0)
    assert np.isclose(feat_df["pulse_pressure"].iloc[0], 50.0)

def test_preprocessing_pipeline_split():
    """Verify dataset split shapes and feature transformations."""
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(save_artifacts=False)
    
    assert len(X_train) + len(X_test) == 9000
    assert len(X_train) == 7200
    assert len(X_test) == 1800
    assert len(feature_names) == 31
