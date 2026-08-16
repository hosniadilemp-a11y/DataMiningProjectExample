"""
Unit tests for Machine Learning Models, Optimization, and Artifact Loading.
"""

import pytest
import pandas as pd
import joblib
from src.utils import MODELS_DIR, DATA_PROCESSED

def test_model_artifacts_exist():
    """Verify best model and pipeline artifacts were saved."""
    best_model_path = MODELS_DIR / "best_model.joblib"
    pipeline_path = MODELS_DIR / "preprocessing_pipeline.joblib"
    
    assert best_model_path.exists(), "best_model.joblib is missing"
    assert pipeline_path.exists(), "preprocessing_pipeline.joblib is missing"

def test_model_predict_sample():
    """Verify saved best model and pipeline can load and predict on sample input."""
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.joblib")
    
    X_raw_test = pd.read_csv(DATA_PROCESSED / "X_test_raw.csv")
    sample_df = X_raw_test.head(2)
    
    # Preprocess
    proc_data = pipeline.transform(sample_df)
    preds = model.predict(proc_data)
    
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})
