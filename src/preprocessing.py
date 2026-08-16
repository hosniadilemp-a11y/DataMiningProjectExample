"""
Preprocessing and Feature Engineering module for Heart Disease Risk Prediction.
Ensures strict prevention of data leakage via sklearn ColumnTransformers and Pipelines.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from src.utils import (
    load_raw_data, DATA_PROCESSED, MODELS_DIR,
    TARGET_COL, ID_COL, NUMERICAL_COLS, CATEGORICAL_COLS, BOOLEAN_COLS
)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer clinical domain features:
    1. cholesterol_hdl_ratio: Total cholesterol / HDL (Cardiovascular risk indicator)
    2. pulse_pressure: Systolic BP - Diastolic BP
    3. mean_arterial_pressure: Diastolic BP + 1/3 * Pulse Pressure
    """
    df = df.copy()
    df["cholesterol_hdl_ratio"] = df["cholesterol_total"] / np.maximum(df["hdl"], 1.0)
    df["pulse_pressure"] = df["resting_bp_systolic"] - df["resting_bp_diastolic"]
    df["mean_arterial_pressure"] = df["resting_bp_diastolic"] + (df["pulse_pressure"] / 3.0)
    return df

def get_feature_lists():
    """Return updated feature list including engineered features."""
    num_cols = NUMERICAL_COLS + ["cholesterol_hdl_ratio", "pulse_pressure", "mean_arterial_pressure"]
    cat_cols = CATEGORICAL_COLS
    bool_cols = BOOLEAN_COLS
    return num_cols, cat_cols, bool_cols

def _convert_bool_to_float(x):
    return x.astype(float)

def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construct scikit-learn ColumnTransformer for non-leaking feature transformation.
    """
    num_cols, cat_cols, bool_cols = get_feature_lists()
    
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
    ])
    
    bool_pipeline = Pipeline([
        ("convert", FunctionTransformer(_convert_bool_to_float, validate=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("bool", bool_pipeline, bool_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor

def get_feature_names_out(preprocessor: ColumnTransformer) -> list:
    """Extract column names from fitted ColumnTransformer."""
    feature_names = []
    
    # Num
    num_cols = preprocessor.transformers_[0][2]
    feature_names.extend(num_cols)
    
    # Cat
    cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]
    cat_cols = preprocessor.transformers_[1][2]
    cat_encoded = cat_encoder.get_feature_names_out(cat_cols).tolist()
    feature_names.extend(cat_encoded)
    
    # Bool
    bool_cols = preprocessor.transformers_[2][2]
    feature_names.extend(bool_cols)
    
    return feature_names

def prepare_data(save_artifacts: bool = True):
    """
    Full preprocessing execution pipeline:
    1. Load raw dataset
    2. Apply feature engineering
    3. Perform stratified train-test split (80% train, 20% test)
    4. Fit preprocessing pipeline ONLY on train set
    5. Save processed datasets and pipeline artifact
    """
    print("=== Executing Preprocessing & Train/Test Split ===")
    df = load_raw_data()
    
    # Feature engineering
    df_feat = add_engineered_features(df)
    
    X = df_feat.drop(columns=[ID_COL, TARGET_COL])
    y = df_feat[TARGET_COL]
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    print(f"Train Class Balance: {y_train.value_counts().to_dict()}")
    print(f"Test Class Balance: {y_test.value_counts().to_dict()}")
    
    # Preprocessing pipeline fit on train only
    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    feature_names = get_feature_names_out(preprocessor)
    print(f"Total Processed Features: {len(feature_names)}")
    
    X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
    
    if save_artifacts:
        # Save processed dataframes
        X_train_df.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
        X_test_df.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
        y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
        y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)
        
        # Save raw train/test splits (for Streamlit preview / pipeline fitting)
        X_train.to_csv(DATA_PROCESSED / "X_train_raw.csv", index=False)
        X_test.to_csv(DATA_PROCESSED / "X_test_raw.csv", index=False)
        
        # Save preprocessor artifact
        joblib.dump(preprocessor, MODELS_DIR / "preprocessing_pipeline.joblib")
        print(f"Saved preprocessor to {MODELS_DIR / 'preprocessing_pipeline.joblib'}")
        print(f"Saved processed CSVs to {DATA_PROCESSED}")
        
    return X_train, X_test, y_train, y_test, preprocessor, feature_names

if __name__ == "__main__":
    prepare_data(save_artifacts=True)
