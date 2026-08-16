"""
Utility functions and configuration constants for Heart Disease Risk Prediction project.
"""

from pathlib import Path
import pandas as pd
import json

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "heart_disease_risk_2026.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

# Ensure directories exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Feature Definitions
TARGET_COL = "has_heart_disease"
ID_COL = "patient_id"

NUMERICAL_COLS = [
    "age",
    "resting_bp_systolic",
    "resting_bp_diastolic",
    "cholesterol_total",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_blood_sugar",
    "hba1c",
    "bmi",
    "resting_heart_rate",
    "max_heart_rate_achieved",
    "st_depression",
    "alcohol_units_per_week",
    "exercise_minutes_per_week",
    "sleep_hours",
    "stress_score",
    "daily_steps",
    "diet_quality_score"
]

CATEGORICAL_COLS = [
    "sex",
    "chest_pain_type",
    "smoker_status"
]

BOOLEAN_COLS = [
    "exercise_induced_angina",
    "family_history",
    "wearable_owner"
]

STUDENT_ROLES = {
    "AISD05": "Data Acquisition, Quality Audit, EDA, Statistical Hypothesis Testing, Feature Engineering",
    "SIAD19": "Preprocessing Pipeline, Model Development (Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, SVM), Optimization & Cross-Validation",
    "RSD20": "Streamlit Web Application, Interactive Dashboards, Risk Predictor Interface, Software Quality & Documentation"
}

def load_raw_data() -> pd.DataFrame:
    """Load raw dataset from data/raw/heart_disease_risk_2026.csv."""
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Raw data file not found at {DATA_RAW}. Please run scripts/download_data.py first.")
    return pd.read_csv(DATA_RAW)

def save_json(data: dict, filepath: Path):
    """Save dictionary to JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: Path) -> dict:
    """Load JSON file to dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
