"""
Script to create the 3 mandatory executable Jupyter Notebooks for the project:
1. notebooks/01_eda_and_cleaning.ipynb
2. notebooks/02_preprocessing_encoding.ipynb
3. notebooks/03_model_training_evaluation.ipynb
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def make_cell(cell_type, source):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    }

def create_nb1():
    cells = [
        make_cell("markdown", "# 01 — Exploratory Data Analysis & Quality Audit\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: AISD05*"),
        make_cell("code", [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "from scipy import stats",
            "from pathlib import Path",
            "import kagglehub",
            "",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
        ]),
        make_cell("markdown", "## 1. Kaggle Dataset Acquisition & Schema Audit"),
        make_cell("code", [
            "# KaggleHub official fetch",
            "path = kagglehub.dataset_download('uditjain13/heart-disease-risk-2026')",
            "print('Path to Kaggle dataset files:', path)",
            "",
            "csv_path = Path(path) / 'heart_disease_risk_2026.csv'",
            "df = pd.read_csv(csv_path)",
            "print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')",
            "df.head(5)"
        ]),
        make_cell("markdown", "## 2. Missing Value & Data Quality Check"),
        make_cell("code", [
            "print('Total missing values:', df.isnull().sum().sum())",
            "print('Duplicate rows:', df.duplicated().sum())",
            "print('\\nTarget Class Balance:')",
            "print(df['has_heart_disease'].value_counts(normalize=True))"
        ]),
        make_cell("markdown", "## 3. Exploratory Plots & Correlations"),
        make_cell("code", [
            "fig, ax = plt.subplots(figsize=(6, 4))",
            "sns.countplot(data=df, x='has_heart_disease', palette=['#2b5c8f', '#d95f02'], ax=ax)",
            "ax.set_title('Target Class Distribution (N=9,000)')",
            "plt.show()"
        ]),
        make_cell("markdown", "## 4. Statistical Testing (Mann-Whitney & Chi-Square)"),
        make_cell("code", [
            "# Mann-Whitney U test for ST Depression",
            "g0 = df[df['has_heart_disease'] == 0]['st_depression']",
            "g1 = df[df['has_heart_disease'] == 1]['st_depression']",
            "stat, p = stats.mannwhitneyu(g0, g1)",
            "print(f'ST Depression Mann-Whitney U = {stat:.1f}, p-value = {p:.4e}')"
        ])
    ]
    nb = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(NOTEBOOKS_DIR / "01_eda_and_cleaning.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/01_eda_and_cleaning.ipynb")

def create_nb2():
    cells = [
        make_cell("markdown", "# 02 — Data Preprocessing & Feature Engineering\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: SIAD19*"),
        make_cell("code", [
            "import pandas as pd",
            "import numpy as np",
            "import joblib",
            "from src.preprocessing import prepare_data, add_engineered_features"
        ]),
        make_cell("markdown", "## 1. Domain Feature Engineering"),
        make_cell("code", [
            "df_sample = pd.DataFrame({",
            "    'cholesterol_total': [220, 180], 'hdl': [45, 55],",
            "    'resting_bp_systolic': [140, 120], 'resting_bp_diastolic': [90, 80]",
            "})",
            "feat_sample = add_engineered_features(df_sample)",
            "feat_sample"
        ]),
        make_cell("markdown", "## 2. Execute Leakage-Free Preprocessing Pipeline"),
        make_cell("code", [
            "X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(save_artifacts=True)",
            "print('Processed Feature Vector Length:', len(feature_names))",
            "print('Processed Train Shape:', X_train.shape)"
        ])
    ]
    nb = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(NOTEBOOKS_DIR / "02_preprocessing_encoding.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/02_preprocessing_encoding.ipynb")

def create_nb3():
    cells = [
        make_cell("markdown", "# 03 — Model Training, Tuning & Benchmark Evaluation\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: SIAD19 / RSD20*"),
        make_cell("code", [
            "import pandas as pd",
            "import joblib",
            "from src.utils import MODELS_DIR, DATA_PROCESSED",
            "from src.models import train_and_optimize_all",
            "from src.evaluation import evaluate_all_models"
        ]),
        make_cell("markdown", "## 1. Load Processed Train/Test Sets"),
        make_cell("code", [
            "X_train = pd.read_csv(DATA_PROCESSED / 'X_train.csv')",
            "X_test = pd.read_csv(DATA_PROCESSED / 'X_test.csv')",
            "y_train = pd.read_csv(DATA_PROCESSED / 'y_train.csv').squeeze()",
            "y_test = pd.read_csv(DATA_PROCESSED / 'y_test.csv').squeeze()",
            "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')"
        ]),
        make_cell("markdown", "## 2. Load Model Comparison Benchmark"),
        make_cell("code", [
            "comp_df = pd.read_csv(DATA_PROCESSED / 'model_comparison.csv')",
            "comp_df"
        ]),
        make_cell("markdown", "## 3. Verify Saved Best Classifier"),
        make_cell("code", [
            "best_model = joblib.load(MODELS_DIR / 'best_model.joblib')",
            "print('Best Model Artifact:', type(best_model))"
        ])
    ]
    nb = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(NOTEBOOKS_DIR / "03_model_training_evaluation.ipynb", "w") as f:
        json.dump(nb, f, indent=2)
    print("Created notebooks/03_model_training_evaluation.ipynb")

if __name__ == "__main__":
    create_nb1()
    create_nb2()
    create_nb3()
