"""
Script to create the 4 mandatory executable Jupyter Notebooks for the project:
1. notebooks/01_eda_and_cleaning.ipynb
2. notebooks/02_statistical_analysis.ipynb
3. notebooks/03_preprocessing.ipynb
4. notebooks/04_model_training_evaluation.ipynb
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
            "import kagglehub",
            "from pathlib import Path",
            "",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
        ]),
        make_cell("markdown", "## 1. Kaggle Dataset Fetch & Audit"),
        make_cell("code", [
            "path = kagglehub.dataset_download('uditjain13/heart-disease-risk-2026')",
            "print('Kaggle path:', path)",
            "df = pd.read_csv(Path(path) / 'heart_disease_risk_2026.csv')",
            "print(f'Shape: {df.shape[0]} rows, {df.shape[1]} columns')",
            "df.head(5)"
        ]),
        make_cell("markdown", "## 2. Quality & Missing Values Profile"),
        make_cell("code", [
            "print('Missing Values:', df.isnull().sum().sum())",
            "print('Duplicates:', df.duplicated().sum())",
            "print('Target Distribution:')",
            "print(df['has_heart_disease'].value_counts(normalize=True))"
        ])
    ]
    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}
    with open(NOTEBOOKS_DIR / "01_eda_and_cleaning.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

def create_nb2():
    cells = [
        make_cell("markdown", "# 02 — Statistical Analysis & Hypothesis Testing Suite\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: AISD05*"),
        make_cell("code", [
            "import pandas as pd",
            "from src.statistical_tests import run_hypothesis_suite"
        ]),
        make_cell("markdown", "## 1. Run 6 Hypothesis Tests (FDR Corrected)"),
        make_cell("code", [
            "results = run_hypothesis_suite()",
            "stat_df = pd.DataFrame(results)",
            "stat_df[['id', 'test', 'statistic', 'p_value_raw', 'p_value_fdr', 'effect_size_type', 'effect_size_val']]"
        ])
    ]
    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}
    with open(NOTEBOOKS_DIR / "02_statistical_analysis.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

def create_nb3():
    cells = [
        make_cell("markdown", "# 03 — Preprocessing & Feature Engineering Pipeline\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: SIAD19*"),
        make_cell("code", [
            "import pandas as pd",
            "from src.preprocessing import prepare_data"
        ]),
        make_cell("markdown", "## 1. Execute Preprocessing Split"),
        make_cell("code", [
            "X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(save_artifacts=True)",
            "print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')",
            "print('Total features:', len(feature_names))"
        ])
    ]
    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}
    with open(NOTEBOOKS_DIR / "03_preprocessing.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

def create_nb4():
    cells = [
        make_cell("markdown", "# 04 — Model Training, Tuning & Benchmark Evaluation\n**Heart Disease Risk Prediction Reference Project**\n*Simulated Contributor: SIAD19 / RSD20*"),
        make_cell("code", [
            "import pandas as pd",
            "import joblib",
            "from src.utils import MODELS_DIR, DATA_PROCESSED"
        ]),
        make_cell("markdown", "## 1. Load Model Benchmark Results"),
        make_cell("code", [
            "comp_df = pd.read_csv(DATA_PROCESSED / 'model_comparison.csv')",
            "comp_df"
        ]),
        make_cell("markdown", "## 2. Load Best Model Artifact"),
        make_cell("code", [
            "best_model = joblib.load(MODELS_DIR / 'best_model.joblib')",
            "print('Best Model Class:', type(best_model))"
        ])
    ]
    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 2}
    with open(NOTEBOOKS_DIR / "04_model_training_evaluation.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    create_nb1()
    create_nb2()
    create_nb3()
    create_nb4()
    print("Created 4 notebooks in notebooks/")
