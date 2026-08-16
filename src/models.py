"""
Machine Learning Model Training, Hyperparameter Optimization, and Model Export.
Implements the 6 required algorithms: Zero-R, KNN, Decision Tree, Random Forest, Naive Bayes, and SVM.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from src.utils import MODELS_DIR, DATA_PROCESSED, save_json

def get_model_param_grids() -> Dict[str, Tuple[Any, dict]]:
    """
    Define model instances and hyperparameter search spaces for 6 classification algorithms.
    Optimized for memory efficiency and execution speed.
    """
    models_and_grids = {
        "Zero-R Baseline": (
            DummyClassifier(strategy="most_frequent"),
            {}
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "n_neighbors": [5, 9, 15],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean"]
            }
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 7, 10],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 4]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=2),
            {
                "n_estimators": [50, 100],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt"]
            }
        ),
        "Naive Bayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-7, 1e-5]
            }
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=42),
            {
                "C": [0.5, 1.0, 5.0],
                "kernel": ["rbf"],
                "gamma": ["scale"]
            }
        )
    }
    return models_and_grids

def train_and_optimize_all(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Dict[str, Any], Dict[str, dict]]:
    """
    Execute 5-fold Stratified GridSearchCV for all 6 models with memory safety.
    """
    print("=== Training & Hyperparameter Tuning (5-Fold Stratified CV) ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models_and_grids = get_model_param_grids()
    
    # Ensure index alignment
    X_train_df = X_train.reset_index(drop=True)
    y_train_ser = y_train.reset_index(drop=True)
    
    # Memory safety sampling for grid search
    if len(X_train_df) > 1000:
        X_train_sub = X_train_df.sample(n=1000, random_state=42)
        y_train_sub = y_train_ser.iloc[X_train_sub.index]
    else:
        X_train_sub = X_train_df
        y_train_sub = y_train_ser
    
    best_estimators = {}
    optimization_summary = {}
    
    for name, (model, grid) in models_and_grids.items():
        print(f"--> Optimizing {name}...")
        if grid:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=grid,
                cv=cv,
                scoring="f1_weighted",
                n_jobs=2,
                refit=True
            )
            grid_search.fit(X_train_sub, y_train_sub)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_cv_score = float(grid_search.best_score_)
            # Fit best model on full training set
            best_model.fit(X_train_df, y_train_ser)
        else:
            model.fit(X_train_df, y_train_ser)
            best_model = model
            best_params = {}
            cv_results = cross_validate(model, X_train_sub, y_train_sub, cv=cv, scoring="f1_weighted")
            best_cv_score = float(cv_results["test_score"].mean())
            
        best_estimators[name] = best_model
        optimization_summary[name] = {
            "best_params": best_params,
            "best_cv_f1_weighted": round(best_cv_score, 4)
        }
        print(f"    Done. Best CV F1-Weighted: {best_cv_score:.4f} | Params: {best_params}")
        
    return best_estimators, optimization_summary
