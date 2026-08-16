#!/usr/bin/env python3
"""
Master Training Pipeline Script for Heart Disease Risk Prediction Project.
Executes preprocessing, hyperparameter optimization, model benchmarking, artifact export, and plot generation.

Usage:
    python scripts/train_pipeline.py
"""

import pandas as pd
from src.preprocessing import prepare_data
from src.models import train_and_optimize_all
from src.evaluation import evaluate_all_models
from src.visualization import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_feature_importance
)

def run_pipeline():
    print("=========================================================")
    print("STARTING END-TO-END DATA MINING & MODELING PIPELINE")
    print("=========================================================")
    
    # Step 1: Preprocessing & Split
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(save_artifacts=True)
    
    # Load processed X_train and X_test as DataFrames
    from src.utils import DATA_PROCESSED
    X_train_df = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test_df = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    
    # Step 2: Train & Optimize 6 Models
    best_estimators, optimization_summary = train_and_optimize_all(X_train_df, y_train)
    
    # Step 3: Evaluate & Select Best Model
    comparison_df, full_eval_dict, best_model_name = evaluate_all_models(
        best_estimators, optimization_summary, X_train_df, y_train, X_test_df, y_test
    )
    
    # Step 4: Generate Publication Figures
    print("\n=== Generating Figures for Latex Report & Dashboards ===")
    plot_target_distribution(save=True)
    plot_correlation_matrix(save=True)
    plot_feature_distributions(save=True)
    plot_roc_curves(best_estimators, X_test_df, y_test, save=True)
    plot_confusion_matrices(best_estimators, X_test_df, y_test, save=True)
    
    if "Random Forest" in best_estimators:
        plot_feature_importance(best_estimators["Random Forest"], feature_names, save=True)
        
    print("\n=========================================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Best Classifier Identified: {best_model_name}")
    print("=========================================================")

if __name__ == "__main__":
    run_pipeline()
