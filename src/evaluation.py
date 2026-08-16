"""
Model Evaluation, Comparison, and Serialization Module.
Calculates Accuracy, Precision, Recall, F1-Scores, ROC-AUC, and Confusion Matrices.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import MODELS_DIR, DATA_PROCESSED, save_json

def evaluate_single_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate a single fitted model on both test set and 5-fold cross-validation.
    """
    y_pred = model.predict(X_test)
    
    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_prob)
    else:
        y_prob = None
        roc_auc = 0.50
        
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_bin = f1_score(y_test, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weight = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    # 5-fold CV score on train set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    
    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 (Binary)": round(f1_bin, 4),
        "F1 (Macro)": round(f1_macro, 4),
        "F1 (Weighted)": round(f1_weight, 4),
        "ROC-AUC": round(roc_auc, 4),
        "CV F1 Mean": round(cv_mean, 4),
        "CV F1 Std": round(cv_std, 4),
        "Confusion Matrix": cm
    }
    return metrics, y_pred, y_prob

def evaluate_all_models(best_estimators: Dict[str, Any], optimization_summary: Dict[str, dict],
                        X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[pd.DataFrame, dict, str]:
    """
    Evaluate all 6 models, select the best performing model, and export artifacts.
    """
    print("\n=== Model Benchmark & Evaluation on Test Set ===")
    results_list = []
    full_eval_dict = {}
    
    best_model_name = None
    best_score = -1.0
    
    for name, model in best_estimators.items():
        metrics, y_pred, y_prob = evaluate_single_model(model, X_train, y_train, X_test, y_test)
        
        row = {
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 (Weighted)": metrics["F1 (Weighted)"],
            "ROC-AUC": metrics["ROC-AUC"],
            "CV Score (F1)": f"{metrics['CV F1 Mean']:.4f} ± {metrics['CV F1 Std']:.4f}"
        }
        results_list.append(row)
        
        full_eval_dict[name] = {
            "metrics": metrics,
            "best_params": optimization_summary.get(name, {}).get("best_params", {})
        }
        
        # Select best model based on ROC-AUC + Weighted F1
        combined_score = metrics["F1 (Weighted)"] + metrics["ROC-AUC"]
        if combined_score > best_score:
            best_score = combined_score
            best_model_name = name
            
    comparison_df = pd.DataFrame(results_list)
    print("\n" + comparison_df.to_string(index=False))
    print(f"\nBest Selected Model: {best_model_name}")
    
    # Save artifacts
    comparison_df.to_csv(DATA_PROCESSED / "model_comparison.csv", index=False)
    save_json(full_eval_dict, MODELS_DIR / "model_evaluation_metrics.json")
    
    # Save best model
    best_estimator = best_estimators[best_model_name]
    joblib.dump(best_estimator, MODELS_DIR / "best_model.joblib")
    
    # Save metadata
    metadata = {
        "best_model_name": best_model_name,
        "best_model_params": optimization_summary.get(best_model_name, {}).get("best_params", {}),
        "test_metrics": full_eval_dict[best_model_name]["metrics"],
        "dataset_records": len(X_train) + len(X_test),
        "dataset_features": X_train.shape[1]
    }
    save_json(metadata, MODELS_DIR / "model_metadata.json")
    
    print(f"Saved best model ({best_model_name}) to {MODELS_DIR / 'best_model.joblib'}")
    return comparison_df, full_eval_dict, best_model_name
