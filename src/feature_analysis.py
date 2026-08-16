"""
Multi-Perspective Feature Importance Analysis Module.
Compares Statistical Association, Mutual Information, Tree Gini Importance, and Permutation Importance.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from typing import Dict, Any

from src.utils import PROJECT_ROOT, MODELS_DIR, DATA_PROCESSED, save_json

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_multi_perspective_importance() -> pd.DataFrame:
    """
    Compute and normalize feature importance across 4 distinct criteria.
    """
    print("=== Computing Multi-Perspective Feature Importances ===")
    
    # Load processed data
    X_train_df = pd.read_csv(DATA_PROCESSED / "X_train.csv")
    X_test_df = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    
    feature_names = list(X_train_df.columns)
    
    # 1. Mutual Information
    mi_scores = mutual_info_classif(X_train_df, y_train, random_state=42)
    mi_norm = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
    
    # 2. Random Forest Gini Importance
    best_model = joblib.load(MODELS_DIR / "best_model.joblib")
    
    # Check if RF or fit an RF for tree-based comparison
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=2)
    rf.fit(X_train_df, y_train)
    gini_imp = rf.feature_importances_
    gini_norm = gini_imp / np.max(gini_imp)
    
    # 3. Permutation Importance on Best Model
    perm_res = permutation_importance(best_model, X_test_df, y_test, n_repeats=5, random_state=42, n_jobs=2)
    perm_imp = np.maximum(0, perm_res.importances_mean)
    perm_norm = perm_imp / np.max(perm_imp) if np.max(perm_imp) > 0 else perm_imp
    
    # 4. Statistical Correlation Magnitude
    corrs = X_train_df.apply(lambda col: abs(np.corrcoef(col, y_train)[0, 1])).values
    corr_norm = corrs / np.max(corrs)
    
    # Combine into DataFrame
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Mutual Information": np.round(mi_scores, 4),
        "MI Normalized": np.round(mi_norm, 4),
        "Tree Gini Importance": np.round(gini_imp, 4),
        "Gini Normalized": np.round(gini_norm, 4),
        "Permutation Importance": np.round(perm_imp, 4),
        "Permutation Normalized": np.round(perm_norm, 4),
        "Correlation Magnitude": np.round(corrs, 4),
        "Correlation Normalized": np.round(corr_norm, 4)
    })
    
    # Compute Consensus Composite Score
    fi_df["Consensus Score"] = np.round(
        (fi_df["MI Normalized"] + fi_df["Gini Normalized"] + fi_df["Permutation Normalized"] + fi_df["Correlation Normalized"]) / 4.0, 4
    )
    
    fi_df = fi_df.sort_values(by="Consensus Score", ascending=False).reset_index(drop=True)
    
    # Save artifacts
    fi_df.to_csv(RESULTS_DIR / "feature_importances.csv", index=False)
    save_json(fi_df.to_dict(orient="records"), RESULTS_DIR / "feature_importances.json")
    
    print(f"Saved feature importance comparison to {RESULTS_DIR / 'feature_importances.csv'}")
    return fi_df

if __name__ == "__main__":
    compute_multi_perspective_importance()
