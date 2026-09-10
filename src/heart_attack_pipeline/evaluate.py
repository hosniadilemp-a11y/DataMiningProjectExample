"""
Cross-validation and evaluation engine.
Executes 5-fold Stratified Cross Validation, leak-free balancing,
calculates mean and std across Accuracy, Precision, Recall, F1, ROC-AUC, and PR-AUC/AP,
and collects curves and confusion matrices.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from .preprocess import HeartDataPreprocessor


@dataclass
class MetricSummary:
    mean: float
    std: float
    scores: List[float]

    @property
    def formatted(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f}"


@dataclass
class ModelEvaluationResult:
    model_name: str
    tier: str
    model_type: str
    description: str
    course_note: str
    scenario: str  # 'unbalanced' or 'balanced'
    accuracy: MetricSummary
    precision: MetricSummary
    recall: MetricSummary
    f1_score: MetricSummary
    precision_macro: MetricSummary
    recall_macro: MetricSummary
    f1_macro: MetricSummary
    roc_auc: MetricSummary
    average_precision: MetricSummary
    confusion_matrix_total: List[List[int]]
    roc_curve_data: Dict[str, List[float]]  # 'fpr', 'tpr', 'auc'
    pr_curve_data: Dict[str, List[float]]   # 'precision', 'recall', 'ap'
    fit_time_seconds: float
    feature_importances: Optional[Dict[str, float]] = None


class Evaluator:
    """
    Orchestrates 5-Fold Stratified Cross-Validation on all models
    for both Unbalanced and Balanced scenarios.
    """

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def evaluate_model_on_folds(
        self,
        model_name: str,
        meta: Dict[str, Any],
        X_df: pd.DataFrame,
        y: pd.Series,
        preprocessor: HeartDataPreprocessor,
        apply_balancing: bool = False,
    ) -> ModelEvaluationResult:
        """
        Runs 5-fold cross-validation with fold-wise preprocessing and optional SMOTE.
        """
        scenario_name = "balanced" if apply_balancing else "unbalanced"
        estimator = meta["estimator"]

        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []
        prec_macro_scores = []
        rec_macro_scores = []
        f1_macro_scores = []
        roc_auc_scores = []
        ap_scores = []

        total_cm = np.zeros((2, 2), dtype=int)
        y_true_all = []
        y_prob_all = []

        start_time = time.time()
        feature_names = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_df, y)):
            X_train_fold = X_df.iloc[train_idx]
            y_train_fold = y.iloc[train_idx].to_numpy()

            X_val_fold = X_df.iloc[val_idx]
            y_val_fold = y.iloc[val_idx].to_numpy()

            # Preprocess strictly on training fold
            fold_preproc = preprocessor.build_transformer(
                preprocessor.profile.numeric_features,
                preprocessor.profile.categorical_features,
            )
            X_train_trans = fold_preproc.fit_transform(X_train_fold)
            X_val_trans = fold_preproc.transform(X_val_fold)

            # Apply balancing if requested (training fold ONLY)
            if apply_balancing:
                X_train_fit, y_train_fit, _ = HeartDataPreprocessor.balance_training_fold(
                    X_train_trans, y_train_fold, random_state=self.random_state + fold_idx
                )
            else:
                X_train_fit, y_train_fit = X_train_trans, y_train_fold

            # Clone and fit
            clf = clone(estimator)
            clf.fit(X_train_fit, y_train_fit)

            # Predict
            y_pred = clf.predict(X_val_trans)

            # Probability / Decision function for ROC & PR
            if hasattr(clf, "predict_proba"):
                try:
                    probs = clf.predict_proba(X_val_trans)
                    if probs.shape[1] == 2:
                        y_prob = probs[:, 1]
                    else:
                        y_prob = probs[:, 0]
                except Exception:
                    y_prob = y_pred.astype(float)
            elif hasattr(clf, "decision_function"):
                dec = clf.decision_function(X_val_trans)
                # Min-max normalize to [0, 1] range for curve rendering
                dec_min, dec_max = dec.min(), dec.max()
                if dec_max > dec_min:
                    y_prob = (dec - dec_min) / (dec_max - dec_min)
                else:
                    y_prob = y_pred.astype(float)
            else:
                y_prob = y_pred.astype(float)

            # Record predictions
            y_true_all.extend(y_val_fold)
            y_prob_all.extend(y_prob)
            total_cm += confusion_matrix(y_val_fold, y_pred, labels=[0, 1])

            # Calculate fold metrics
            acc_scores.append(accuracy_score(y_val_fold, y_pred))
            prec_scores.append(
                precision_score(y_val_fold, y_pred, zero_division=0)
            )
            rec_scores.append(
                recall_score(y_val_fold, y_pred, zero_division=0)
            )
            f1_scores.append(
                f1_score(y_val_fold, y_pred, zero_division=0)
            )
            prec_macro_scores.append(
                precision_score(y_val_fold, y_pred, average="macro", zero_division=0)
            )
            rec_macro_scores.append(
                recall_score(y_val_fold, y_pred, average="macro", zero_division=0)
            )
            f1_macro_scores.append(
                f1_score(y_val_fold, y_pred, average="macro", zero_division=0)
            )

            try:
                roc_auc_scores.append(roc_auc_score(y_val_fold, y_prob))
            except Exception:
                roc_auc_scores.append(0.5)

            try:
                ap_scores.append(average_precision_score(y_val_fold, y_prob))
            except Exception:
                ap_scores.append(float(np.mean(y_val_fold)))

        elapsed_time = time.time() - start_time

        # Curves over all out-of-fold predictions
        y_true_arr = np.array(y_true_all)
        y_prob_arr = np.array(y_prob_all)

        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)

        # Downsample curve points for compact JSON / HTML embedding
        step_roc = max(1, len(fpr) // 50)
        step_pr = max(1, len(recall_curve) // 50)

        roc_dict = {
            "fpr": [round(float(x), 4) for x in fpr[::step_roc]] + [1.0],
            "tpr": [round(float(x), 4) for x in tpr[::step_roc]] + [1.0],
            "auc": round(float(np.mean(roc_auc_scores)), 4),
        }

        pr_dict = {
            "recall": [round(float(x), 4) for x in recall_curve[::step_pr]],
            "precision": [round(float(x), 4) for x in precision_curve[::step_pr]],
            "ap": round(float(np.mean(ap_scores)), 4),
        }

        # Feature importances if available
        feat_importances = None
        # Fit once on full dataset to extract top global feature importances
        if hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_"):
            try:
                clf_full = clone(estimator)
                X_full_trans, full_feat_names = preprocessor.fit_transform(X_df)
                if apply_balancing:
                    X_full_fit, y_full_fit, _ = HeartDataPreprocessor.balance_training_fold(
                        X_full_trans, y.to_numpy(), random_state=self.random_state
                    )
                else:
                    X_full_fit, y_full_fit = X_full_trans, y.to_numpy()

                clf_full.fit(X_full_fit, y_full_fit)

                if hasattr(clf_full, "feature_importances_"):
                    importances = clf_full.feature_importances_
                    feat_importances = {
                        name: round(float(val), 4)
                        for name, val in zip(full_feat_names, importances)
                    }
                elif hasattr(clf_full, "coef_"):
                    coefs = np.abs(clf_full.coef_[0])
                    feat_importances = {
                        name: round(float(val), 4)
                        for name, val in zip(full_feat_names, coefs)
                    }
            except Exception:
                feat_importances = None

        return ModelEvaluationResult(
            model_name=model_name,
            tier=meta.get("tier", "Modèle"),
            model_type=meta.get("type", "Classifier"),
            description=meta.get("description", ""),
            course_note=meta.get("course_note", ""),
            scenario=scenario_name,
            accuracy=MetricSummary(
                mean=float(np.mean(acc_scores)),
                std=float(np.std(acc_scores)),
                scores=[round(float(s), 4) for s in acc_scores],
            ),
            precision=MetricSummary(
                mean=float(np.mean(prec_scores)),
                std=float(np.std(prec_scores)),
                scores=[round(float(s), 4) for s in prec_scores],
            ),
            recall=MetricSummary(
                mean=float(np.mean(rec_scores)),
                std=float(np.std(rec_scores)),
                scores=[round(float(s), 4) for s in rec_scores],
            ),
            f1_score=MetricSummary(
                mean=float(np.mean(f1_scores)),
                std=float(np.std(f1_scores)),
                scores=[round(float(s), 4) for s in f1_scores],
            ),
            precision_macro=MetricSummary(
                mean=float(np.mean(prec_macro_scores)),
                std=float(np.std(prec_macro_scores)),
                scores=[round(float(s), 4) for s in prec_macro_scores],
            ),
            recall_macro=MetricSummary(
                mean=float(np.mean(rec_macro_scores)),
                std=float(np.std(rec_macro_scores)),
                scores=[round(float(s), 4) for s in rec_macro_scores],
            ),
            f1_macro=MetricSummary(
                mean=float(np.mean(f1_macro_scores)),
                std=float(np.std(f1_macro_scores)),
                scores=[round(float(s), 4) for s in f1_macro_scores],
            ),
            roc_auc=MetricSummary(
                mean=float(np.mean(roc_auc_scores)),
                std=float(np.std(roc_auc_scores)),
                scores=[round(float(s), 4) for s in roc_auc_scores],
            ),
            average_precision=MetricSummary(
                mean=float(np.mean(ap_scores)),
                std=float(np.std(ap_scores)),
                scores=[round(float(s), 4) for s in ap_scores],
            ),
            confusion_matrix_total=total_cm.tolist(),
            roc_curve_data=roc_dict,
            pr_curve_data=pr_dict,
            fit_time_seconds=round(elapsed_time, 2),
            feature_importances=feat_importances,
        )

    def run_benchmark(
        self,
        catalog: Dict[str, Dict],
        X_df: pd.DataFrame,
        y: pd.Series,
        preprocessor: HeartDataPreprocessor,
    ) -> Dict[str, Dict[str, ModelEvaluationResult]]:
        """
        Executes benchmark for both Unbalanced and Balanced scenarios.
        Returns:
            {
               "unbalanced": {model_name: ModelEvaluationResult, ...},
               "balanced": {model_name: ModelEvaluationResult, ...}
            }
        """
        results = {"unbalanced": {}, "balanced": {}}

        print(f"\n{'='*75}")
        print(f"  DÉMARRAGE DU BENCHMARK 5-FOLD CROSS-VALIDATION")
        print(f"  Dataset: {len(y)} échantillons | Ratio Déséquilibre: {preprocessor.profile.imbalance_ratio}:1")
        print(f"{'='*75}\n")

        for scenario_key, apply_balancing in [("unbalanced", False), ("balanced", True)]:
            scenario_title = "DONNÉES ÉQUILIBRÉES (SMOTE)" if apply_balancing else "DONNÉES NON ÉQUILIBRÉES (BRUTES)"
            print(f"\n--- [SCÉNARIO] {scenario_title} ---")

            for name, meta in catalog.items():
                print(f"  ➜ Évaluation: {name:<30} [{meta['tier']}] ...", end="", flush=True)
                res = self.evaluate_model_on_folds(
                    model_name=name,
                    meta=meta,
                    X_df=X_df,
                    y=y,
                    preprocessor=preprocessor,
                    apply_balancing=apply_balancing,
                )
                results[scenario_key][name] = res
                print(f" OK! (Acc: {res.accuracy.formatted} | F1: {res.f1_score.formatted} | ROC-AUC: {res.roc_auc.formatted})")

        return results
