"""
CLI Runner for the Heart Attack Classification Pipeline & Benchmark.
Executes preprocessing, evaluates Unbalanced vs Balanced (SMOTE),
runs 5-fold cross-validation with std, and generates the interactive HTML report.

Usage:
    python -m heart_attack_pipeline.run_pipeline
    python -m heart_attack_pipeline.run_pipeline --dataset path/to/dataset.csv --output my_report.html
"""

import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

from .preprocess import HeartDataPreprocessor
from .models import get_model_catalog
from .evaluate import Evaluator
from .generate_report import ReportGenerator


def find_default_dataset() -> Path:
    """Finds existing heart dataset in the project workspace."""
    candidates = [
        Path("apps/dataminingproject/data/raw/heart_disease_risk_2026.csv"),
        Path("apps/project-evaluation-manager/data/projects/SAID019/repo/data/raw/heart_disease_risk_2026.csv"),
        Path("apps/project-evaluation-manager/data/projects/SAID020/repo/heart.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Could not locate a default heart attack dataset in the workspace.")


def print_ascii_table(title: str, results_dict: dict):
    """Prints a well-formatted terminal table of model metrics with mean and std."""
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    header = (
        f"{'Modèle':<28} | {'Accuracy':<17} | {'Précision':<17} | {'Rappel':<17} | "
        f"{'F1-Score':<17} | {'ROC-AUC':<17} | {'AP':<17}"
    )
    print(header)
    print(f"{'-'*28}-+-{'-'*17}-+-{'-'*17}-+-{'-'*17}-+-{'-'*17}-+-{'-'*17}-+-{'-'*17}")

    for name, r in results_dict.items():
        print(
            f"{name:<28} | "
            f"{r.accuracy.formatted:<17} | "
            f"{r.precision.formatted:<17} | "
            f"{r.recall.formatted:<17} | "
            f"{r.f1_score.formatted:<17} | "
            f"{r.roc_auc.formatted:<17} | "
            f"{r.average_precision.formatted:<17}"
        )
    print(f"{'='*110}\n")


def export_csv_summary(results: dict, output_csv: Path):
    """Exports tabular summary of both scenarios to a CSV file."""
    rows = []
    for scenario, models in results.items():
        for name, r in models.items():
            rows.append({
                "Scenario": scenario,
                "Model": name,
                "Tier": r.tier,
                "Type": r.model_type,
                "Accuracy_Mean": r.accuracy.mean,
                "Accuracy_Std": r.accuracy.std,
                "Precision_Mean": r.precision.mean,
                "Precision_Std": r.precision.std,
                "Recall_Mean": r.recall.mean,
                "Recall_Std": r.recall.std,
                "F1_Mean": r.f1_score.mean,
                "F1_Std": r.f1_score.std,
                "F1_Macro_Mean": r.f1_macro.mean,
                "F1_Macro_Std": r.f1_macro.std,
                "ROC_AUC_Mean": r.roc_auc.mean,
                "ROC_AUC_Std": r.roc_auc.std,
                "Average_Precision_Mean": r.average_precision.mean,
                "Average_Precision_Std": r.average_precision.std,
                "Fit_Time_Seconds": r.fit_time_seconds,
            })
    df_res = pd.DataFrame(rows)
    df_res.to_csv(output_csv, index=False)
    print(f"📁 Tableau récapitulatif exporté vers: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Heart Attack Classification Benchmark & Data Mining Hierarchy Pipeline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV dataset (default: auto-detected heart attack dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="heart_attack_report.html",
        help="Path for generated HTML report (default: heart_attack_report.html)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of folds for Stratified Cross-Validation (default: 5)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of samples to use for fast demo/validation while preserving class ratio (default: 500, 0 for all)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # 1. Dataset Resolution
    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = find_default_dataset()

    print(f"\n[1/4] 📥 Chargement du dataset depuis: {dataset_path}")
    preprocessor = HeartDataPreprocessor()
    df_raw = preprocessor.load_data(dataset_path)

    # 2. Preparation & Profiling
    sample_limit = None if args.sample_size <= 0 else args.sample_size
    print(f"[2/4] ⚙️  Préparation & Profilage des Données (Échantillon cible: {sample_limit or 'Tous'})...")
    X, y, profile = preprocessor.prepare_raw_data(
        df_raw, sample_size=sample_limit, random_state=args.random_state
    )

    print(f"      • Échantillons conservés : {profile.total_samples}")
    print(f"      • Attributs numériques : {len(profile.numeric_features)} ({', '.join(profile.numeric_features[:5])}...)")
    print(f"      • Attributs catégoriels: {len(profile.categorical_features)} ({', '.join(profile.categorical_features)})")
    print(f"      • Variable cible : '{profile.target_column}'")
    print(f"      • Répartition : {profile.target_distribution}")
    print(f"      • Pourcentages: {profile.target_percentages}")
    print(f"      • Déséquilibré : {'OUI (Ratio: ' + str(profile.imbalance_ratio) + ':1)' if profile.is_imbalanced else 'NON (Équilibré)'}")

    # 3. Model Benchmark & 5-Fold Stratified Cross Validation
    print(f"\n[3/4] 🚀 Entraînement & Benchmark ({args.splits}-Fold Stratified CV)...")
    catalog = get_model_catalog(random_state=args.random_state)
    evaluator = Evaluator(n_splits=args.splits, random_state=args.random_state)

    results = evaluator.run_benchmark(catalog, X, y, preprocessor)

    # Print terminal summaries
    print_ascii_table("RÉSULTATS 5-FOLD CV : DONNÉES NON ÉQUILIBRÉES (BRUTES)", results["unbalanced"])
    print_ascii_table("RÉSULTATS 5-FOLD CV : DONNÉES ÉQUILIBRÉES (SMOTE)", results["balanced"])

    # 4. Generate Interactive HTML Report & CSV Export
    output_html = Path(args.output)
    output_csv = output_html.with_suffix(".csv")

    print(f"[4/4] 📊 Génération de la page HTML interactive...")
    generator = ReportGenerator(profile, results)
    generator.generate_html(output_html)
    export_csv_summary(results, output_csv)

    print(f"\n{'*'*75}")
    print(f"  🎉 PIPELINE EXÉCUTÉ AVEC SUCCÈS !")
    print(f"  🌐 Rapport HTML interactif généré : {output_html.resolve()}")
    print(f"  📄 Données chiffrées CSV exportées : {output_csv.resolve()}")
    print(f"{'*'*75}\n")


if __name__ == "__main__":
    main()
