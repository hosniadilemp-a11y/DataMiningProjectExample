#!/usr/bin/env python3
"""
Master Report Asset Generator Script.
Executes statistical hypothesis testing, multi-perspective feature importance analysis,
key findings synthesis, and exports publication-ready figures to docs/figures/.

Usage:
    python scripts/generate_report_assets.py
"""

import pandas as pd
from src.statistical_tests import run_hypothesis_suite
from src.feature_analysis import compute_multi_perspective_importance
from src.findings import generate_key_findings
from src.visualization import (
    plot_target_distribution,
    plot_correlation_matrix,
    plot_feature_distributions,
    plot_bivariate_panel,
    plot_pca_scatter,
    plot_feature_importance_comparison
)

def run_all_asset_generation():
    print("=========================================================")
    print("GENERATING REPRODUCIBLE REPORT & DASHBOARD ASSETS")
    print("=========================================================")
    
    # 1. Run Statistical Suite
    print("\n[1/4] Running Statistical Hypothesis Suite (FDR & Bonferroni)...")
    run_hypothesis_suite()
    
    # 2. Run Feature Importance Analysis
    print("\n[2/4] Computing Multi-Perspective Feature Importances...")
    compute_multi_perspective_importance()
    
    # 3. Synthesize Key Findings
    print("\n[3/4] Synthesizing Key Findings...")
    generate_key_findings()
    
    # 4. Generate Figures
    print("\n[4/4] Generating Publication Figures in docs/figures/...")
    plot_target_distribution(save=True)
    plot_correlation_matrix(save=True)
    plot_feature_distributions(save=True)
    plot_bivariate_panel(save=True)
    plot_pca_scatter(save=True)
    plot_feature_importance_comparison(save=True)
    
    print("=========================================================")
    print("ALL REPORT & DASHBOARD ASSETS SUCCESSFULLY GENERATED!")
    print("=========================================================")

if __name__ == "__main__":
    run_all_asset_generation()
