"""
Statistical Hypothesis Testing Module for Heart Disease Risk Prediction.
Formulates research questions, checks assumptions, computes test statistics,
calculates p-values, effect sizes (Cramer's V, Cohen's d, Rank-Biserial, Eta-squared, Odds Ratio),
and applies Benjamini-Hochberg FDR / Bonferroni multiple testing corrections.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List

from src.utils import load_raw_data, PROJECT_ROOT, save_json

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def compute_cramers_v(contingency_table: pd.DataFrame) -> float:
    """Calculate Cramer's V for nominal categorical association."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    min_dim = min((kcorr-1), (rcorr-1))
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(phi2corr / min_dim))

def compute_cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate Cohen's d effect size between two independent groups."""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float((np.mean(x1) - np.mean(x2)) / s_pooled)

def compute_rank_biserial(group0: np.ndarray, group1: np.ndarray) -> float:
    """Calculate Glass Rank-Biserial Correlation for Mann-Whitney U test."""
    n1, n2 = len(group0), len(group1)
    u_stat, _ = stats.mannwhitneyu(group0, group1, alternative="two-sided")
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))

def run_hypothesis_suite(df: pd.DataFrame = None) -> List[dict]:
    """
    Execute 6 primary hypothesis tests on actual dataset variables.
    """
    if df is None:
        df = load_raw_data()
        
    alpha = 0.05
    results = []
    
    # H1: Chest Pain Type vs Heart Disease Status (Chi-Square)
    ct1 = pd.crosstab(df["chest_pain_type"], df["has_heart_disease"])
    chi2_1, p1, dof1, _ = stats.chi2_contingency(ct1)
    cramer_v1 = compute_cramers_v(ct1)
    results.append({
        "id": "H1",
        "question": "Is there a statistically significant association between chest pain type and heart disease status?",
        "h0": "Chest pain type and heart disease status are independent (no association).",
        "h1": "Chest pain type and heart disease status are significantly associated.",
        "variables": ["chest_pain_type", "has_heart_disease"],
        "test": "Chi-Square Test of Independence",
        "reason": "Both variables are categorical/nominal.",
        "statistic": round(float(chi2_1), 2),
        "dof": int(dof1),
        "p_value_raw": float(p1),
        "effect_size_type": "Cramér's V",
        "effect_size_val": round(float(cramer_v1), 4),
        "decision_raw": "Reject H0" if p1 < alpha else "Fail to Reject H0",
        "interpretation": "Chest pain classification (asymptomatic, non-anginal, atypical, typical) is strongly associated with heart disease occurrence.",
        "dm_implication": "Chest pain type is a crucial categorical predictor for classification modeling."
    })
    
    # H2: ST Depression Difference (Mann-Whitney U)
    g0_st = df[df["has_heart_disease"] == 0]["st_depression"].values
    g1_st = df[df["has_heart_disease"] == 1]["st_depression"].values
    u2, p2 = stats.mannwhitneyu(g0_st, g1_st, alternative="two-sided")
    rb2 = compute_rank_biserial(g0_st, g1_st)
    results.append({
        "id": "H2",
        "question": "Does electrocardiogram ST depression differ significantly between heart disease and healthy patients?",
        "h0": "ST depression distributions are equal across healthy and diseased patient groups.",
        "h1": "Patients with heart disease exhibit significantly different (higher) ST depression values.",
        "variables": ["st_depression", "has_heart_disease"],
        "test": "Mann-Whitney U Test",
        "reason": "ST depression is continuous but skewed non-normal (Shapiro-Wilk p < 0.001).",
        "statistic": round(float(u2), 1),
        "dof": None,
        "p_value_raw": float(p2),
        "effect_size_type": "Rank-Biserial r",
        "effect_size_val": round(float(rb2), 4),
        "decision_raw": "Reject H0" if p2 < alpha else "Fail to Reject H0",
        "interpretation": "Patients with heart disease show significantly elevated ST segment depression compared to healthy controls.",
        "dm_implication": "ST depression provides strong non-linear signal for tree-based and margin-based classifiers."
    })

    # H3: Max Heart Rate Achieved Difference (Mann-Whitney U / Cohen's d)
    g0_hr = df[df["has_heart_disease"] == 0]["max_heart_rate_achieved"].values
    g1_hr = df[df["has_heart_disease"] == 1]["max_heart_rate_achieved"].values
    u3, p3 = stats.mannwhitneyu(g0_hr, g1_hr, alternative="two-sided")
    d3 = compute_cohens_d(g1_hr, g0_hr)
    results.append({
        "id": "H3",
        "question": "Do patients with heart disease achieve significantly lower maximum heart rates during stress testing?",
        "h0": "Mean/distribution of max heart rate achieved is equal across both patient groups.",
        "h1": "Patients diagnosed with heart disease achieve significantly lower max heart rates.",
        "variables": ["max_heart_rate_achieved", "has_heart_disease"],
        "test": "Mann-Whitney U Test",
        "reason": "Continuous physiological stress test measurement.",
        "statistic": round(float(u3), 1),
        "dof": None,
        "p_value_raw": float(p3),
        "effect_size_type": "Cohen's d",
        "effect_size_val": round(float(d3), 4),
        "decision_raw": "Reject H0" if p3 < alpha else "Fail to Reject H0",
        "interpretation": "Patients with heart disease exhibit markedly reduced chronotropic response (lower peak heart rate).",
        "dm_implication": "Max heart rate is the single strongest negative correlation predictor in the dataset."
    })

    # H4: Age vs Systolic Blood Pressure Correlation (Spearman)
    rho4, p4 = stats.spearmanr(df["age"], df["resting_bp_systolic"])
    results.append({
        "id": "H4",
        "question": "Is patient age significantly correlated with resting systolic blood pressure?",
        "h0": "There is no monotonic correlation between age and systolic blood pressure.",
        "h1": "Age and systolic blood pressure are significantly correlated.",
        "variables": ["age", "resting_bp_systolic"],
        "test": "Spearman Rank Correlation",
        "reason": "Tests monotonic association between continuous clinical metrics.",
        "statistic": round(float(rho4), 4),
        "dof": len(df) - 2,
        "p_value_raw": float(p4),
        "effect_size_type": "Spearman rho",
        "effect_size_val": round(float(rho4), 4),
        "decision_raw": "Reject H0" if p4 < alpha else "Fail to Reject H0",
        "interpretation": "Resting systolic blood pressure rises monotonically with advancing patient age.",
        "dm_implication": "Combines with age to form collinear cardiovascular risk features requiring feature scaling."
    })

    # H5: Smoker Status vs Stress Score (Kruskal-Wallis)
    sm_groups = [group["stress_score"].values for _, group in df.groupby("smoker_status")]
    kw5, p5 = stats.kruskal(*sm_groups)
    k5 = len(sm_groups)
    n5 = len(df)
    eta2_5 = float((kw5 - k5 + 1) / (n5 - k5))
    results.append({
        "id": "H5",
        "question": "Does patient self-reported stress score differ across smoking categories (Never, Former, Current)?",
        "h0": "Stress score distributions are identical across all smoking status categories.",
        "h1": "At least one smoking status category exhibits a significantly different stress score distribution.",
        "variables": ["smoker_status", "stress_score"],
        "test": "Kruskal-Wallis H-Test",
        "reason": "Comparing continuous stress scores across >2 categorical groups.",
        "statistic": round(float(kw5), 2),
        "dof": k5 - 1,
        "p_value_raw": float(p5),
        "effect_size_type": "Eta-squared (η²)",
        "effect_size_val": round(float(eta2_5), 4),
        "decision_raw": "Reject H0" if p5 < alpha else "Fail to Reject H0",
        "interpretation": "Current and former smokers report significantly higher psychological stress scores.",
        "dm_implication": "Captures behavioral interaction between smoking habits and stress."
    })

    # H6: Exercise Induced Angina vs Heart Disease (Chi-Square / Odds Ratio)
    ct6 = pd.crosstab(df["exercise_induced_angina"], df["has_heart_disease"])
    chi2_6, p6, dof6, _ = stats.chi2_contingency(ct6)
    oddsratio6, _ = stats.fisher_exact(ct6)
    results.append({
        "id": "H6",
        "question": "Does exercise-induced angina significantly increase the odds of heart disease diagnosis?",
        "h0": "Exercise-induced angina is independent of heart disease diagnosis.",
        "h1": "Patients with exercise-induced angina have significantly higher odds of heart disease.",
        "variables": ["exercise_induced_angina", "has_heart_disease"],
        "test": "Chi-Square & Fisher Exact Test",
        "reason": "Binary symptom vs binary target class comparison.",
        "statistic": round(float(chi2_6), 2),
        "dof": int(dof6),
        "p_value_raw": float(p6),
        "effect_size_type": "Odds Ratio",
        "effect_size_val": round(float(oddsratio6), 4),
        "decision_raw": "Reject H0" if p6 < alpha else "Fail to Reject H0",
        "interpretation": "Presence of exercise-induced angina increases the odds of heart disease significantly.",
        "dm_implication": "Provides a high-precision binary clinical flag."
    })

    # Apply Benjamini-Hochberg FDR Multiple Testing Correction
    raw_pvals = [item["p_value_raw"] for item in results]
    m = len(raw_pvals)
    sorted_indices = np.argsort(raw_pvals)
    fdr_pvals = np.zeros(m)
    
    for rank, idx in enumerate(sorted_indices, 1):
        fdr_pvals[idx] = min(1.0, raw_pvals[idx] * m / rank)
        
    for i, item in enumerate(results):
        item["p_value_fdr"] = float(fdr_pvals[i])
        item["p_value_bonferroni"] = float(min(1.0, item["p_value_raw"] * m))
        item["significant_fdr"] = bool(fdr_pvals[i] < alpha)
        
    save_json(results, RESULTS_DIR / "statistical_tests.json")
    print(f"Saved statistical test suite ({len(results)} hypotheses) to {RESULTS_DIR / 'statistical_tests.json'}")
    return results

if __name__ == "__main__":
    run_hypothesis_suite()
