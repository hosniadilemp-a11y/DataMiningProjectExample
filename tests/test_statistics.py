"""
Unit tests for Statistical Hypothesis Testing Module.
"""

import pytest
import pandas as pd
from src.statistical_tests import run_hypothesis_suite, compute_cramers_v, compute_cohens_d

def test_cramers_v():
    """Verify Cramér's V calculation."""
    cont_df = pd.DataFrame([[100, 20], [30, 90]])
    v = compute_cramers_v(cont_df)
    assert 0.0 <= v <= 1.0

def test_cohens_d():
    """Verify Cohen's d calculation."""
    g1 = pd.Series([10.0, 12.0, 11.0, 13.0]).values
    g2 = pd.Series([5.0, 6.0, 4.0, 5.5]).values
    d = compute_cohens_d(g1, g2)
    assert d > 0.0

def test_run_hypothesis_suite():
    """Verify hypothesis suite returns 6 structured hypotheses."""
    results = run_hypothesis_suite()
    assert len(results) == 6
    for item in results:
        assert "id" in item
        assert "p_value_raw" in item
        assert "p_value_fdr" in item
        assert "effect_size_val" in item
