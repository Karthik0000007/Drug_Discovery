"""
statistical_analysis.py — Phase 11: Statistical Rigor for Publication.

Implements:
  - Paired t-test / Wilcoxon signed-rank across (folds × seeds)
  - Cohen's d effect size
  - Multiple comparison correction (Bonferroni, FDR)
  - Confidence intervals on metrics (MSE, CI, Pearson, etc.)
  - Model comparison framework

All functions follow publication standards for ML papers at top venues.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Statistical Tests for Model Comparison
# ─────────────────────────────────────────────────────────────────────────────


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Paired t-test between two models' scores across folds/seeds.

    Parameters
    ----------
    scores_a : (N,) scores for model A (e.g. CI values across 5 folds × 3 seeds)
    scores_b : (N,) scores for model B
    alternative : 'two-sided', 'greater', 'less'

    Returns
    -------
    dict with 't_statistic', 'p_value', 'mean_diff', 'std_diff'
    """
    from scipy import stats

    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    diff = scores_a - scores_b
    t_stat, p_val = stats.ttest_rel(scores_a, scores_b, alternative=alternative)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std()),
        "n_samples": len(scores_a),
        "significant_005": bool(p_val < 0.05),
        "significant_001": bool(p_val < 0.01),
    }


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    More robust when assumptions of normality may not hold (small N).

    Parameters
    ----------
    scores_a : (N,) scores for model A
    scores_b : (N,) scores for model B
    alternative : 'two-sided', 'greater', 'less'

    Returns
    -------
    dict with 'statistic', 'p_value'
    """
    from scipy import stats

    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    diff = scores_a - scores_b

    # Handle case where all differences are zero
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": 0.0,
            "significant_005": False,
        }

    try:
        stat, p_val = stats.wilcoxon(
            scores_a, scores_b, alternative=alternative, zero_method="wilcox"
        )
    except ValueError:
        # Fallback if all differences are equal
        stat, p_val = 0.0, 1.0

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "mean_diff": float(diff.mean()),
        "significant_005": bool(p_val < 0.05),
        "significant_001": bool(p_val < 0.01),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Effect Size
# ─────────────────────────────────────────────────────────────────────────────


def cohens_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """
    Cohen's d effect size for paired samples.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 ≤ |d| < 0.5: small
    - 0.5 ≤ |d| < 0.8: medium
    - |d| ≥ 0.8: large

    Parameters
    ----------
    scores_a, scores_b : (N,) paired scores

    Returns
    -------
    d : float
        Cohen's d
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    diff = scores_a - scores_b
    d = diff.mean() / (diff.std(ddof=1) + 1e-10)
    return float(d)


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Comparison Correction
# ─────────────────────────────────────────────────────────────────────────────


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list of float
        Raw p-values from multiple tests
    alpha : float
        Family-wise error rate

    Returns
    -------
    dict with 'corrected_alpha', 'corrected_p_values', 'significant'
    """
    n = len(p_values)
    corrected_alpha = alpha / n
    corrected_p = [min(p * n, 1.0) for p in p_values]
    significant = [p < corrected_alpha for p in p_values]

    return {
        "corrected_alpha": corrected_alpha,
        "corrected_p_values": corrected_p,
        "significant": significant,
        "n_tests": n,
        "n_significant": sum(significant),
    }


def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values
    alpha : float
        False discovery rate threshold

    Returns
    -------
    dict with 'corrected_p_values', 'significant', 'threshold'
    """
    p_arr = np.asarray(p_values)
    n = len(p_arr)
    sorted_idx = np.argsort(p_arr)
    sorted_p = p_arr[sorted_idx]

    # BH procedure
    thresholds = alpha * np.arange(1, n + 1) / n
    max_idx = -1
    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            max_idx = i

    if max_idx < 0:
        threshold = 0.0
        significant = [False] * n
    else:
        threshold = float(thresholds[max_idx])
        significant = [False] * n
        for i in range(max_idx + 1):
            significant[sorted_idx[i]] = True

    # Corrected p-values (Adjusted)
    corrected_p = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    # Enforce monotonicity
    for i in range(n - 2, -1, -1):
        corrected_p[i] = min(corrected_p[i], corrected_p[i + 1])
    # Map back to original order
    result_p = np.empty(n)
    result_p[sorted_idx] = corrected_p

    return {
        "corrected_p_values": result_p.tolist(),
        "significant": significant,
        "threshold": threshold,
        "n_significant": sum(significant),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────


def confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a metric.

    Parameters
    ----------
    values : (N,) metric values across seeds/folds
    confidence : float
        Confidence level (default 0.95)

    Returns
    -------
    (mean, ci_lower, ci_upper)
    """
    from scipy import stats

    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    if n < 2:
        return mean, mean, mean

    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * se

    return mean, mean - margin, mean + margin


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval (non-parametric).

    Parameters
    ----------
    values : (N,) metric values
    confidence : float
    n_bootstrap : int
    seed : int

    Returns
    -------
    (mean, ci_lower, ci_upper)
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)

    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot_means, 100 * alpha))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    mean = float(values.mean())

    return mean, ci_lower, ci_upper


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Model Comparison
# ─────────────────────────────────────────────────────────────────────────────


def compare_models(
    results_a: List[float],
    results_b: List[float],
    model_a_name: str = "proposed",
    model_b_name: str = "baseline",
    metric_name: str = "CI",
    alpha: float = 0.05,
) -> Dict[str, any]:
    """
    Full statistical comparison between two models.

    Parameters
    ----------
    results_a : list of float
        Metric values for model A across seeds/folds
    results_b : list of float
        Metric values for model B
    model_a_name, model_b_name : str
        Model names for reporting
    metric_name : str
        Name of the metric being compared
    alpha : float
        Significance level

    Returns
    -------
    dict with comprehensive comparison results
    """
    a = np.asarray(results_a, dtype=float)
    b = np.asarray(results_b, dtype=float)

    # Basic statistics
    mean_a, ci_low_a, ci_high_a = confidence_interval(a)
    mean_b, ci_low_b, ci_high_b = confidence_interval(b)

    # Tests
    ttest = paired_ttest(a, b)
    wilcox = wilcoxon_test(a, b)
    d = cohens_d(a, b)

    return {
        "metric": metric_name,
        model_a_name: {
            "mean": mean_a,
            "std": float(a.std()),
            "ci_95": (ci_low_a, ci_high_a),
        },
        model_b_name: {
            "mean": mean_b,
            "std": float(b.std()),
            "ci_95": (ci_low_b, ci_high_b),
        },
        "difference": {
            "mean": float(a.mean() - b.mean()),
            "pct_improvement": float((a.mean() - b.mean()) / (abs(b.mean()) + 1e-10) * 100),
        },
        "paired_ttest": ttest,
        "wilcoxon": wilcox,
        "cohens_d": d,
        "effect_size": effect_size_interpretation(d),
        "conclusion": (
            f"{model_a_name} is significantly better (p<{alpha})"
            if ttest["p_value"] < alpha and a.mean() > b.mean()
            else f"No significant difference at α={alpha}"
        ),
    }
