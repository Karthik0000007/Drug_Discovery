"""
evidential.py — Phase 7: Uncertainty Estimation via Evidential Deep Regression.

Implements:
  - EvidentialRegressionHead: outputs (mu, v, alpha, beta) for Normal-Inverse-Gamma prior
  - evidential_loss: NLL + regularization term penalizing overconfidence
  - Calibration metrics: ECE, coverage probability
  - Cold-start reliability: distance-based reliability scoring,
    uncertainty-error correlation analysis

Mathematical basis:
  y ~ N(mu, sigma^2),  sigma^2 ~ InvGamma(alpha, beta)
  Parameters: mu (mean), v (>0), alpha (>1), beta (>0)
  Epistemic uncertainty: beta / (v * (alpha - 1))
  Aleatoric uncertainty:  beta / (alpha - 1)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Evidential Regression Head
# ─────────────────────────────────────────────────────────────────────────────


class EvidentialRegressionHead(nn.Module):
    """
    Replaces the deterministic regression head with an uncertainty-aware output.

    Outputs four parameters per sample: (mu, v, alpha, beta)
    - mu: predicted affinity (mean)
    - v > 0: virtual evidence for the mean
    - alpha > 1: shape parameter for inverse-gamma
    - beta > 0: scale parameter for inverse-gamma

    Parameters
    ----------
    input_dim : int
        Dimension of the input feature vector (from encoder concatenation).
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim // 2

        # Separate heads for each parameter
        self.mu_head = nn.Linear(self.out_dim, 1)
        self.v_head = nn.Linear(self.out_dim, 1)       # softplus → > 0
        self.alpha_head = nn.Linear(self.out_dim, 1)    # softplus + 1 → > 1
        self.beta_head = nn.Linear(self.out_dim, 1)     # softplus → > 0

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim) feature vector

        Returns
        -------
        dict with keys: 'mean', 'v', 'alpha', 'beta', 'uncertainty', 'epistemic', 'aleatoric'
        """
        h = self.shared(x)

        mu = self.mu_head(h).squeeze(-1)              # (B,)
        v = F.softplus(self.v_head(h)).squeeze(-1)     # (B,) > 0
        alpha = F.softplus(self.alpha_head(h)).squeeze(-1) + 1.0  # (B,) > 1
        beta = F.softplus(self.beta_head(h)).squeeze(-1)   # (B,) > 0

        # Derived uncertainties
        epistemic = beta / (v * (alpha - 1.0).clamp(min=1e-6))
        aleatoric = beta / (alpha - 1.0).clamp(min=1e-6)
        total_uncertainty = epistemic + aleatoric

        return {
            "mean": mu,
            "v": v,
            "alpha": alpha,
            "beta": beta,
            "uncertainty": total_uncertainty,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
            "raw_params": (v, alpha, beta),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Evidential Loss Function
# ─────────────────────────────────────────────────────────────────────────────


def evidential_loss(
    y_true: torch.Tensor,
    mu: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    reg_weight: float = 0.01,
) -> torch.Tensor:
    """
    Evidential deep regression loss (Amini et al., NeurIPS 2020).

    Components:
    1. NLL of Normal-Inverse-Gamma posterior
    2. Regularization: penalizes overconfidence (high evidence when wrong)

    Key behavior:
    - High error → increase uncertainty (reduce v, reduce alpha)
    - Low error → reduce uncertainty (increase v, increase alpha)

    Parameters
    ----------
    y_true : (B,) true affinity values
    mu : (B,) predicted means
    v : (B,) virtual evidence (> 0)
    alpha : (B,) shape parameter (> 1)
    beta : (B,) scale parameter (> 0)
    reg_weight : float
        Weight for the regularization term

    Returns
    -------
    Scalar loss
    """
    # NLL term
    twoBlambda = 2.0 * beta * (1.0 + v)
    nll = (
        0.5 * torch.log(math.pi / v.clamp(min=1e-8))
        - alpha * torch.log(twoBlambda.clamp(min=1e-8))
        + (alpha + 0.5) * torch.log(
            ((y_true - mu) ** 2 * v + twoBlambda).clamp(min=1e-8)
        )
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Regularization: penalize high evidence when prediction is wrong
    # Encourages model to be uncertain when it makes errors
    error = torch.abs(y_true - mu)
    reg = error * (2.0 * v + alpha)

    loss = nll.mean() + reg_weight * reg.mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Metrics
# ─────────────────────────────────────────────────────────────────────────────


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) for regression.

    Bins samples by predicted uncertainty and compares expected vs actual errors.

    Parameters
    ----------
    y_true : (N,) true values
    y_pred : (N,) predictions
    uncertainties : (N,) predicted uncertainties (e.g. total_uncertainty)
    n_bins : int
        Number of confidence bins

    Returns
    -------
    ece : float
        Expected calibration error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    errors = np.abs(y_true - y_pred)
    N = len(y_true)

    # Bin by uncertainty level
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # ensure max is included

    ece = 0.0
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_error = errors[mask].mean()
        bin_uncertainty = uncertainties[mask].mean()
        ece += mask.sum() / N * abs(bin_error - bin_uncertainty)

    return float(ece)


def coverage_probability(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    confidence: float = 0.90,
) -> float:
    """
    Compute coverage: fraction of true values within predicted intervals.

    Interval: [y_pred - q * uncertainty, y_pred + q * uncertainty]
    where q is chosen from standard normal for the given confidence level.

    Parameters
    ----------
    y_true : (N,) true values
    y_pred : (N,) predicted values
    uncertainties : (N,) predicted uncertainties (std dev scale)
    confidence : float
        Target confidence level (e.g. 0.90)

    Returns
    -------
    coverage : float
        Fraction of true values within the interval
    """
    from scipy import stats

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    # z-score for confidence level
    z = stats.norm.ppf((1 + confidence) / 2)

    lower = y_pred - z * np.sqrt(uncertainties.clip(min=1e-8))
    upper = y_pred + z * np.sqrt(uncertainties.clip(min=1e-8))

    within = ((y_true >= lower) & (y_true <= upper)).astype(float)
    return float(within.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Cold-Start Reliability Metrics
# ─────────────────────────────────────────────────────────────────────────────


def distance_based_reliability(
    test_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute minimum embedding distance from each test sample to training data.

    Higher distance → more novel → likely less reliable prediction.

    Parameters
    ----------
    test_embeddings : (N_test, D) test sample embeddings
    train_embeddings : (N_train, D) training sample embeddings
    metric : str
        Distance metric ('euclidean' or 'cosine')

    Returns
    -------
    distances : (N_test,) minimum distances
    """
    test_embeddings = np.asarray(test_embeddings, dtype=float)
    train_embeddings = np.asarray(train_embeddings, dtype=float)

    if metric == "cosine":
        # Normalize
        test_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=1, keepdims=True) + 1e-8)
        train_norm = train_embeddings / (np.linalg.norm(train_embeddings, axis=1, keepdims=True) + 1e-8)
        # Cosine distance = 1 - cosine similarity
        sim = test_norm @ train_norm.T  # (N_test, N_train)
        distances = 1.0 - sim.max(axis=1)
    else:
        # Euclidean: use chunked computation to avoid memory issues
        chunk_size = 1000
        distances = np.full(len(test_embeddings), float("inf"))
        for i in range(0, len(train_embeddings), chunk_size):
            chunk = train_embeddings[i:i + chunk_size]
            # (N_test, chunk_size)
            diffs = np.linalg.norm(
                test_embeddings[:, None, :] - chunk[None, :, :], axis=2
            )
            distances = np.minimum(distances, diffs.min(axis=1))

    return distances


def uncertainty_error_correlation(
    errors: np.ndarray,
    uncertainties: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze correlation between predicted uncertainty and actual error.

    Expected behavior: higher uncertainty → higher error.

    Parameters
    ----------
    errors : (N,) absolute prediction errors
    uncertainties : (N,) predicted uncertainties

    Returns
    -------
    dict with 'pearson_r', 'spearman_rho', 'auc_oracle'
    """
    errors = np.asarray(errors, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    # Pearson correlation
    if errors.std() > 0 and uncertainties.std() > 0:
        pearson_r = np.corrcoef(errors, uncertainties)[0, 1]
    else:
        pearson_r = 0.0

    # Spearman rank correlation
    from scipy.stats import spearmanr
    try:
        spearman_rho, _ = spearmanr(errors, uncertainties)
    except Exception:
        spearman_rho = 0.0

    # Oracle AUC: if we reject samples with highest uncertainty,
    # does remaining error decrease?
    sorted_idx = np.argsort(uncertainties)
    cumulative_error = np.cumsum(errors[sorted_idx]) / np.arange(1, len(errors) + 1)
    auc_oracle = float(cumulative_error.mean() / (errors.mean() + 1e-8))

    return {
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
        "auc_oracle": auc_oracle,
    }


def compute_all_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    test_embeddings: Optional[np.ndarray] = None,
    train_embeddings: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all uncertainty and calibration metrics.

    Returns
    -------
    dict with ECE, coverage, correlation metrics, and optionally distance reliability.
    """
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    uncertainties = np.asarray(uncertainties)

    metrics = {
        "ece": expected_calibration_error(y_true, y_pred, uncertainties),
        "coverage_90": coverage_probability(y_true, y_pred, uncertainties, 0.90),
        "coverage_95": coverage_probability(y_true, y_pred, uncertainties, 0.95),
        "mean_uncertainty": float(uncertainties.mean()),
        "std_uncertainty": float(uncertainties.std()),
    }

    # Uncertainty-error correlation
    corr = uncertainty_error_correlation(errors, uncertainties)
    metrics.update({f"ue_{k}": v for k, v in corr.items()})

    # Distance-based reliability (if embeddings provided)
    if test_embeddings is not None and train_embeddings is not None:
        distances = distance_based_reliability(test_embeddings, train_embeddings)
        dist_corr = uncertainty_error_correlation(errors, distances)
        metrics.update({
            "dist_mean": float(distances.mean()),
            "dist_error_pearson": dist_corr["pearson_r"],
        })

    return metrics
