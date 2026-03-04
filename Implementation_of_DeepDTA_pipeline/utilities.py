"""
utilities.py — Metrics, seed management, and helper functions for CL-DTA.

Metrics implemented:
  MSE, RMSE, MAE, Pearson r, Spearman ρ (average-rank), Concordance Index,
  sampled CI (O(m) approximation), modified r² (r²_m).
"""

import random
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Seed management
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Regression metrics
# ──────────────────────────────────────────────

def mse(y_true, y_pred):
    """Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def pearsonr_np(y_true, y_pred):
    """Pearson correlation coefficient."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt ** 2).sum() * (yp ** 2).sum())
    return float((yt * yp).sum() / denom) if denom != 0 else 0.0


def _average_rank(x):
    """Compute average ranks (ties get the mean of the positions they span)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    n = len(x)
    i = 0
    while i < n:
        j = i
        # find the end of the tie group
        while j < n - 1 and x[order[j]] == x[order[j + 1]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearmanr_np(y_true, y_pred):
    """Spearman rank correlation using proper average ranking for ties."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    r_true = _average_rank(y_true)
    r_pred = _average_rank(y_pred)
    return pearsonr_np(r_true, r_pred)


# ──────────────────────────────────────────────
# Concordance Index
# ──────────────────────────────────────────────

def concordance_index(y_true, y_pred):
    """
    Exact pairwise concordance index — O(n²).
    Acceptable for test sets ≤ ~15 K samples.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = 0
    h_sum = 0.0
    N = len(y_true)
    for i in range(N):
        for j in range(i + 1, N):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
               (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                h_sum += 1.0
            elif y_pred[i] == y_pred[j]:
                h_sum += 0.5
    return float(h_sum / n) if n > 0 else 0.0


def concordance_index_sampled(y_true, y_pred, m: int = 100_000, seed: int = 0):
    """
    Sampled CI approximation — O(m) instead of O(n²).
    Draws *m* random index pairs and computes concordance over them.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    N = len(y_true)
    if N < 2:
        return 0.0
    rng = np.random.RandomState(seed)
    idx_i = rng.randint(0, N, size=m)
    idx_j = rng.randint(0, N, size=m)
    # Re-sample pairs where i == j or true values are tied
    mask = (idx_i != idx_j) & (y_true[idx_i] != y_true[idx_j])
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if len(idx_i) == 0:
        return 0.0
    concordant = (
        ((y_pred[idx_i] > y_pred[idx_j]) & (y_true[idx_i] > y_true[idx_j])) |
        ((y_pred[idx_i] < y_pred[idx_j]) & (y_true[idx_i] < y_true[idx_j]))
    ).astype(float)
    tied = (y_pred[idx_i] == y_pred[idx_j]).astype(float) * 0.5
    return float((concordant + tied).sum() / len(idx_i))


def ci_auto(y_true, y_pred, exact_threshold: int = 15_000, m: int = 100_000):
    """Use exact CI for small sets, sampled CI for large sets."""
    if len(y_true) <= exact_threshold:
        return concordance_index(y_true, y_pred)
    logger.info("Using sampled CI (m=%d) for %d samples.", m, len(y_true))
    return concordance_index_sampled(y_true, y_pred, m=m)


# ──────────────────────────────────────────────
# Modified r² (r²_m)
# ──────────────────────────────────────────────

def r_squared_m(y_true, y_pred):
    """
    Modified r² metric used in KIBA literature.
    r²_m = r² × (1 − √|r² − r₀²|)
    where r₀² is the coefficient of determination from regression through the origin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return 0.0
    # Standard r²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    # r₀²: regression through origin (y_pred vs y_true)
    # y_pred_hat = k * y_true, where k = (y_pred · y_true) / (y_true · y_true)
    k = np.dot(y_pred, y_true) / np.dot(y_true, y_true) if np.dot(y_true, y_true) != 0 else 0
    ss_res_0 = np.sum((y_pred - k * y_true) ** 2)
    ss_tot_pred = np.sum((y_pred - y_pred.mean()) ** 2)
    r0_sq = 1.0 - ss_res_0 / ss_tot_pred if ss_tot_pred != 0 else 0.0
    r_m_sq = r2 * (1.0 - np.sqrt(abs(r2 - r0_sq)))
    return float(r_m_sq)


# ──────────────────────────────────────────────
# Unified metrics dict
# ──────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred):
    """Compute all evaluation metrics and return as a dict."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "ci": ci_auto(y_true, y_pred),
        "pearson_r": pearsonr_np(y_true, y_pred),
        "spearman_rho": spearmanr_np(y_true, y_pred),
        "r_m_squared": r_squared_m(y_true, y_pred),
    }
