"""
deepdta.py -- single-file DeepDTA-style baseline for compound-protein affinity (regression).

Usage examples:
  # random split, affinities are already on the correct scale:
  python deepdta.py --data data.csv --epochs 30 --batch 128 --split random

  # if your affinity column is IC50 in nM and you want to convert to pIC50:
  python deepdta.py --data data.csv --epochs 30 --batch 64 --ic50-nanomolar --split random

CSV input format (minimally required columns):
  smiles, sequence, affinity

If you want cold-drug or cold-target splits, include:
  drug_id, target_id
"""

import random
import numpy as np
import torch

# ---------------------------
# Utilities: metrics & seeds
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def pearsonr_np(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    denom = np.sqrt((yt ** 2).sum() * (yp ** 2).sum())
    return float( (yt * yp).sum() / denom ) if denom != 0 else 0.0

def spearmanr_np(y_true, y_pred):
    # simple rank correlation without scipy
    def rank(x):
        # average ranks for ties
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        # convert 0..n-1 -> 1..n
        return ranks + 1.0
    r_true = rank(y_true)
    r_pred = rank(y_pred)
    return pearsonr_np(r_true, r_pred)

def concordance_index(y_true, y_pred):
    # pairwise concordance: O(n^2), OK for validation/test sets typically
    n = 0
    h_sum = 0.0
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    N = len(y_true)
    for i in range(N):
        for j in range(i+1, N):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            # concordant if predicted order matches true order
            if (y_pred[i] > y_pred[j] and y_true[i] > y_true[j]) or \
               (y_pred[i] < y_pred[j] and y_true[i] < y_true[j]):
                h_sum += 1.0
            elif y_pred[i] == y_pred[j]:
                h_sum += 0.5
    return float(h_sum / n) if n > 0 else 0.0
