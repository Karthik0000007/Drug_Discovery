"""
data_loading.py — Train/val/test splitting with four evaluation-critical split protocols.

Split modes:
  random      — i.i.d. sample-level split (drugs/targets may appear in all sets)
  cold_drug   — test set contains entirely unseen drug entities
  cold_target — test set contains entirely unseen target entities
  cold_both   — test set has BOTH unseen drugs AND unseen targets (hardest)
"""

from __future__ import annotations

import logging
from typing import Tuple, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_SPLITS = ("random", "cold_drug", "cold_target", "cold_both")


# ──────────────────────────────────────────────
# Core splitting
# ──────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    split: str = "random",
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / val / test DataFrames.

    Parameters
    ----------
    df : DataFrame with columns ``smiles, sequence, affinity``
         (and ``drug_id, target_id`` for cold splits).
    split : one of ``'random'``, ``'cold_drug'``, ``'cold_target'``, ``'cold_both'``.
    test_frac, val_frac : approximate fraction of data for test / validation.
    seed : random seed for reproducibility.

    Returns
    -------
    (train_df, val_df, test_df) with zero entity leakage guaranteed for cold splits.
    """
    assert split in VALID_SPLITS, f"Unknown split '{split}'. Choose from {VALID_SPLITS}."
    rng = np.random.RandomState(seed)

    if split == "random":
        return _random_split(df, test_frac, val_frac, rng)
    elif split == "cold_both":
        return _cold_both_split(df, test_frac, val_frac, rng)
    else:
        key = "drug_id" if split == "cold_drug" else "target_id"
        return _cold_single_split(df, key, test_frac, val_frac, rng, split)


# ── Random split ──────────────────────────────

def _random_split(df, test_frac, val_frac, rng):
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# ── Single-entity cold split ─────────────────

def _cold_single_split(df, key, test_frac, val_frac, rng, split_name):
    if key not in df.columns:
        raise ValueError(
            f"To perform '{split_name}' split, the DataFrame must contain column '{key}'."
        )
    groups = df[key].unique().tolist()
    rng.shuffle(groups)
    n_test = max(1, int(len(groups) * test_frac))
    n_val = max(1, int(len(groups) * val_frac))
    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])

    test_mask = df[key].isin(test_groups)
    val_mask = df[key].isin(val_groups)
    train_mask = ~test_mask & ~val_mask

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    verify_no_leakage(train_df, val_df, test_df, split_name)
    return train_df, val_df, test_df


# ── Cold-both split ──────────────────────────

def _cold_both_split(df, test_frac, val_frac, rng):
    """
    Cold-both: test set contains BOTH unseen drugs AND unseen targets.

    Algorithm:
      1. Partition drug IDs into 3 disjoint groups (train/val/test).
      2. Partition target IDs into 3 disjoint groups (train/val/test).
      3. Test  = {(d, t) : d ∈ test_drugs  AND t ∈ test_targets}
      4. Val   = {(d, t) : d ∈ val_drugs   AND t ∈ val_targets}
      5. Train = {(d, t) : d ∈ train_drugs AND t ∈ train_targets}
      6. Cross-group pairs (e.g., train_drug × test_target) are discarded.
    """
    for col in ("drug_id", "target_id"):
        if col not in df.columns:
            raise ValueError(
                f"'cold_both' split requires column '{col}' in the DataFrame."
            )

    # Partition drug IDs
    drugs = df["drug_id"].unique().tolist()
    rng.shuffle(drugs)
    nd_test = max(1, int(len(drugs) * test_frac))
    nd_val = max(1, int(len(drugs) * val_frac))
    test_drugs = set(drugs[:nd_test])
    val_drugs = set(drugs[nd_test : nd_test + nd_val])
    train_drugs = set(drugs[nd_test + nd_val :])

    # Partition target IDs
    targets = df["target_id"].unique().tolist()
    rng.shuffle(targets)
    nt_test = max(1, int(len(targets) * test_frac))
    nt_val = max(1, int(len(targets) * val_frac))
    test_targets = set(targets[:nt_test])
    val_targets = set(targets[nt_test : nt_test + nt_val])
    train_targets = set(targets[nt_test + nt_val :])

    # Build masks
    train_mask = df["drug_id"].isin(train_drugs) & df["target_id"].isin(train_targets)
    val_mask = df["drug_id"].isin(val_drugs) & df["target_id"].isin(val_targets)
    test_mask = df["drug_id"].isin(test_drugs) & df["target_id"].isin(test_targets)

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    discarded = len(df) - len(train_df) - len(val_df) - len(test_df)
    logger.info(
        "cold_both split — train: %d, val: %d, test: %d, discarded (cross-group): %d",
        len(train_df), len(val_df), len(test_df), discarded,
    )

    if len(test_df) == 0:
        raise ValueError(
            "cold_both split produced an empty test set. "
            "Try increasing test_frac or check entity counts."
        )

    verify_no_leakage(train_df, val_df, test_df, "cold_both")
    return train_df, val_df, test_df


# ──────────────────────────────────────────────
# Leakage verification
# ──────────────────────────────────────────────

def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str,
) -> None:
    """Assert that cold-split entity-separation constraints are respected."""

    def _ids(frame: pd.DataFrame, col: str) -> Set[str]:
        return set(frame[col].unique()) if col in frame.columns else set()

    train_drugs = _ids(train_df, "drug_id")
    val_drugs = _ids(val_df, "drug_id")
    test_drugs = _ids(test_df, "drug_id")
    train_targets = _ids(train_df, "target_id")
    val_targets = _ids(val_df, "target_id")
    test_targets = _ids(test_df, "target_id")

    if split_type == "cold_drug":
        assert len(test_drugs & train_drugs) == 0, "Drug leakage: test ∩ train!"
        assert len(val_drugs & test_drugs) == 0, "Drug leakage: val ∩ test!"
    elif split_type == "cold_target":
        assert len(test_targets & train_targets) == 0, "Target leakage: test ∩ train!"
        assert len(val_targets & test_targets) == 0, "Target leakage: val ∩ test!"
    elif split_type == "cold_both":
        assert len(test_drugs & train_drugs) == 0, "Drug leakage: test ∩ train!"
        assert len(test_targets & train_targets) == 0, "Target leakage: test ∩ train!"
        assert len(val_drugs & test_drugs) == 0, "Drug leakage: val ∩ test!"
        assert len(val_targets & test_targets) == 0, "Target leakage: val ∩ test!"

    logger.debug("Leakage verification passed for '%s' split.", split_type)