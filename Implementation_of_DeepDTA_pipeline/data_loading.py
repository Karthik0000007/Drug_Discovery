"""
data_loading.py — Train/val/test splitting with four evaluation-critical split protocols.

Split modes:
  random      — i.i.d. sample-level split (drugs/targets may appear in all sets)
  cold_drug   — test set contains entirely unseen drug entities
  cold_target — test set contains entirely unseen target entities
  cold_both   — test set has BOTH unseen drugs AND unseen targets (hardest)

Phase 2 enhancements:
  - SplitInfo class for structured split metadata
  - 5-fold entity-group cross-validation
  - Edge case handling with retry logic
  - Comprehensive logging and leakage verification
"""

from __future__ import annotations

import logging
from typing import Tuple, Set, List, Dict, Optional
from dataclasses import dataclass, asdict
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_SPLITS = ("random", "cold_drug", "cold_target", "cold_both", "cold_pharos")


# ──────────────────────────────────────────────
# Phase 2: SplitInfo class for metadata tracking
# ──────────────────────────────────────────────

@dataclass
class SplitInfo:
    """
    Structured information about a train/val/test split.
    
    Tracks entity counts, sample counts, and provides leakage verification.
    """
    split_type: str
    seed: int
    
    # Sample counts
    n_train: int
    n_val: int
    n_test: int
    n_total: int
    n_discarded: int = 0
    
    # Entity counts (for cold splits)
    train_drugs: Set[str] = None
    val_drugs: Set[str] = None
    test_drugs: Set[str] = None
    train_targets: Set[str] = None
    val_targets: Set[str] = None
    test_targets: Set[str] = None
    
    # Fractions
    train_frac: float = 0.0
    val_frac: float = 0.0
    test_frac: float = 0.0
    retention_rate: float = 1.0  # (train+val+test) / total
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.n_total > 0:
            self.train_frac = self.n_train / self.n_total
            self.val_frac = self.n_val / self.n_total
            self.test_frac = self.n_test / self.n_total
            used = self.n_train + self.n_val + self.n_test
            self.retention_rate = used / self.n_total
    
    def verify_no_leakage(self) -> None:
        """
        Programmatically verify zero entity leakage for cold splits.
        
        Raises
        ------
        AssertionError if any leakage is detected.
        """
        if self.split_type == "random":
            return  # No entity constraints for random split
        
        if self.split_type == "cold_drug":
            assert self.test_drugs is not None and self.train_drugs is not None
            overlap_train = self.test_drugs & self.train_drugs
            overlap_val = self.test_drugs & self.val_drugs if self.val_drugs else set()
            assert len(overlap_train) == 0, (
                f"Drug leakage detected: {len(overlap_train)} drugs in both test and train!"
            )
            assert len(overlap_val) == 0, (
                f"Drug leakage detected: {len(overlap_val)} drugs in both test and val!"
            )
            logger.info("✓ No drug leakage detected (cold_drug split)")
        
        elif self.split_type == "cold_target":
            assert self.test_targets is not None and self.train_targets is not None
            overlap_train = self.test_targets & self.train_targets
            overlap_val = self.test_targets & self.val_targets if self.val_targets else set()
            assert len(overlap_train) == 0, (
                f"Target leakage detected: {len(overlap_train)} targets in both test and train!"
            )
            assert len(overlap_val) == 0, (
                f"Target leakage detected: {len(overlap_val)} targets in both test and val!"
            )
            logger.info("✓ No target leakage detected (cold_target split)")
        
        elif self.split_type == "cold_both":
            assert all([
                self.test_drugs is not None, self.train_drugs is not None,
                self.test_targets is not None, self.train_targets is not None
            ])
            drug_overlap_train = self.test_drugs & self.train_drugs
            drug_overlap_val = self.test_drugs & self.val_drugs if self.val_drugs else set()
            target_overlap_train = self.test_targets & self.train_targets
            target_overlap_val = self.test_targets & self.val_targets if self.val_targets else set()
            
            assert len(drug_overlap_train) == 0, (
                f"Drug leakage detected: {len(drug_overlap_train)} drugs in both test and train!"
            )
            assert len(drug_overlap_val) == 0, (
                f"Drug leakage detected: {len(drug_overlap_val)} drugs in both test and val!"
            )
            assert len(target_overlap_train) == 0, (
                f"Target leakage detected: {len(target_overlap_train)} targets in both test and train!"
            )
            assert len(target_overlap_val) == 0, (
                f"Target leakage detected: {len(target_overlap_val)} targets in both test and val!"
            )
            logger.info("✓ No drug or target leakage detected (cold_both split)")
    
    def summary(self) -> str:
        """Return a human-readable summary of the split."""
        lines = [
            f"Split Type: {self.split_type}",
            f"Seed: {self.seed}",
            f"Samples — Train: {self.n_train} ({self.train_frac:.1%}), "
            f"Val: {self.n_val} ({self.val_frac:.1%}), "
            f"Test: {self.n_test} ({self.test_frac:.1%})",
            f"Total: {self.n_total}, Discarded: {self.n_discarded}, "
            f"Retention: {self.retention_rate:.1%}",
        ]
        
        if self.train_drugs is not None:
            lines.append(
                f"Drugs — Train: {len(self.train_drugs)}, "
                f"Val: {len(self.val_drugs) if self.val_drugs else 0}, "
                f"Test: {len(self.test_drugs)}"
            )
        if self.train_targets is not None:
            lines.append(
                f"Targets — Train: {len(self.train_targets)}, "
                f"Val: {len(self.val_targets) if self.val_targets else 0}, "
                f"Test: {len(self.test_targets)}"
            )
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        # Convert sets to lists for JSON serialization
        for key in d:
            if isinstance(d[key], set):
                d[key] = list(d[key]) if d[key] is not None else None
        return d
    
    def save(self, path: str) -> None:
        """Save split info to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ──────────────────────────────────────────────
# Core splitting with enhanced error handling
# ──────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    split: str = "random",
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
    min_samples_threshold: int = 100,
    max_retry_attempts: int = 5,
    verify_leakage: bool = True,
    pharos_proteins: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Split *df* into train / val / test DataFrames with enhanced error handling.

    Parameters
    ----------
    df : DataFrame with columns ``smiles, sequence, affinity``
         (and ``drug_id, target_id`` for cold splits).
    split : one of ``'random'``, ``'cold_drug'``, ``'cold_target'``, ``'cold_both'``, ``'cold_pharos'``.
    test_frac, val_frac : approximate fraction of data for test / validation.
    seed : random seed for reproducibility.
    min_samples_threshold : minimum samples required per split (raises error if violated).
    max_retry_attempts : number of retry attempts for cold splits if constraints fail.
    verify_leakage : whether to run leakage verification (recommended: True).
    pharos_proteins : Set of dark protein IDs for cold_pharos split (required if split='cold_pharos').

    Returns
    -------
    (train_df, val_df, test_df, split_info) with zero entity leakage guaranteed for cold splits.
    
    Raises
    ------
    ValueError if split constraints cannot be satisfied after max_retry_attempts.
    """
    assert split in VALID_SPLITS, f"Unknown split '{split}'. Choose from {VALID_SPLITS}."
    
    # Special handling for cold_pharos
    if split == "cold_pharos":
        if pharos_proteins is None:
            raise ValueError("cold_pharos split requires pharos_proteins parameter")
        return _cold_pharos_split(df, pharos_proteins, val_frac, seed)
    
    attempt = 0
    last_error = None
    
    while attempt < max_retry_attempts:
        try:
            rng = np.random.RandomState(seed + attempt)
            
            if split == "random":
                train_df, val_df, test_df, split_info = _random_split(
                    df, test_frac, val_frac, rng, seed
                )
            elif split == "cold_both":
                train_df, val_df, test_df, split_info = _cold_both_split(
                    df, test_frac, val_frac, rng, seed
                )
            else:
                key = "drug_id" if split == "cold_drug" else "target_id"
                train_df, val_df, test_df, split_info = _cold_single_split(
                    df, key, test_frac, val_frac, rng, split, seed
                )
            
            # Validate minimum sample requirements
            if len(train_df) < min_samples_threshold:
                raise ValueError(
                    f"Train set has only {len(train_df)} samples "
                    f"(minimum: {min_samples_threshold})"
                )
            if len(test_df) < min_samples_threshold // 10:  # More lenient for test
                raise ValueError(
                    f"Test set has only {len(test_df)} samples "
                    f"(minimum: {min_samples_threshold // 10})"
                )
            
            # Verify leakage if requested
            if verify_leakage:
                split_info.verify_no_leakage()
            
            # Success!
            logger.info("Split successful on attempt %d/%d", attempt + 1, max_retry_attempts)
            logger.info("\n" + split_info.summary())
            return train_df, val_df, test_df, split_info
        
        except (ValueError, AssertionError) as e:
            last_error = e
            attempt += 1
            logger.warning(
                "Split attempt %d/%d failed: %s. Retrying with adjusted seed...",
                attempt, max_retry_attempts, str(e)
            )
    
    # All attempts failed
    raise ValueError(
        f"Failed to create valid split after {max_retry_attempts} attempts. "
        f"Last error: {last_error}. Try adjusting test_frac/val_frac or check data distribution."
    )


# ── Random split ──────────────────────────────

def _random_split(df, test_frac, val_frac, rng, seed):
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    n_val = int(len(idx) * val_frac)
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    split_info = SplitInfo(
        split_type="random",
        seed=seed,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        n_total=len(df),
        n_discarded=0,
    )
    
    return train_df, val_df, test_df, split_info


# ── Single-entity cold split ─────────────────

def _cold_single_split(df, key, test_frac, val_frac, rng, split_name, seed):
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
    train_groups = set(groups[n_test + n_val :])

    test_mask = df[key].isin(test_groups)
    val_mask = df[key].isin(val_groups)
    train_mask = ~test_mask & ~val_mask

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)
    
    # Build SplitInfo
    if key == "drug_id":
        split_info = SplitInfo(
            split_type=split_name,
            seed=seed,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            n_total=len(df),
            train_drugs=train_groups,
            val_drugs=val_groups,
            test_drugs=test_groups,
        )
    else:  # target_id
        split_info = SplitInfo(
            split_type=split_name,
            seed=seed,
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            n_total=len(df),
            train_targets=train_groups,
            val_targets=val_groups,
            test_targets=test_groups,
        )

    return train_df, val_df, test_df, split_info


# ── Cold-both split ──────────────────────────

def _cold_both_split(df, test_frac, val_frac, rng, seed):
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
    
    split_info = SplitInfo(
        split_type="cold_both",
        seed=seed,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        n_total=len(df),
        n_discarded=discarded,
        train_drugs=train_drugs,
        val_drugs=val_drugs,
        test_drugs=test_drugs,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )

    return train_df, val_df, test_df, split_info


# ── Cold-pharos split (Phase 3) ──────────────

def _cold_pharos_split(
    df: pd.DataFrame,
    pharos_proteins: Set[str],
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitInfo]:
    """
    Cold-pharos split: test set contains ONLY dark proteins (Pharos).
    
    Algorithm:
      1. Test set = all samples with target_id in pharos_proteins
      2. Train set = all samples with target_id NOT in pharos_proteins
      3. Validation set = carved from train set (val_frac of train)
      4. Drugs MAY overlap (realistic setting)
      5. Proteins do NOT overlap (zero-shot on dark proteins)
    
    Parameters
    ----------
    df : Full dataset
    pharos_proteins : Set of dark protein IDs
    val_frac : Fraction of training set to use for validation
    seed : Random seed for validation split
    
    Returns
    -------
    (train_df, val_df, test_df, split_info)
    """
    if "target_id" not in df.columns:
        raise ValueError("cold_pharos split requires 'target_id' column")
    
    # Split based on pharos proteins
    test_mask = df["target_id"].isin(pharos_proteins)
    train_mask = ~test_mask
    
    test_df = df[test_mask].reset_index(drop=True)
    train_full = df[train_mask].reset_index(drop=True)
    
    if len(test_df) == 0:
        raise ValueError(
            "cold_pharos split produced empty test set. "
            "Check that pharos_proteins exist in dataset."
        )
    
    # Carve validation from training
    rng = np.random.RandomState(seed)
    n_val = max(1, int(len(train_full) * val_frac))
    val_idx = rng.choice(len(train_full), size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(train_full)), val_idx)
    
    val_df = train_full.iloc[val_idx].reset_index(drop=True)
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    
    # Build SplitInfo
    train_proteins = set(train_df["target_id"].unique())
    val_proteins = set(val_df["target_id"].unique())
    test_proteins = pharos_proteins & set(test_df["target_id"].unique())
    
    split_info = SplitInfo(
        split_type="cold_pharos",
        seed=seed,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        n_total=len(df),
        train_targets=train_proteins,
        val_targets=val_proteins,
        test_targets=test_proteins,
    )
    
    logger.info(
        "cold_pharos split — train: %d, val: %d, test: %d (dark proteins: %d)",
        len(train_df), len(val_df), len(test_df), len(test_proteins),
    )
    
    # Verify no protein overlap
    assert len(test_proteins & train_proteins) == 0, (
        "Protein leakage in cold_pharos: test proteins found in train!"
    )
    
    return train_df, val_df, test_df, split_info


# ──────────────────────────────────────────────
# Phase 2: 5-Fold Entity-Group Cross-Validation
# ──────────────────────────────────────────────

def create_entity_group_folds(
    df: pd.DataFrame,
    n_folds: int = 5,
    split_type: str = "cold_both",
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]]:
    """
    Create k-fold entity-group cross-validation splits for cold-start evaluation.
    
    For cold_drug/cold_target/cold_both splits, entities (drugs/targets) are
    partitioned into k groups, and each fold uses one group as test set.
    
    Parameters
    ----------
    df : DataFrame with drug_id, target_id, smiles, sequence, affinity
    n_folds : number of folds (default: 5)
    split_type : 'cold_drug' | 'cold_target' | 'cold_both'
    seed : random seed for reproducibility
    
    Returns
    -------
    List of (train_df, test_df, split_info) tuples, one per fold.
    Validation set is carved from training set (10% of train).
    
    Notes
    -----
    For cold_both, both drugs AND targets are partitioned into k groups.
    Fold i uses drug_group_i × target_group_i as test set.
    """
    assert split_type in ("cold_drug", "cold_target", "cold_both"), (
        f"Entity-group CV only supports cold splits, got '{split_type}'"
    )
    
    rng = np.random.RandomState(seed)
    folds = []
    
    if split_type == "cold_drug":
        drugs = df["drug_id"].unique().tolist()
        rng.shuffle(drugs)
        drug_groups = np.array_split(drugs, n_folds)
        
        for fold_idx in range(n_folds):
            test_drugs = set(drug_groups[fold_idx])
            train_drugs = set([d for i, g in enumerate(drug_groups) if i != fold_idx for d in g])
            
            test_mask = df["drug_id"].isin(test_drugs)
            train_mask = ~test_mask
            
            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)
            
            # Carve out validation from train (10%)
            val_size = max(1, int(len(train_df) * 0.1))
            val_df = train_df.iloc[:val_size].reset_index(drop=True)
            train_df = train_df.iloc[val_size:].reset_index(drop=True)
            
            split_info = SplitInfo(
                split_type=f"{split_type}_fold{fold_idx+1}",
                seed=seed,
                n_train=len(train_df),
                n_val=len(val_df),
                n_test=len(test_df),
                n_total=len(df),
                train_drugs=train_drugs - set(val_df["drug_id"].unique()),
                test_drugs=test_drugs,
            )
            folds.append((train_df, test_df, split_info))
    
    elif split_type == "cold_target":
        targets = df["target_id"].unique().tolist()
        rng.shuffle(targets)
        target_groups = np.array_split(targets, n_folds)
        
        for fold_idx in range(n_folds):
            test_targets = set(target_groups[fold_idx])
            train_targets = set([t for i, g in enumerate(target_groups) if i != fold_idx for t in g])
            
            test_mask = df["target_id"].isin(test_targets)
            train_mask = ~test_mask
            
            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)
            
            val_size = max(1, int(len(train_df) * 0.1))
            val_df = train_df.iloc[:val_size].reset_index(drop=True)
            train_df = train_df.iloc[val_size:].reset_index(drop=True)
            
            split_info = SplitInfo(
                split_type=f"{split_type}_fold{fold_idx+1}",
                seed=seed,
                n_train=len(train_df),
                n_val=len(val_df),
                n_test=len(test_df),
                n_total=len(df),
                train_targets=train_targets - set(val_df["target_id"].unique()),
                test_targets=test_targets,
            )
            folds.append((train_df, test_df, split_info))
    
    elif split_type == "cold_both":
        drugs = df["drug_id"].unique().tolist()
        targets = df["target_id"].unique().tolist()
        rng.shuffle(drugs)
        rng.shuffle(targets)
        drug_groups = np.array_split(drugs, n_folds)
        target_groups = np.array_split(targets, n_folds)
        
        for fold_idx in range(n_folds):
            test_drugs = set(drug_groups[fold_idx])
            test_targets = set(target_groups[fold_idx])
            train_drugs = set([d for i, g in enumerate(drug_groups) if i != fold_idx for d in g])
            train_targets = set([t for i, g in enumerate(target_groups) if i != fold_idx for t in g])
            
            # Apply cold_both filtering
            test_mask = df["drug_id"].isin(test_drugs) & df["target_id"].isin(test_targets)
            train_mask = df["drug_id"].isin(train_drugs) & df["target_id"].isin(train_targets)
            
            train_df = df[train_mask].reset_index(drop=True)
            test_df = df[test_mask].reset_index(drop=True)
            
            val_size = max(1, int(len(train_df) * 0.1))
            val_df = train_df.iloc[:val_size].reset_index(drop=True)
            train_df = train_df.iloc[val_size:].reset_index(drop=True)
            
            discarded = len(df) - len(train_df) - len(val_df) - len(test_df)
            
            split_info = SplitInfo(
                split_type=f"{split_type}_fold{fold_idx+1}",
                seed=seed,
                n_train=len(train_df),
                n_val=len(val_df),
                n_test=len(test_df),
                n_total=len(df),
                n_discarded=discarded,
                train_drugs=train_drugs,
                test_drugs=test_drugs,
                train_targets=train_targets,
                test_targets=test_targets,
            )
            folds.append((train_df, test_df, split_info))
    
    logger.info(
        "Created %d-fold entity-group CV for %s: avg test size = %.1f samples",
        n_folds, split_type, np.mean([len(fold[1]) for fold in folds])
    )
    
    return folds


# ──────────────────────────────────────────────
# Legacy compatibility (deprecated)
# ──────────────────────────────────────────────

def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str,
) -> None:
    """
    DEPRECATED: Use SplitInfo.verify_no_leakage() instead.
    
    Assert that cold-split entity-separation constraints are respected.
    """
    logger.warning(
        "verify_no_leakage() is deprecated. Use SplitInfo.verify_no_leakage() instead."
    )

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