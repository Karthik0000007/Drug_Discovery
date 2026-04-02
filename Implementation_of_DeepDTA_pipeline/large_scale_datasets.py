"""
large_scale_datasets.py — Phase 3: Large-scale dataset integration.

Implements:
  - BindingDB dataset loading and preprocessing (~millions of samples)
  - Pharos dark protein dataset for zero-shot evaluation
  - Memory-efficient lazy loading with optional caching
  - Affinity normalization (Ki, IC50, Kd → pKd/pKi)
  - Data filtering and quality control
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Affinity Normalization
# ──────────────────────────────────────────────

def normalize_affinity(
    value: float,
    unit: str,
    affinity_type: str = "Kd",
) -> Optional[float]:
    """
    Normalize affinity values to pKd/pKi scale.
    
    Parameters
    ----------
    value : affinity value (numeric)
    unit : 'nM' | 'uM' | 'mM' | 'M' | 'pM'
    affinity_type : 'Kd' | 'Ki' | 'IC50' | 'EC50'
    
    Returns
    -------
    Normalized value on pKd/pKi scale (higher = stronger binding)
    None if value is invalid or cannot be normalized
    
    Notes
    -----
    Formula: pX = -log10(value_in_molar)
    - nM → M: multiply by 1e-9
    - uM → M: multiply by 1e-6
    - mM → M: multiply by 1e-3
    - pM → M: multiply by 1e-12
    """
    if value <= 0 or np.isnan(value) or np.isinf(value):
        return None
    
    # Convert to molar
    unit = unit.strip().lower()
    if unit == "nm":
        molar = value * 1e-9
    elif unit == "um" or unit == "μm":
        molar = value * 1e-6
    elif unit == "mm":
        molar = value * 1e-3
    elif unit == "pm":
        molar = value * 1e-12
    elif unit == "m":
        molar = value
    else:
        logger.warning(f"Unknown unit '{unit}', skipping")
        return None
    
    # Convert to pKd/pKi scale
    try:
        p_value = -np.log10(molar)
        # Sanity check: typical range is 3-12 for pKd/pKi
        if p_value < 0 or p_value > 15:
            logger.warning(f"Unusual pKd value {p_value:.2f} from {value} {unit}")
            return None
        return p_value
    except (ValueError, OverflowError):
        return None


# ──────────────────────────────────────────────
# BindingDB Dataset Loading
# ──────────────────────────────────────────────

@dataclass
class BindingDBConfig:
    """Configuration for BindingDB preprocessing."""
    min_affinity: float = 3.0          # Minimum pKd (weaker than 1 mM)
    max_affinity: float = 12.0         # Maximum pKd (stronger than 1 pM)
    min_sequence_length: int = 50
    max_sequence_length: int = 5000
    min_smiles_length: int = 5
    max_smiles_length: int = 200
    allowed_affinity_types: List[str] = None
    remove_duplicates: bool = True
    keep_highest_quality: bool = True  # If duplicates, keep highest affinity
    
    def __post_init__(self):
        if self.allowed_affinity_types is None:
            self.allowed_affinity_types = ["Kd", "Ki", "IC50"]


def load_bindingdb(
    path: str,
    config: Optional[BindingDBConfig] = None,
    cache_path: Optional[str] = None,
    force_reprocess: bool = False,
) -> pd.DataFrame:
    """
    Load and preprocess BindingDB dataset.
    
    Parameters
    ----------
    path : Path to raw BindingDB TSV file
    config : BindingDBConfig for filtering parameters
    cache_path : Optional path to save/load preprocessed .parquet file
    force_reprocess : If True, ignore cache and reprocess from raw
    
    Returns
    -------
    DataFrame with columns: drug_id, target_id, smiles, sequence, affinity
    
    Notes
    -----
    Expected BindingDB columns (subset):
    - Ligand SMILES
    - Target Sequence
    - Ki (nM), Kd (nM), IC50 (nM), EC50 (nM)
    - Target Name / UniProt ID
    """
    config = config or BindingDBConfig()
    
    # Check cache
    if cache_path and os.path.exists(cache_path) and not force_reprocess:
        logger.info(f"Loading cached BindingDB from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} cached samples")
        return df
    
    logger.info(f"Loading raw BindingDB from {path}")
    
    # Read raw file (adjust column names based on actual BindingDB format)
    # This is a template - actual column names may vary
    try:
        raw_df = pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
            usecols=[
                "Ligand SMILES",
                "Target Sequence",
                "Ki (nM)",
                "Kd (nM)",
                "IC50 (nM)",
                "Target Name",
                "UniProt (SwissProt) Primary ID of Target Chain",
            ],
        )
    except Exception as e:
        logger.error(f"Failed to load BindingDB: {e}")
        logger.info("Attempting to load with flexible column detection...")
        raw_df = pd.read_csv(path, sep="\t", low_memory=False)
    
    logger.info(f"Raw BindingDB: {len(raw_df)} rows")
    
    # Process each row
    rows = []
    for idx, row in raw_df.iterrows():
        if idx % 100000 == 0:
            logger.info(f"Processing row {idx}/{len(raw_df)}")
        
        # Extract SMILES and sequence
        smiles = row.get("Ligand SMILES", "")
        sequence = row.get("Target Sequence", "")
        
        if pd.isna(smiles) or pd.isna(sequence):
            continue
        
        smiles = str(smiles).strip()
        sequence = str(sequence).strip()
        
        # Filter by length
        if not (config.min_smiles_length <= len(smiles) <= config.max_smiles_length):
            continue
        if not (config.min_sequence_length <= len(sequence) <= config.max_sequence_length):
            continue
        
        # Extract affinity (try multiple columns)
        affinity = None
        affinity_type = None
        
        for aff_type in config.allowed_affinity_types:
            col_name = f"{aff_type} (nM)"
            if col_name in row and not pd.isna(row[col_name]):
                try:
                    val = float(row[col_name])
                    normalized = normalize_affinity(val, "nM", aff_type)
                    if normalized is not None:
                        affinity = normalized
                        affinity_type = aff_type
                        break
                except (ValueError, TypeError):
                    continue
        
        if affinity is None:
            continue
        
        # Filter by affinity range
        if not (config.min_affinity <= affinity <= config.max_affinity):
            continue
        
        # Generate IDs
        target_id = row.get("UniProt (SwissProt) Primary ID of Target Chain", "")
        if pd.isna(target_id) or target_id == "":
            target_id = row.get("Target Name", f"target_{idx}")
        
        drug_id = f"drug_{idx}"  # BindingDB doesn't have stable drug IDs
        
        rows.append({
            "drug_id": str(drug_id),
            "target_id": str(target_id),
            "smiles": smiles,
            "sequence": sequence,
            "affinity": affinity,
            "affinity_type": affinity_type,
        })
    
    df = pd.DataFrame(rows)
    logger.info(f"After filtering: {len(df)} samples")
    
    # Remove duplicates
    if config.remove_duplicates:
        n_before = len(df)
        if config.keep_highest_quality:
            # Keep highest affinity for each (smiles, sequence) pair
            df = df.sort_values("affinity", ascending=False)
            df = df.drop_duplicates(subset=["smiles", "sequence"], keep="first")
        else:
            df = df.drop_duplicates(subset=["smiles", "sequence"])
        logger.info(f"Removed {n_before - len(df)} duplicates")
    
    # Drop affinity_type column (not needed for training)
    df = df.drop(columns=["affinity_type"])
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached preprocessed BindingDB to {cache_path}")
    
    logger.info(f"Final BindingDB dataset: {len(df)} samples")
    logger.info(f"  Unique drugs: {df['drug_id'].nunique()}")
    logger.info(f"  Unique targets: {df['target_id'].nunique()}")
    logger.info(f"  Affinity range: [{df['affinity'].min():.2f}, {df['affinity'].max():.2f}]")
    
    return df


# ──────────────────────────────────────────────
# Pharos Dark Protein Dataset
# ──────────────────────────────────────────────

def load_pharos(
    path: str,
    interaction_threshold: int = 10,
    cache_path: Optional[str] = None,
    force_reprocess: bool = False,
) -> pd.DataFrame:
    """
    Load Pharos dark protein dataset for zero-shot evaluation.
    
    Parameters
    ----------
    path : Path to Pharos dataset (TSV or CSV)
    interaction_threshold : Proteins with < this many interactions are "dark"
    cache_path : Optional path to save/load preprocessed .parquet file
    force_reprocess : If True, ignore cache and reprocess from raw
    
    Returns
    -------
    DataFrame with columns: drug_id, target_id, smiles, sequence, affinity
    Only includes proteins with < interaction_threshold known ligands
    
    Notes
    -----
    "Dark proteins" are understudied proteins with few known interactions.
    This creates a true zero-shot benchmark where test proteins have
    minimal training data.
    """
    # Check cache
    if cache_path and os.path.exists(cache_path) and not force_reprocess:
        logger.info(f"Loading cached Pharos from {cache_path}")
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded {len(df)} cached Pharos samples")
        return df
    
    logger.info(f"Loading raw Pharos from {path}")
    
    # Load raw data
    if path.endswith(".tsv"):
        raw_df = pd.read_csv(path, sep="\t", low_memory=False)
    else:
        raw_df = pd.read_csv(path, low_memory=False)
    
    logger.info(f"Raw Pharos: {len(raw_df)} rows")
    
    # Count interactions per protein
    if "target_id" in raw_df.columns:
        interaction_counts = raw_df["target_id"].value_counts()
    elif "UniProt" in raw_df.columns:
        interaction_counts = raw_df["UniProt"].value_counts()
    else:
        raise ValueError("Pharos dataset must have 'target_id' or 'UniProt' column")
    
    # Identify dark proteins
    dark_proteins = set(
        interaction_counts[interaction_counts < interaction_threshold].index
    )
    logger.info(
        f"Identified {len(dark_proteins)} dark proteins "
        f"(< {interaction_threshold} interactions)"
    )
    
    # Filter to dark proteins only
    if "target_id" in raw_df.columns:
        df = raw_df[raw_df["target_id"].isin(dark_proteins)].copy()
    else:
        df = raw_df[raw_df["UniProt"].isin(dark_proteins)].copy()
        df = df.rename(columns={"UniProt": "target_id"})
    
    # Ensure required columns
    required_cols = ["smiles", "sequence", "affinity"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Pharos dataset must have column '{col}'")
    
    # Add drug_id if missing
    if "drug_id" not in df.columns:
        df["drug_id"] = [f"pharos_drug_{i}" for i in range(len(df))]
    
    # Clean and filter
    df = df.dropna(subset=["smiles", "sequence", "affinity"])
    df = df[df["smiles"].str.len() > 0]
    df = df[df["sequence"].str.len() >= 50]
    
    df = df[["drug_id", "target_id", "smiles", "sequence", "affinity"]].reset_index(drop=True)
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached preprocessed Pharos to {cache_path}")
    
    logger.info(f"Final Pharos dataset: {len(df)} samples")
    logger.info(f"  Dark proteins: {df['target_id'].nunique()}")
    logger.info(f"  Unique drugs: {df['drug_id'].nunique()}")
    
    return df


# ──────────────────────────────────────────────
# Memory-Efficient Dataset (Lazy Loading)
# ──────────────────────────────────────────────

class LazyDtaDataset(Dataset):
    """
    Memory-efficient dataset for large-scale DTA data.
    
    Uses lazy loading with optional memory mapping or chunked reading.
    Supports optional pre-tokenization caching.
    """
    
    def __init__(
        self,
        parquet_path: str,
        sml_stoi: Dict[str, int],
        prot_stoi: Dict[str, int],
        max_sml_len: int = 120,
        max_prot_len: int = 1000,
        cache_tokenized: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        parquet_path : Path to .parquet file with preprocessed data
        sml_stoi, prot_stoi : Vocabulary mappings
        max_sml_len, max_prot_len : Maximum sequence lengths
        cache_tokenized : If True, cache tokenized tensors to disk
        cache_dir : Directory for tokenized cache files
        """
        self.parquet_path = parquet_path
        self.sml_stoi = sml_stoi
        self.prot_stoi = prot_stoi
        self.max_sml_len = max_sml_len
        self.max_prot_len = max_prot_len
        
        # Load metadata only (not full data)
        self.df = pd.read_parquet(parquet_path, columns=["drug_id", "target_id"])
        self._length = len(self.df)
        
        # Tokenization cache
        self.cache_tokenized = cache_tokenized
        self.cache_dir = cache_dir
        self._token_cache = {}
        
        if cache_tokenized and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "tokenized_cache.pt")
            if os.path.exists(cache_file):
                logger.info(f"Loading tokenized cache from {cache_file}")
                self._token_cache = torch.load(cache_file)
                logger.info(f"Loaded {len(self._token_cache)} cached samples")
    
    def __len__(self) -> int:
        return self._length
    
    def __getitem__(self, idx: int):
        # Check cache first
        if idx in self._token_cache:
            return self._token_cache[idx]
        
        # Load single row from parquet (efficient with columnar format)
        row = pd.read_parquet(
            self.parquet_path,
            columns=["smiles", "sequence", "affinity"],
        ).iloc[idx]
        
        # Tokenize
        from .tokenizers_and_datasets import tokenize_seq
        s_ids = tokenize_seq(row["smiles"], self.sml_stoi, self.max_sml_len)
        p_ids = tokenize_seq(row["sequence"], self.prot_stoi, self.max_prot_len)
        
        sample = {
            "smiles": torch.LongTensor(s_ids),
            "seq": torch.LongTensor(p_ids),
            "aff": torch.FloatTensor([float(row["affinity"])]),
        }
        
        # Cache if enabled
        if self.cache_tokenized:
            self._token_cache[idx] = sample
        
        return sample
    
    def save_cache(self):
        """Save tokenized cache to disk."""
        if self.cache_tokenized and self.cache_dir and self._token_cache:
            cache_file = os.path.join(self.cache_dir, "tokenized_cache.pt")
            torch.save(self._token_cache, cache_file)
            logger.info(f"Saved {len(self._token_cache)} tokenized samples to {cache_file}")


# ──────────────────────────────────────────────
# Dataset Statistics
# ──────────────────────────────────────────────

def compute_dataset_stats(df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive statistics for a DTA dataset.
    
    Returns
    -------
    Dict with keys:
    - n_samples, n_drugs, n_targets
    - affinity_mean, affinity_std, affinity_min, affinity_max
    - smiles_len_mean, smiles_len_max
    - sequence_len_mean, sequence_len_max
    - affinity_distribution (histogram bins)
    """
    stats = {
        "n_samples": len(df),
        "n_drugs": df["drug_id"].nunique(),
        "n_targets": df["target_id"].nunique(),
        "affinity_mean": df["affinity"].mean(),
        "affinity_std": df["affinity"].std(),
        "affinity_min": df["affinity"].min(),
        "affinity_max": df["affinity"].max(),
        "smiles_len_mean": df["smiles"].str.len().mean(),
        "smiles_len_max": df["smiles"].str.len().max(),
        "sequence_len_mean": df["sequence"].str.len().mean(),
        "sequence_len_max": df["sequence"].str.len().max(),
    }
    
    # Affinity distribution
    hist, bins = np.histogram(df["affinity"], bins=20)
    stats["affinity_hist"] = hist.tolist()
    stats["affinity_bins"] = bins.tolist()
    
    return stats


def log_dataset_stats(df: pd.DataFrame, name: str = "Dataset"):
    """Log comprehensive dataset statistics."""
    stats = compute_dataset_stats(df)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Samples:          {stats['n_samples']:,}")
    logger.info(f"Unique drugs:     {stats['n_drugs']:,}")
    logger.info(f"Unique targets:   {stats['n_targets']:,}")
    logger.info(f"Affinity:         {stats['affinity_mean']:.2f} ± {stats['affinity_std']:.2f}")
    logger.info(f"  Range:          [{stats['affinity_min']:.2f}, {stats['affinity_max']:.2f}]")
    logger.info(f"SMILES length:    {stats['smiles_len_mean']:.1f} (max: {stats['smiles_len_max']})")
    logger.info(f"Sequence length:  {stats['sequence_len_mean']:.1f} (max: {stats['sequence_len_max']})")
    logger.info(f"{'='*60}\n")
