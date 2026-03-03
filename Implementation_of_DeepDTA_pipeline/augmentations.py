"""
augmentations.py — Domain-specific augmentation strategies for contrastive pretraining.

Drug (SMILES) augmentations:
  1. smiles_enum      — RDKit non-canonical SMILES (strongest)
  2. atom_mask        — Random token masking with <MASK>
  3. substruct_dropout— Random atom removal + validity check

Protein (sequence) augmentations:
  4. subseq_crop      — Random contiguous window (70-100%)
  5. residue_mask     — Random amino acid masking
  6. residue_sub      — BLOSUM62-based biochemically similar substitution

Each augmentation: str → str. Tokenizer handles conversion to tensors.
"""

from __future__ import annotations

import random
import logging
from typing import List, Callable, Dict

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Try importing RDKit (graceful fallback)
# ──────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not installed. SMILES enumeration and substructure dropout will fall back to identity.")


# ──────────────────────────────────────────────
# BLOSUM62 simplified substitution probabilities
# ──────────────────────────────────────────────

BLOSUM62_PROBS: Dict[str, Dict[str, float]] = {
    "A": {"A": 0.30, "G": 0.14, "S": 0.12, "T": 0.10, "V": 0.10, "L": 0.08, "I": 0.08, "P": 0.08},
    "R": {"R": 0.30, "K": 0.20, "Q": 0.15, "H": 0.10, "N": 0.10, "E": 0.08, "D": 0.07},
    "N": {"N": 0.30, "D": 0.18, "S": 0.14, "H": 0.12, "K": 0.10, "Q": 0.08, "T": 0.08},
    "D": {"D": 0.30, "N": 0.18, "E": 0.16, "Q": 0.10, "S": 0.10, "H": 0.08, "K": 0.08},
    "C": {"C": 0.60, "S": 0.15, "A": 0.10, "T": 0.08, "V": 0.07},
    "Q": {"Q": 0.30, "E": 0.16, "R": 0.14, "K": 0.12, "N": 0.10, "H": 0.10, "D": 0.08},
    "E": {"E": 0.30, "D": 0.16, "Q": 0.16, "K": 0.12, "R": 0.10, "N": 0.08, "A": 0.08},
    "G": {"G": 0.40, "A": 0.18, "S": 0.14, "N": 0.10, "D": 0.10, "T": 0.08},
    "H": {"H": 0.35, "N": 0.14, "Q": 0.14, "Y": 0.12, "R": 0.10, "D": 0.08, "K": 0.07},
    "I": {"I": 0.28, "V": 0.20, "L": 0.20, "M": 0.12, "F": 0.10, "A": 0.05, "T": 0.05},
    "L": {"L": 0.28, "I": 0.18, "V": 0.14, "M": 0.14, "F": 0.12, "A": 0.07, "W": 0.07},
    "K": {"K": 0.28, "R": 0.22, "Q": 0.14, "E": 0.12, "N": 0.10, "H": 0.07, "A": 0.07},
    "M": {"M": 0.28, "L": 0.20, "I": 0.16, "V": 0.14, "F": 0.10, "Q": 0.06, "A": 0.06},
    "F": {"F": 0.30, "Y": 0.20, "W": 0.14, "L": 0.14, "I": 0.10, "V": 0.06, "M": 0.06},
    "P": {"P": 0.50, "A": 0.14, "S": 0.10, "T": 0.10, "G": 0.08, "V": 0.08},
    "S": {"S": 0.28, "T": 0.18, "A": 0.16, "N": 0.12, "G": 0.10, "P": 0.08, "D": 0.08},
    "T": {"T": 0.28, "S": 0.20, "A": 0.16, "V": 0.12, "N": 0.10, "I": 0.07, "P": 0.07},
    "W": {"W": 0.50, "F": 0.14, "Y": 0.14, "L": 0.10, "R": 0.06, "H": 0.06},
    "Y": {"Y": 0.30, "F": 0.20, "W": 0.14, "H": 0.12, "L": 0.08, "I": 0.08, "V": 0.08},
    "V": {"V": 0.28, "I": 0.20, "L": 0.16, "A": 0.12, "M": 0.10, "T": 0.07, "F": 0.07},
}


# ──────────────────────────────────────────────
# Drug (SMILES) augmentations
# ──────────────────────────────────────────────

def smiles_enumeration(smiles: str) -> str:
    """Generate a random non-canonical SMILES for the same molecule."""
    if not RDKIT_AVAILABLE:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(atom_order)
        renumbered = Chem.RenumberAtoms(mol, atom_order)
        return Chem.MolToSmiles(renumbered, canonical=False)
    except Exception:
        return smiles


def atom_masking(smiles: str, mask_ratio: float = 0.15) -> str:
    """Replace mask_ratio fraction of SMILES characters with <MASK>."""
    chars = list(smiles)
    if len(chars) == 0:
        return smiles
    n_mask = max(1, int(len(chars) * mask_ratio))
    mask_positions = random.sample(range(len(chars)), min(n_mask, len(chars)))
    for pos in mask_positions:
        chars[pos] = "<MASK>"
    return "".join(chars)


def substructure_dropout(smiles: str, drop_prob: float = 0.1, max_retries: int = 5) -> str:
    """Remove a random atom from the molecule; validate with RDKit."""
    if not RDKIT_AVAILABLE:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() <= 3:
        return smiles
    for _ in range(max_retries):
        atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
        try:
            edit_mol = Chem.RWMol(mol)
            edit_mol.RemoveAtom(atom_idx)
            result = Chem.MolToSmiles(edit_mol)
            if Chem.MolFromSmiles(result) is not None:
                return result
        except Exception:
            continue
    return smiles  # fallback


# ──────────────────────────────────────────────
# Protein (sequence) augmentations
# ──────────────────────────────────────────────

def subsequence_crop(sequence: str, min_ratio: float = 0.7) -> str:
    """Randomly crop a contiguous window of 70–100% of sequence length."""
    seq_len = len(sequence)
    if seq_len <= 1:
        return sequence
    crop_len = random.randint(int(seq_len * min_ratio), seq_len)
    start = random.randint(0, seq_len - crop_len)
    return sequence[start : start + crop_len]


def residue_masking(sequence: str, mask_ratio: float = 0.15) -> str:
    """Replace mask_ratio fraction of amino acid characters with <MASK>."""
    chars = list(sequence)
    if len(chars) == 0:
        return sequence
    n_mask = max(1, int(len(chars) * mask_ratio))
    mask_positions = random.sample(range(len(chars)), min(n_mask, len(chars)))
    for pos in mask_positions:
        chars[pos] = "<MASK>"
    return "".join(chars)


def residue_substitution(sequence: str, sub_ratio: float = 0.10) -> str:
    """Replace sub_ratio fraction of residues with biochemically similar amino acids."""
    chars = list(sequence)
    if len(chars) == 0:
        return sequence
    n_sub = max(1, int(len(chars) * sub_ratio))
    sub_positions = random.sample(range(len(chars)), min(n_sub, len(chars)))
    for pos in sub_positions:
        original = chars[pos].upper()
        if original in BLOSUM62_PROBS:
            candidates = list(BLOSUM62_PROBS[original].keys())
            weights = list(BLOSUM62_PROBS[original].values())
            chars[pos] = random.choices(candidates, weights=weights, k=1)[0]
    return "".join(chars)


# ──────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────

DRUG_AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    "smiles_enum": smiles_enumeration,
    "atom_mask": atom_masking,
    "substruct_dropout": substructure_dropout,
}

PROTEIN_AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    "subseq_crop": subsequence_crop,
    "residue_mask": residue_masking,
    "residue_sub": residue_substitution,
}


def apply_random_augmentation(
    s: str,
    aug_names: List[str],
    registry: Dict[str, Callable],
    **kwargs,
) -> str:
    """Apply one randomly selected augmentation from *aug_names*."""
    if not aug_names:
        return s
    name = random.choice(aug_names)
    fn = registry[name]
    # Pass relevant kwargs
    import inspect
    sig = inspect.signature(fn)
    valid_kw = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(s, **valid_kw)
