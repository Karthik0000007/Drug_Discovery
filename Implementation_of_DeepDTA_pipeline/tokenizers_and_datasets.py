"""
tokenizers_and_datasets.py — Character-level tokenization and PyTorch Datasets.

Vocabulary layout:
  0 = <PAD>
  1 = <UNK>
  2 = <MASK>   (used by contrastive augmentations)
  3.. = data characters (sorted)
"""

from typing import List, Dict, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<MASK>"]
PAD_IDX = 0
UNK_IDX = 1
MASK_IDX = 2


def build_vocab(sequences: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a character-level vocabulary from *sequences*.

    Returns
    -------
    stoi : dict mapping character → index
    itos : list mapping index → character
    """
    freq: Dict[str, int] = {}
    for s in sequences:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    items = sorted(ch for ch, cnt in freq.items() if cnt >= min_freq)
    itos = SPECIAL_TOKENS + items
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def tokenize_seq(s: str, stoi: Dict[str, int], max_len: int) -> List[int]:
    """Convert string *s* to a list of integer token IDs (truncated / padded)."""
    ids = [stoi.get(ch, UNK_IDX) for ch in s]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [PAD_IDX] * (max_len - len(ids))


# ──────────────────────────────────────────────
# Supervised DTA Dataset
# ──────────────────────────────────────────────

class DtaDataset(Dataset):
    """
    PyTorch Dataset for drug–target affinity regression.
    Returns dicts: {smiles: LongTensor, seq: LongTensor, aff: FloatTensor}.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sml_stoi: Dict[str, int],
        prot_stoi: Dict[str, int],
        max_sml_len: int = 120,
        max_prot_len: int = 1000,
    ):
        self.df = df.reset_index(drop=True)
        self.sml_stoi = sml_stoi
        self.prot_stoi = prot_stoi
        self.max_sml_len = max_sml_len
        self.max_prot_len = max_prot_len

        for col in ("smiles", "sequence", "affinity"):
            assert col in df.columns, f"DataFrame must have column '{col}'"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        s_ids = tokenize_seq(row["smiles"], self.sml_stoi, self.max_sml_len)
        p_ids = tokenize_seq(row["sequence"], self.prot_stoi, self.max_prot_len)
        return {
            "smiles": torch.LongTensor(s_ids),
            "seq": torch.LongTensor(p_ids),
            "aff": torch.FloatTensor([float(row["affinity"])]),
        }