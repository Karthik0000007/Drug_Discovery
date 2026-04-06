"""
tokenizers_and_datasets.py — Character-level + pretrained Hugging Face tokenization.

Vocabulary layout (character-level):
  0 = <PAD>
  1 = <UNK>
  2 = <MASK>   (used by contrastive augmentations)

Phase 4 additions:
  - Optional Hugging Face tokenizers (ESM/ProtBERT/ChemBERTa/MolFormer)
  - Tokenization caching to avoid recomputation
  - Datasets return raw text + tokenized tensors for LLM alignment
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import torch
from torch.utils.data import Dataset

# Optional HF tokenizer wrapper (lazy import to avoid hard dependency when unused)
try:
    from .pretrained_tokenizers import PretrainedTokenizerWrapper
except Exception:
    PretrainedTokenizerWrapper = None

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary utilities (character-level baseline)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face tokenization helpers (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def maybe_tokenize_hf(
    text: str | List[str],
    tokenizer: Optional["PretrainedTokenizerWrapper"],
) -> Optional[Dict[str, torch.Tensor]]:
    """Tokenize with HF wrapper if provided, else return None."""
    if tokenizer is None:
        return None
    return tokenizer.tokenize(text, return_tensors="pt")


# ─────────────────────────────────────────────────────────────────────────────
# Supervised DTA Dataset
# ─────────────────────────────────────────────────────────────────────────────

class DtaDataset(Dataset):
    """
    PyTorch Dataset for drug–target affinity regression.

    Returns dicts containing character-level tensors (for legacy CNN) and,
    optionally, Hugging Face tokenized tensors + raw text for LLM alignment.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sml_stoi: Dict[str, int],
        prot_stoi: Dict[str, int],
        max_sml_len: int = 120,
        max_prot_len: int = 1000,
        use_pretrained_tokenizers: bool = False,
        drug_tokenizer: Optional[Any] = None,
        prot_tokenizer: Optional[Any] = None,
        return_text: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.sml_stoi = sml_stoi
        self.prot_stoi = prot_stoi
        self.max_sml_len = max_sml_len
        self.max_prot_len = max_prot_len
        self.use_pretrained_tokenizers = use_pretrained_tokenizers or (drug_tokenizer is not None) or (prot_tokenizer is not None)
        self.drug_tokenizer = drug_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.return_text = return_text or self.use_pretrained_tokenizers

        for col in ("smiles", "sequence", "affinity"):
            assert col in df.columns, f"DataFrame must have column '{col}'"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]

        s_ids = tokenize_seq(smiles, self.sml_stoi, self.max_sml_len)
        p_ids = tokenize_seq(sequence, self.prot_stoi, self.max_prot_len)

        sample = {
            "smiles": torch.LongTensor(s_ids),
            "seq": torch.LongTensor(p_ids),
            "aff": torch.FloatTensor([float(row["affinity"])]),
        }

        if self.return_text:
            sample["smiles_text"] = smiles
            sample["sequence_text"] = sequence

        if self.use_pretrained_tokenizers:
            sample["smiles_tokens"] = maybe_tokenize_hf(smiles, self.drug_tokenizer)
            sample["seq_tokens"] = maybe_tokenize_hf(sequence, self.prot_tokenizer)

        return sample
