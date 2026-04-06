"""
contrastive_dataset.py — Contrastive datasets with Phase 4 pretrained tokenizer support.

Adds:
  - Raw augmented text for LLM alignment
  - Optional Hugging Face tokenization with caching
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import (
    apply_random_augmentation,
    DRUG_AUGMENTATION_REGISTRY,
    PROTEIN_AUGMENTATION_REGISTRY,
)
from .tokenizers_and_datasets import tokenize_seq, maybe_tokenize_hf


class ContrastiveDrugDataset(Dataset):
    """
    Contrastive dataset for drug SMILES.
    Each __getitem__ returns two augmented views of the same SMILES.
    """

    def __init__(
        self,
        smiles_list: List[str],
        stoi: Dict[str, int],
        max_len: int = 120,
        aug_names: List[str] | None = None,
        mask_ratio: float = 0.15,
        drop_prob: float = 0.1,
        use_pretrained_tokenizer: bool = False,
        tokenizer: Optional[Any] = None,
    ):
        self.smiles_list = smiles_list
        self.stoi = stoi
        self.max_len = max_len
        self.aug_names = aug_names or ["smiles_enum", "atom_mask"]
        self.aug_kwargs = {"mask_ratio": mask_ratio, "drop_prob": drop_prob}
        self.use_pretrained_tokenizer = use_pretrained_tokenizer or tokenizer is not None
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int):
        original = self.smiles_list[idx]
        view1 = apply_random_augmentation(
            original, self.aug_names, DRUG_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        view2 = apply_random_augmentation(
            original, self.aug_names, DRUG_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        v1_ids = tokenize_seq(view1, self.stoi, self.max_len)
        v2_ids = tokenize_seq(view2, self.stoi, self.max_len)

        sample = {
            "view1": torch.LongTensor(v1_ids),
            "view2": torch.LongTensor(v2_ids),
            "index": idx,
            "view1_text": view1,
            "view2_text": view2,
        }

        if self.use_pretrained_tokenizer:
            sample["view1_tokens"] = maybe_tokenize_hf(view1, self.tokenizer)
            sample["view2_tokens"] = maybe_tokenize_hf(view2, self.tokenizer)

        return sample


class ContrastiveProteinDataset(Dataset):
    """
    Contrastive dataset for protein sequences.
    Each __getitem__ returns two augmented views of the same sequence.
    """

    def __init__(
        self,
        sequences: List[str],
        stoi: Dict[str, int],
        max_len: int = 1000,
        aug_names: List[str] | None = None,
        mask_ratio: float = 0.15,
        crop_min_ratio: float = 0.7,
        sub_ratio: float = 0.10,
        use_pretrained_tokenizer: bool = False,
        tokenizer: Optional[Any] = None,
    ):
        self.sequences = sequences
        self.stoi = stoi
        self.max_len = max_len
        self.aug_names = aug_names or ["subseq_crop", "residue_mask"]
        self.aug_kwargs = {
            "mask_ratio": mask_ratio,
            "min_ratio": crop_min_ratio,
            "sub_ratio": sub_ratio,
        }
        self.use_pretrained_tokenizer = use_pretrained_tokenizer or tokenizer is not None
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        original = self.sequences[idx]
        prot_view1 = apply_random_augmentation(
            original, self.aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        prot_view2 = apply_random_augmentation(
            original, self.aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        v1_ids = tokenize_seq(prot_view1, self.stoi, self.max_len)
        v2_ids = tokenize_seq(prot_view2, self.stoi, self.max_len)

        sample = {
            "view1": torch.LongTensor(v1_ids),
            "view2": torch.LongTensor(v2_ids),
            "index": idx,
            "view1_text": prot_view1,
            "view2_text": prot_view2,
        }

        if self.use_pretrained_tokenizer:
            sample["view1_tokens"] = maybe_tokenize_hf(prot_view1, self.tokenizer)
            sample["view2_tokens"] = maybe_tokenize_hf(prot_view2, self.tokenizer)

        return sample


class ContrastiveCrossModalDataset(Dataset):
    """
    Cross-modal contrastive dataset for known drug–target pairs.
    
    Phase 1 Enhanced: Returns TWO augmented views for EACH modality:
    - drug_view1, drug_view2 (for intra-modal drug contrastive loss)
    - prot_view1, prot_view2 (for intra-modal protein contrastive loss)
    - All views are from the same binding pair (aligned indices for cross-modal loss)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sml_stoi: Dict[str, int],
        prot_stoi: Dict[str, int],
        max_sml_len: int = 120,
        max_prot_len: int = 1000,
        drug_aug_names: List[str] | None = None,
        prot_aug_names: List[str] | None = None,
        mask_ratio: float = 0.15,
        drop_prob: float = 0.1,
        crop_min_ratio: float = 0.7,
        sub_ratio: float = 0.10,
        use_pretrained_tokenizers: bool = False,
        drug_tokenizer: Optional[Any] = None,
        prot_tokenizer: Optional[Any] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.sml_stoi = sml_stoi
        self.prot_stoi = prot_stoi
        self.max_sml_len = max_sml_len
        self.max_prot_len = max_prot_len
        self.drug_aug_names = drug_aug_names or ["smiles_enum", "atom_mask"]
        self.prot_aug_names = prot_aug_names or ["subseq_crop", "residue_mask"]
        self.drug_kwargs = {"mask_ratio": mask_ratio, "drop_prob": drop_prob}
        self.prot_kwargs = {
            "mask_ratio": mask_ratio,
            "min_ratio": crop_min_ratio,
            "sub_ratio": sub_ratio,
        }
        self.use_pretrained_tokenizers = use_pretrained_tokenizers or (drug_tokenizer is not None) or (prot_tokenizer is not None)
        self.drug_tokenizer = drug_tokenizer
        self.prot_tokenizer = prot_tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]

        # Generate two independent augmented views for drug
        drug_view1 = apply_random_augmentation(
            smiles, self.drug_aug_names, DRUG_AUGMENTATION_REGISTRY, **self.drug_kwargs
        )
        drug_view2 = apply_random_augmentation(
            smiles, self.drug_aug_names, DRUG_AUGMENTATION_REGISTRY, **self.drug_kwargs
        )
        
        # Generate two independent augmented views for protein
        prot_view1 = apply_random_augmentation(
            sequence, self.prot_aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.prot_kwargs
        )
        prot_view2 = apply_random_augmentation(
            sequence, self.prot_aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.prot_kwargs
        )

        # Tokenize all views (character-level baseline)
        d1_ids = tokenize_seq(drug_view1, self.sml_stoi, self.max_sml_len)
        d2_ids = tokenize_seq(drug_view2, self.sml_stoi, self.max_sml_len)
        p1_ids = tokenize_seq(prot_view1, self.prot_stoi, self.max_prot_len)
        p2_ids = tokenize_seq(prot_view2, self.prot_stoi, self.max_prot_len)

        sample = {
            "drug_view1": torch.LongTensor(d1_ids),
            "drug_view2": torch.LongTensor(d2_ids),
            "prot_view1": torch.LongTensor(p1_ids),
            "prot_view2": torch.LongTensor(p2_ids),
            "index": idx,
            # Raw augmented text (needed for LLM alignment and caching)
            "drug_view1_text": drug_view1,
            "drug_view2_text": drug_view2,
            "prot_view1_text": prot_view1,
            "prot_view2_text": prot_view2,
        }

        if self.use_pretrained_tokenizers:
            sample["drug_view1_tokens"] = maybe_tokenize_hf(drug_view1, self.drug_tokenizer)
            sample["drug_view2_tokens"] = maybe_tokenize_hf(drug_view2, self.drug_tokenizer)
            sample["prot_view1_tokens"] = maybe_tokenize_hf(prot_view1, self.prot_tokenizer)
            sample["prot_view2_tokens"] = maybe_tokenize_hf(prot_view2, self.prot_tokenizer)

        return sample
