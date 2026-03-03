"""
contrastive_dataset.py — ContrastiveDataset for positive pair generation.

For each sample, two independent augmentations are applied to produce views
(x_i, x_i^+). Negative pairs come from other samples in the minibatch
(NT-Xent style — no explicit negative sampling).
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augmentations import (
    apply_random_augmentation,
    DRUG_AUGMENTATION_REGISTRY,
    PROTEIN_AUGMENTATION_REGISTRY,
)
from .tokenizers_and_datasets import tokenize_seq


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
    ):
        self.smiles_list = smiles_list
        self.stoi = stoi
        self.max_len = max_len
        self.aug_names = aug_names or ["smiles_enum", "atom_mask"]
        self.aug_kwargs = {"mask_ratio": mask_ratio, "drop_prob": drop_prob}

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
        return {
            "view1": torch.LongTensor(v1_ids),
            "view2": torch.LongTensor(v2_ids),
            "index": idx,
        }


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

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        original = self.sequences[idx]
        view1 = apply_random_augmentation(
            original, self.aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        view2 = apply_random_augmentation(
            original, self.aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.aug_kwargs
        )
        v1_ids = tokenize_seq(view1, self.stoi, self.max_len)
        v2_ids = tokenize_seq(view2, self.stoi, self.max_len)
        return {
            "view1": torch.LongTensor(v1_ids),
            "view2": torch.LongTensor(v2_ids),
            "index": idx,
        }


class ContrastiveCrossModalDataset(Dataset):
    """
    Cross-modal contrastive dataset for known drug–target pairs.
    Returns augmented drug view + augmented protein view for the same pair.
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        smiles = row["smiles"]
        sequence = row["sequence"]

        drug_view = apply_random_augmentation(
            smiles, self.drug_aug_names, DRUG_AUGMENTATION_REGISTRY, **self.drug_kwargs
        )
        prot_view = apply_random_augmentation(
            sequence, self.prot_aug_names, PROTEIN_AUGMENTATION_REGISTRY, **self.prot_kwargs
        )

        d_ids = tokenize_seq(drug_view, self.sml_stoi, self.max_sml_len)
        p_ids = tokenize_seq(prot_view, self.prot_stoi, self.max_prot_len)

        return {
            "drug_view": torch.LongTensor(d_ids),
            "prot_view": torch.LongTensor(p_ids),
            "index": idx,
        }
