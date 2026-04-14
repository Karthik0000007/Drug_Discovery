"""
meta_dataset.py — Meta-learning task (episode) construction for few-shot DTA.

Samples cold-start scenarios:
  - Cold-drug: unseen drug, seen proteins
  - Cold-target: seen drugs, unseen proteins
  - Cold-both: unseen drug AND protein

Each task contains support set (few-shot training) and query set (evaluation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path


class MetaDTADataset:
    """
    Meta-learning dataset for few-shot drug-target interaction prediction.

    Samples tasks (episodes) where each task simulates a cold-start scenario.

    Parameters
    ----------
    df : pd.DataFrame
        DTA dataframe with columns: ['Drug_ID', 'Target_ID', 'Binding_Affinity']
        (or similar SMILES/Protein columns)
    split_type : {'cold_drug', 'cold_target', 'cold_both', 'mixed'}
        Type of cold-start scenario(s) to sample
    seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        df: pd.DataFrame,
        drug_col: str = "Drug_ID",
        target_col: str = "Target_ID",
        affinity_col: str = "Binding_Affinity",
        split_type: Literal["cold_drug", "cold_target", "cold_both", "mixed"] = "mixed",
        seed: int = 42,
    ):
        """Initialize meta-learning dataset."""
        self.df = df.copy()
        self.drug_col = drug_col
        self.target_col = target_col
        self.affinity_col = affinity_col
        self.split_type = split_type
        self.seed = seed

        np.random.seed(seed)

        # Extract unique entities
        self.unique_drugs = self.df[drug_col].unique()
        self.unique_targets = self.df[target_col].unique()

        # Build entity→pairs mapping for efficient sampling
        self.drug_to_pairs = self._build_entity_to_pairs(drug_col)
        self.target_to_pairs = self._build_entity_to_pairs(target_col)

        self.n_drugs = len(self.unique_drugs)
        self.n_targets = len(self.unique_targets)

        print(f"[MetaDTADataset] Loaded {len(self.df)} pairs, "
              f"{self.n_drugs} drugs, {self.n_targets} targets")

    def _build_entity_to_pairs(self, entity_col: str) -> Dict:
        """Build mapping: entity_id → list of row indices with that entity."""
        entity_to_pairs = {}
        for entity in self.df[entity_col].unique():
            indices = self.df[self.df[entity_col] == entity].index.tolist()
            entity_to_pairs[entity] = indices
        return entity_to_pairs

    def sample_task(
        self,
        k_support: int = 5,
        k_query: int = 10,
        task_type: Optional[str] = None,
    ) -> Dict:
        """
        Sample a single meta-learning task.

        Parameters
        ----------
        k_support : int
            Number of support samples (few-shot training)
        k_query : int
            Number of query samples (evaluation)
        task_type : {'cold_drug', 'cold_target', 'cold_both', None}
            If None, randomly sample from split_type

        Returns
        -------
        dict
            Task with keys:
            - 'support_indices': list of support row indices
            - 'query_indices': list of query row indices (disjoint from support)
            - 'task_type': type of cold-start scenario
            - 'drug_id': specific drug (if cold_drug or cold_both)
            - 'target_id': specific target (if cold_target or cold_both)
        """
        if task_type is None:
            if self.split_type == "mixed":
                task_type = np.random.choice(
                    ["cold_drug", "cold_target", "cold_both"],
                    p=[0.33, 0.33, 0.34]
                )
            else:
                task_type = self.split_type

        if task_type == "cold_drug":
            return self._sample_cold_drug_task(k_support, k_query)
        elif task_type == "cold_target":
            return self._sample_cold_target_task(k_support, k_query)
        elif task_type == "cold_both":
            return self._sample_cold_both_task(k_support, k_query)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def _sample_cold_drug_task(self, k_support: int, k_query: int) -> Dict:
        """
        Cold-drug task: select a drug, sample its interactions as (support, query).

        Ensures query set is disjoint from support set.
        """
        # Sample a drug uniformly
        drug_id = np.random.choice(self.unique_drugs)

        # Find all pairs involving this drug
        drug_pairs = self.drug_to_pairs[drug_id]

        if len(drug_pairs) < k_support + k_query:
            # Not enough samples; pad with remaining or return smaller task
            available = len(drug_pairs)
            if available < k_support:
                # Fallback: sample from other drugs
                return self._sample_cold_drug_task_relaxed(drug_id, k_support, k_query)

        # Shuffle and split
        shuffled = np.random.permutation(drug_pairs)
        support_indices = shuffled[:k_support].tolist()
        query_indices = shuffled[k_support:k_support + k_query].tolist()

        return {
            "support_indices": support_indices,
            "query_indices": query_indices,
            "task_type": "cold_drug",
            "cold_entity": drug_id,
        }

    def _sample_cold_drug_task_relaxed(
        self, drug_id: str, k_support: int, k_query: int
    ) -> Dict:
        """Fallback for cold-drug with insufficient data: mix this drug's and other drugs."""
        drug_pairs = self.drug_to_pairs[drug_id]
        n_available = len(drug_pairs)

        support_from_drug = min(k_support, n_available)
        support_indices = np.random.choice(
            drug_pairs, size=support_from_drug, replace=False
        ).tolist()

        # Query from other drugs (to maintain cold-drug scenario)
        other_drugs = [d for d in self.unique_drugs if d != drug_id]
        query_indices = []
        for _ in range(k_query):
            other_drug = np.random.choice(other_drugs)
            candidates = [
                idx for idx in self.drug_to_pairs[other_drug]
                if idx not in support_indices
            ]
            if candidates:
                query_indices.append(np.random.choice(candidates))

        return {
            "support_indices": support_indices,
            "query_indices": query_indices[:k_query],
            "task_type": "cold_drug_relaxed",
            "cold_entity": drug_id,
        }

    def _sample_cold_target_task(self, k_support: int, k_query: int) -> Dict:
        """
        Cold-target task: select a target, sample its interactions as (support, query).
        """
        target_id = np.random.choice(self.unique_targets)
        target_pairs = self.target_to_pairs[target_id]

        if len(target_pairs) < k_support + k_query:
            return self._sample_cold_target_task_relaxed(target_id, k_support, k_query)

        shuffled = np.random.permutation(target_pairs)
        support_indices = shuffled[:k_support].tolist()
        query_indices = shuffled[k_support:k_support + k_query].tolist()

        return {
            "support_indices": support_indices,
            "query_indices": query_indices,
            "task_type": "cold_target",
            "cold_entity": target_id,
        }

    def _sample_cold_target_task_relaxed(
        self, target_id: str, k_support: int, k_query: int
    ) -> Dict:
        """Fallback for cold-target."""
        target_pairs = self.target_to_pairs[target_id]
        n_available = len(target_pairs)

        support_from_target = min(k_support, n_available)
        support_indices = np.random.choice(
            target_pairs, size=support_from_target, replace=False
        ).tolist()

        other_targets = [t for t in self.unique_targets if t != target_id]
        query_indices = []
        for _ in range(k_query):
            other_target = np.random.choice(other_targets)
            candidates = [
                idx for idx in self.target_to_pairs[other_target]
                if idx not in support_indices
            ]
            if candidates:
                query_indices.append(np.random.choice(candidates))

        return {
            "support_indices": support_indices,
            "query_indices": query_indices[:k_query],
            "task_type": "cold_target_relaxed",
            "cold_entity": target_id,
        }

    def _sample_cold_both_task(self, k_support: int, k_query: int) -> Dict:
        """
        Cold-both task: select drug AND target (with minimal prior interactions).
        """
        # Sample drug and target
        drug_id = np.random.choice(self.unique_drugs)
        target_id = np.random.choice(self.unique_targets)

        # Find pairs involving this (drug, target) combination
        mask_drug = self.df[self.drug_col] == drug_id
        mask_target = self.df[self.target_col] == target_id
        mask_both = mask_drug & mask_target

        both_indices = self.df[mask_both].index.tolist()

        if len(both_indices) >= k_support + k_query:
            # Use only this (drug, target) combination
            shuffled = np.random.permutation(both_indices)
            support_indices = shuffled[:k_support].tolist()
            query_indices = shuffled[k_support:k_support + k_query].tolist()
        else:
            # Mix: support from (drug, target), query from other interactions
            support_indices = both_indices

            # Query from interactions with this drug OR this target (but not both)
            other_indices = self.df[mask_drug | mask_target].index.tolist()
            other_indices = [idx for idx in other_indices if idx not in support_indices]

            if len(other_indices) < k_query:
                # Last resort: query from all data
                all_indices = [
                    idx for idx in self.df.index if idx not in support_indices
                ]
                query_indices = np.random.choice(
                    all_indices, size=k_query, replace=True
                ).tolist()
            else:
                query_indices = np.random.choice(
                    other_indices, size=k_query, replace=False
                ).tolist()

        return {
            "support_indices": support_indices,
            "query_indices": query_indices,
            "task_type": "cold_both",
            "cold_drug": drug_id,
            "cold_target": target_id,
        }

    def sample_task_batch(
        self,
        meta_batch_size: int,
        k_support: int,
        k_query: int,
    ) -> List[Dict]:
        """
        Sample a batch of independent meta-learning tasks.

        Parameters
        ----------
        meta_batch_size : int
            Number of tasks to sample
        k_support : int
            Support set size per task
        k_query : int
            Query set size per task

        Returns
        -------
        list of dicts
            Each dict is a single task
        """
        tasks = []
        for _ in range(meta_batch_size):
            task = self.sample_task(k_support=k_support, k_query=k_query)
            tasks.append(task)
        return tasks

    def get_task_data(self, task: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract support and query dataframes from task dict.

        Returns
        -------
        support_df, query_df : pd.DataFrame
            Support and query sets
        """
        support_df = self.df.iloc[task["support_indices"]].reset_index(drop=True)
        query_df = self.df.iloc[task["query_indices"]].reset_index(drop=True)
        return support_df, query_df

    def get_task_batch_data(
        self, tasks: List[Dict]
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """Extract support and query dataframes for task batch."""
        support_list = []
        query_list = []
        for task in tasks:
            sup, qry = self.get_task_data(task)
            support_list.append(sup)
            query_list.append(qry)
        return support_list, query_list
