"""
meta_eval.py — Few-shot evaluation functions for meta-learned DTA models.

Evaluates performance with different k-shot scenarios (1, 5, 10) and
compares baseline (no adaptation) vs adapted (MAML-adapted) models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from .meta_train import few_shot_evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Few-Shot Evaluation Across Different Support Set Sizes
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_few_shot_performance(
    model: torch.nn.Module,
    meta_dataset,
    config: Any,  # MetaLearningConfig
    k_shots: List[int] = [1, 5, 10],
    k_query: int = 10,
    num_tasks_per_k: int = 50,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model performance across different few-shot scenarios.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained DTA model
    meta_dataset : MetaDTADataset
        Meta-learning dataset for task sampling
    config : MetaLearningConfig
        Meta-learning configuration
    k_shots : list of int
        Support set sizes to evaluate (e.g., [1, 5, 10])
    k_query : int
        Query set size
    num_tasks_per_k : int
        Number of tasks to evaluate per k-shot scenario
    device : str
        Device to compute on

    Returns
    -------
    dict
        Structure:
        {
            '1-shot': {
                'before': {...},  # Baseline (no adaptation)
                'after': {...},   # With MAML adaptation
                'improvement': {...}
            },
            '5-shot': {...},
            '10-shot': {...},
            'summary': {...}
        }
    """
    model.eval()
    results = {}

    print(f"\n[Few-Shot Evaluation] Testing k-shots: {k_shots}")

    for k in k_shots:
        print(f"\nEvaluating {k}-shot scenario ({num_tasks_per_k} tasks)...")

        metrics_before = defaultdict(list)
        metrics_after = defaultdict(list)
        improvements = defaultdict(list)

        for task_idx in range(num_tasks_per_k):
            # Sample task
            task = meta_dataset.sample_task(k_support=k, k_query=k_query)
            support_df, query_df = meta_dataset.get_task_data(task)

            # Placeholder: convert to model-ready format
            # (integrate with actual tokenizers/dataloader)
            support_batch = _df_to_batch(support_df, device)
            query_batch = _df_to_batch(query_df, device)
            support_labels = torch.tensor(
                support_df["affinity"].values, dtype=torch.float32, device=device
            )
            query_labels = torch.tensor(
                query_df["affinity"].values, dtype=torch.float32, device=device
            )

            # Evaluate
            eval_result = few_shot_evaluate(
                model,
                support_batch,
                support_labels,
                query_batch,
                query_labels,
                config,
                device=device,
            )

            metrics_before[f"{k}-shot"].append(eval_result["loss_before"])
            metrics_after[f"{k}-shot"].append(eval_result["loss_after"])
            improvements[f"{k}-shot"].append(eval_result["improvement_pct"])

        # Aggregate results for this k
        k_key = f"{k}-shot"
        results[k_key] = {
            "loss_before": {
                "mean": np.mean(metrics_before[k_key]),
                "std": np.std(metrics_before[k_key]),
            },
            "loss_after": {
                "mean": np.mean(metrics_after[k_key]),
                "std": np.std(metrics_after[k_key]),
            },
            "improvement_pct": {
                "mean": np.mean(improvements[k_key]),
                "std": np.std(improvements[k_key]),
            },
        }

        print(f"  Loss (before): {results[k_key]['loss_before']['mean']:.4f} ± {results[k_key]['loss_before']['std']:.4f}")
        print(f"  Loss (after):  {results[k_key]['loss_after']['mean']:.4f} ± {results[k_key]['loss_after']['std']:.4f}")
        print(f"  Improvement:   {results[k_key]['improvement_pct']['mean']:.2f} ± {results[k_key]['improvement_pct']['std']:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Task-Type Specific Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_by_task_type(
    model: torch.nn.Module,
    meta_dataset,
    config: Any,
    task_types: List[str] = ["cold_drug", "cold_target", "cold_both"],
    k_support: int = 5,
    k_query: int = 10,
    num_tasks_per_type: int = 50,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate few-shot performance stratified by task type.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained model
    meta_dataset : MetaDTADataset
        Meta-dataset
    config : MetaLearningConfig
        Config
    task_types : list of str
        Task types to evaluate
    k_support : int
        Support set size
    k_query : int
        Query set size
    num_tasks_per_type : int
        Tasks per type
    device : str
        Device

    Returns
    -------
    dict
        Results per task type with 'loss_before', 'loss_after', 'improvement'
    """
    model.eval()
    results = {}

    print(f"\n[Task-Type Evaluation] Testing task types: {task_types}")

    for task_type in task_types:
        print(f"\nEvaluating {task_type} tasks ({num_tasks_per_type} tasks)...")

        metrics_before = []
        metrics_after = []
        improvements = []

        for _ in range(num_tasks_per_type):
            task = meta_dataset.sample_task(
                k_support=k_support, k_query=k_query, task_type=task_type
            )

            support_df, query_df = meta_dataset.get_task_data(task)

            support_batch = _df_to_batch(support_df, device)
            query_batch = _df_to_batch(query_df, device)
            support_labels = torch.tensor(
                support_df["affinity"].values, dtype=torch.float32, device=device
            )
            query_labels = torch.tensor(
                query_df["affinity"].values, dtype=torch.float32, device=device
            )

            eval_result = few_shot_evaluate(
                model,
                support_batch,
                support_labels,
                query_batch,
                query_labels,
                config,
                device=device,
            )

            metrics_before.append(eval_result["loss_before"])
            metrics_after.append(eval_result["loss_after"])
            improvements.append(eval_result["improvement_pct"])

        results[task_type] = {
            "loss_before": {
                "mean": np.mean(metrics_before),
                "std": np.std(metrics_before),
            },
            "loss_after": {
                "mean": np.mean(metrics_after),
                "std": np.std(metrics_after),
            },
            "improvement_pct": {
                "mean": np.mean(improvements),
                "std": np.std(improvements),
            },
        }

        print(f"  Loss (before): {results[task_type]['loss_before']['mean']:.4f} ± {results[task_type]['loss_before']['std']:.4f}")
        print(f"  Loss (after):  {results[task_type]['loss_after']['mean']:.4f} ± {results[task_type]['loss_after']['std']:.4f}")
        print(f"  Improvement:   {results[task_type]['improvement_pct']['mean']:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Studies
# ─────────────────────────────────────────────────────────────────────────────


def ablation_adaptation_scope(
    model: torch.nn.Module,
    meta_dataset,
    config: Any,
    scopes: List[str] = ["head_only", "partial_encoder", "full"],
    k_support: int = 5,
    k_query: int = 10,
    num_tasks: int = 50,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Ablation: compare different adaptation scopes.

    Parameters
    ----------
    model : torch.nn.Module
        Model
    meta_dataset : MetaDTADataset
        Dataset
    config : MetaLearningConfig
        Config (will be modified)
    scopes : list of str
        Adaptation scopes to compare
    k_support, k_query : int
        Task sizes
    num_tasks : int
        Tasks per scope
    device : str
        Device

    Returns
    -------
    dict
        Results per scope
    """
    results = {}

    print(f"\n[Ablation: Adaptation Scope] Testing scopes: {scopes}")

    for scope in scopes:
        print(f"\nTesting adaptation_scope={scope} ({num_tasks} tasks)...")

        config.adaptation_scope = scope

        metrics_improvement = []

        for _ in range(num_tasks):
            task = meta_dataset.sample_task(k_support=k_support, k_query=k_query)
            support_df, query_df = meta_dataset.get_task_data(task)

            support_batch = _df_to_batch(support_df, device)
            query_batch = _df_to_batch(query_df, device)
            support_labels = torch.tensor(
                support_df["affinity"].values, dtype=torch.float32, device=device
            )
            query_labels = torch.tensor(
                query_df["affinity"].values, dtype=torch.float32, device=device
            )

            eval_result = few_shot_evaluate(
                model,
                support_batch,
                support_labels,
                query_batch,
                query_labels,
                config,
                device=device,
            )

            metrics_improvement.append(eval_result["improvement_pct"])

        results[scope] = {
            "improvement_pct": {
                "mean": np.mean(metrics_improvement),
                "std": np.std(metrics_improvement),
            }
        }

        print(f"  Improvement: {results[scope]['improvement_pct']['mean']:.2f}%")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _df_to_batch(df: pd.DataFrame, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dataframe to model-ready batch using character-level tokenization.

    Returns tuple of (smiles_ids, seq_ids).
    """
    from .tokenizers_and_datasets import char_tokenize

    batch_size = len(df)

    # Character-level tokenization - handle different column name variations
    if "smiles" in df.columns:
        smiles_list = df["smiles"].values
    elif "SMILES" in df.columns:
        smiles_list = df["SMILES"].values
    elif "Drug_SMILES" in df.columns:
        smiles_list = df["Drug_SMILES"].values
    else:
        smiles_list = []

    if "sequence" in df.columns:
        seq_list = df["sequence"].values
    elif "Protein_Seq" in df.columns:
        seq_list = df["Protein_Seq"].values
    elif "Target_Seq" in df.columns:
        seq_list = df["Target_Seq"].values
    else:
        seq_list = []

    # Tokenize SMILES
    smiles_ids = []
    for sml in smiles_list:
        sml = str(sml) if pd.notna(sml) else ""
        tokens = char_tokenize(sml, vocab=None, max_len=128)
        smiles_ids.append(tokens)
    smiles_ids = torch.tensor(smiles_ids, dtype=torch.long, device=device)

    # Tokenize protein sequences
    seq_ids = []
    for seq in seq_list:
        seq = str(seq) if pd.notna(seq) else ""
        tokens = char_tokenize(seq, vocab=None, max_len=512)
        seq_ids.append(tokens)
    seq_ids = torch.tensor(seq_ids, dtype=torch.long, device=device)

    return (smiles_ids, seq_ids)


def print_few_shot_results(results: Dict) -> None:
    """Pretty-print few-shot evaluation results."""
    print("\n" + "="*80)
    print("FEW-SHOT EVALUATION RESULTS")
    print("="*80)

    for k_shot, metrics in results.items():
        if k_shot == "summary":
            continue
        print(f"\n{k_shot.upper()}:")
        print(f"  Loss (no adaptation): {metrics['loss_before']['mean']:.4f} ± {metrics['loss_before']['std']:.4f}")
        print(f"  Loss (with MAML):     {metrics['loss_after']['mean']:.4f} ± {metrics['loss_after']['std']:.4f}")
        print(f"  Improvement:          {metrics['improvement_pct']['mean']:.2f}% ± {metrics['improvement_pct']['std']:.2f}%")

    print("\n" + "="*80)
