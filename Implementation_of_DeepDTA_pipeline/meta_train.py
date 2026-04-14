"""
meta_train.py — MAML (Model-Agnostic Meta-Learning) implementation for few-shot DTA.

Implements:
  - Inner loop: task-specific adaptation on support set
  - Outer loop: meta-update on query set results
  - Efficient gradient computation through adaptation steps
  - Support for different adaptation scopes (head_only, partial_encoder, full)
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# MAML Inner Loop (Task-Specific Adaptation)
# ─────────────────────────────────────────────────────────────────────────────


def get_adaptable_params(
    model: nn.Module,
    adaptation_scope: str = "head_only"
) -> List[nn.Parameter]:
    """
    Get parameters that should be adapted during inner loop.

    Parameters
    ----------
    model : nn.Module
        DTA model
    adaptation_scope : {'head_only', 'partial_encoder', 'full'}
        Which parameters to adapt

    Returns
    -------
    list of nn.Parameter
        Parameters to adapt
    """
    if adaptation_scope == "head_only":
        # Only prediction head (last FC layers)
        adaptable = []
        for name, param in model.named_parameters():
            if "fc" in name or "head" in name or "predict" in name:
                adaptable.append(param)
        if not adaptable:
            # Fallback: all parameters if no clear head found
            print("[Warning] No 'fc'/'head' params found, adapting all params")
            adaptable = list(model.parameters())
        return adaptable

    elif adaptation_scope == "partial_encoder":
        # Last K encoder layers + head
        adaptable = []
        for name, param in model.named_parameters():
            if "fc" in name or "head" in name:
                adaptable.append(param)
            # Add last layers from encoders
            elif "encoder" in name or "conv" in name:
                # Simple heuristic: if param is in last layers
                adaptable.append(param)
        return adaptable[:len(adaptable) // 2:] if len(adaptable) > 10 else adaptable

    elif adaptation_scope == "full":
        return list(model.parameters())

    else:
        raise ValueError(f"Unknown adaptation_scope: {adaptation_scope}")


def inner_loop_step(
    model: nn.Module,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    support_labels: torch.Tensor,
    inner_lr: float = 1e-3,
    adaptation_scope: str = "head_only",
    device: str = "cuda",
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Single inner loop step: adapt model on support set.

    Parameters
    ----------
    model : nn.Module
        Original model
    support_batch : tuple of (smiles_ids, seq_ids)
        Support batch from dataloader
    support_labels : torch.Tensor
        Support labels (shape: [batch_size])
    inner_lr : float
        Learning rate for inner loop
    adaptation_scope : str
        Which parameters to adapt
    device : str
        Device to compute on

    Returns
    -------
    adapted_model : nn.Module
        Model after one adaptation step
    support_loss : torch.Tensor
        Loss on support set (scalar)
    """
    # Create a shallow copy to avoid modifying original
    adapted_model = copy.deepcopy(model)
    adapted_model.train()

    # Forward pass on support set
    smiles_ids, seq_ids = support_batch
    outputs = adapted_model(smiles_ids, seq_ids)  # shape: [batch_size]

    # Compute support loss (MSE for regression task)
    support_loss = F.mse_loss(outputs.squeeze(), support_labels)

    # Get adaptable parameters
    adaptable_params = get_adaptable_params(adapted_model, adaptation_scope)

    # Compute gradients only for adaptable parameters
    grads = torch.autograd.grad(
        support_loss,
        adaptable_params,
        create_graph=True,  # Keep graph for meta-grad computation
        retain_graph=True,
    )

    # Update adaptable parameters with SGD step
    with torch.no_grad():
        for param, grad in zip(adaptable_params, grads):
            if grad is not None:
                param.sub_(inner_lr * grad)

    return adapted_model, support_loss


def inner_loop(
    model: nn.Module,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    support_labels: torch.Tensor,
    num_inner_steps: int = 3,
    inner_lr: float = 1e-3,
    adaptation_scope: str = "head_only",
    device: str = "cuda",
) -> Tuple[nn.Module, List[torch.Tensor]]:
    """
    Complete inner loop: multiple adaptation steps on support set.

    Parameters
    ----------
    model : nn.Module
        Original model
    support_batch : tuple of (smiles_ids, seq_ids)
        Support batch
    support_labels : torch.Tensor
        Support labels
    num_inner_steps : int
        Number of gradient steps
    inner_lr : float
        Inner loop learning rate
    adaptation_scope : str
        Adaptation scope
    device : str
        Device

    Returns
    -------
    adapted_model : nn.Module
        Adapted model after num_inner_steps
    losses : list of torch.Tensor
        Loss at each step (for logging)
    """
    adapted_model = model
    losses = []

    for step in range(num_inner_steps):
        adapted_model, loss = inner_loop_step(
            adapted_model,
            support_batch,
            support_labels,
            inner_lr=inner_lr,
            adaptation_scope=adaptation_scope,
            device=device,
        )
        losses.append(loss.detach())

    return adapted_model, losses


# ─────────────────────────────────────────────────────────────────────────────
# MAML Meta-Training Loop
# ─────────────────────────────────────────────────────────────────────────────


def meta_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    task_batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    config: Any,  # MetaLearningConfig
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Single MAML meta-training step (one meta-batch of tasks).

    Parameters
    ----------
    model : nn.Module
        Original model (updated after this step)
    optimizer : torch.optim.Optimizer
        Meta-optimizer (Adam)
    task_batch : list
        List of tasks, each task = (support_batch, support_labels, query_batch, query_labels)
        where support_batch and query_batch are tuples of (smiles_ids, seq_ids)
    config : MetaLearningConfig
        Meta-learning configuration
    device : str
        Device

    Returns
    -------
    dict
        Metrics: {'meta_loss', 'support_loss', 'query_loss', 'adaptation_improvement'}
    """
    model.train()
    optimizer.zero_grad()

    query_losses = []
    support_losses = []
    adaptation_improvements = []

    # Accumulate gradients across tasks
    for support_batch, support_labels, query_batch, query_labels in task_batch:
        # Move data to device
        support_labels = support_labels.to(device).float()
        query_labels = query_labels.to(device).float()

        # Inner loop: adapt to support set
        adapted_model, inner_losses = inner_loop(
            model,
            support_batch,
            support_labels,
            num_inner_steps=config.num_inner_steps,
            inner_lr=config.inner_lr,
            adaptation_scope=config.adaptation_scope,
            device=device,
        )

        support_loss = inner_losses[-1]  # Loss after adaptation
        support_losses.append(support_loss.item())

        # Query set evaluation (meta-loss computation)
        adapted_model.eval()
        with torch.enable_grad():  # Keep gradients for meta-update
            smiles_ids, seq_ids = query_batch
            query_outputs = adapted_model(smiles_ids, seq_ids)
            query_loss = F.mse_loss(query_outputs.squeeze(), query_labels)

        query_losses.append(query_loss.item())

        # Compute adaptation improvement
        with torch.no_grad():
            smiles_ids, seq_ids = query_batch
            original_outputs = model(smiles_ids, seq_ids)
            original_loss = F.mse_loss(original_outputs.squeeze(), query_labels)
            improvement = (original_loss - query_loss).item()
            adaptation_improvements.append(improvement)

        # Backprop through adaptation steps to compute gradients for meta-model
        query_loss.backward()

    # Meta-update: step on aggregated gradients
    optimizer.step()

    return {
        "meta_loss": np.mean(query_losses),
        "support_loss": np.mean(support_losses),
        "query_loss": np.mean(query_losses),
        "adaptation_improvement": np.mean(adaptation_improvements),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Few-Shot Evaluation (Without Gradient Updates)
# ─────────────────────────────────────────────────────────────────────────────


def few_shot_evaluate(
    model: nn.Module,
    support_batch: Tuple[torch.Tensor, torch.Tensor],
    support_labels: torch.Tensor,
    query_batch: Tuple[torch.Tensor, torch.Tensor],
    query_labels: torch.Tensor,
    config: Any,  # MetaLearningConfig
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate model with few-shot adaptation (without updating original model).

    Parameters
    ----------
    model : nn.Module
        Pretrained model
    support_batch : tuple of (smiles_ids, seq_ids)
        Support set batch
    support_labels : torch.Tensor
        Support labels
    query_batch : tuple of (smiles_ids, seq_ids)
        Query set batch
    query_labels : torch.Tensor
        Query labels
    config : MetaLearningConfig
        Meta-learning config
    device : str
        Device

    Returns
    -------
    dict
        Metrics: {'loss_before', 'loss_after', 'improvement_pct'}
    """
    model.eval()
    support_labels = support_labels.to(device).float()
    query_labels = query_labels.to(device).float()

    # Evaluate BEFORE adaptation
    with torch.no_grad():
        smiles_ids, seq_ids = query_batch
        outputs_before = model(smiles_ids, seq_ids)
        loss_before = F.mse_loss(outputs_before.squeeze(), query_labels)

    # Adapt on support set (no gradient tracking for meta-model)
    adapted_model, _ = inner_loop(
        model,
        support_batch,
        support_labels,
        num_inner_steps=config.num_inner_steps,
        inner_lr=config.inner_lr,
        adaptation_scope=config.adaptation_scope,
        device=device,
    )

    # Evaluate AFTER adaptation
    adapted_model.eval()
    with torch.no_grad():
        smiles_ids, seq_ids = query_batch
        outputs_after = adapted_model(smiles_ids, seq_ids)
        loss_after = F.mse_loss(outputs_after.squeeze(), query_labels)

    improvement = ((loss_before - loss_after) / (loss_before + 1e-8) * 100).item()

    return {
        "loss_before": loss_before.item(),
        "loss_after": loss_after.item(),
        "improvement_pct": improvement,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch Collate (Convert DataFrame to Model-Ready Format)
# ─────────────────────────────────────────────────────────────────────────────


def collate_dta_batch(
    df,
    tokenizer=None,
    max_sml_len: int = 128,
    max_prot_len: int = 512,
    device: str = "cuda",
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Convert dataframe batch to model-ready tensors.

    Parameters
    ----------
    df : pd.DataFrame
        Batch dataframe with columns: ['SMILES', 'Protein_Seq', 'Binding_Affinity']
    tokenizer : optional
        Pretrained tokenizer (if using Phase 4)
    max_sml_len : int
        Max SMILES length
    max_prot_len : int
        Max protein length
    device : str
        Device

    Returns
    -------
    batch : tuple of (smiles_ids, seq_ids)
        Character-level tokenized SMILES and protein sequences
    labels : torch.Tensor
        Binding affinity labels
    """
    from .tokenizers_and_datasets import char_tokenize

    batch_size = len(df)

    # Character-level tokenization (simple)
    # Try different column name variations
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
        tokens = char_tokenize(sml, vocab=None, max_len=max_sml_len)
        smiles_ids.append(tokens)
    smiles_ids = torch.tensor(smiles_ids, dtype=torch.long, device=device)

    # Tokenize protein sequences
    seq_ids = []
    for seq in seq_list:
        seq = str(seq) if pd.notna(seq) else ""
        tokens = char_tokenize(seq, vocab=None, max_len=max_prot_len)
        seq_ids.append(tokens)
    seq_ids = torch.tensor(seq_ids, dtype=torch.long, device=device)

    if "affinity" in df.columns:
        affinity_col = "affinity"
    elif "Binding_Affinity" in df.columns:
        affinity_col = "Binding_Affinity"
    else:
        affinity_col = "affinity"

    labels = torch.tensor(
        df[affinity_col].values, dtype=torch.float32, device=device
    )

    return (smiles_ids, seq_ids), labels


# ─────────────────────────────────────────────────────────────────────────────
# High-Level Training Interface
# ─────────────────────────────────────────────────────────────────────────────


class MetaTrainer:
    """
    Convenient wrapper for MAML meta-training.

    Usage:
        trainer = MetaTrainer(model, config, device='cuda')
        trainer.train_epoch(meta_dataset, dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,  # MetaLearningConfig
        meta_lr: float = 1e-4,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device
        self.meta_optimizer = Adam(model.parameters(), lr=meta_lr, weight_decay=1e-5)

        self.epoch_metrics = defaultdict(list)
        self.best_query_loss = float("inf")

    def train_step(
        self,
        task_batch: List[Tuple],
    ) -> Dict[str, float]:
        """Execute single meta-training step."""
        return meta_train_step(
            self.model,
            self.meta_optimizer,
            task_batch,
            self.config,
            device=self.device,
        )

    def train_epoch(
        self,
        meta_dataset,
        num_tasks_per_epoch: int = 100,
    ) -> Dict[str, float]:
        """Execute full meta-training epoch."""
        metrics_accum = defaultdict(list)

        num_meta_batches = num_tasks_per_epoch // self.config.meta_batch_size

        for batch_idx in range(num_meta_batches):
            # Sample task batch
            tasks = meta_dataset.sample_task_batch(
                self.config.meta_batch_size,
                self.config.k_support,
                self.config.k_query,
            )

            # Convert tasks to (support_batch, support_labels, query_batch, query_labels)
            task_batch_data = []
            for task in tasks:
                support_df, query_df = meta_dataset.get_task_data(task)

                # Collate batches
                support_batch, support_labels = collate_dta_batch(
                    support_df,
                    max_sml_len=128,
                    max_prot_len=512,
                    device=self.device,
                )
                query_batch, query_labels = collate_dta_batch(
                    query_df,
                    max_sml_len=128,
                    max_prot_len=512,
                    device=self.device,
                )

                task_batch_data.append(
                    (support_batch, support_labels, query_batch, query_labels)
                )

            # Meta-training step
            metrics = self.train_step(task_batch_data)

            for k, v in metrics.items():
                metrics_accum[k].append(v)

        # Aggregate epoch metrics
        epoch_metrics = {
            k: np.mean(v) for k, v in metrics_accum.items()
        }

        return epoch_metrics

    def evaluate_epoch(
        self,
        meta_dataset,
        num_tasks: int = 100,
    ) -> Dict[str, float]:
        """Evaluate on tasks without gradient updates."""
        self.model.eval()

        metrics_accum = defaultdict(list)

        for _ in range(num_tasks):
            task = meta_dataset.sample_task(
                self.config.k_support,
                self.config.k_query,
            )

            support_df, query_df = meta_dataset.get_task_data(task)

            support_batch, support_labels = collate_dta_batch(
                support_df,
                max_sml_len=128,
                max_prot_len=512,
                device=self.device,
            )
            query_batch, query_labels = collate_dta_batch(
                query_df,
                max_sml_len=128,
                max_prot_len=512,
                device=self.device,
            )

            metrics = few_shot_evaluate(
                self.model,
                support_batch,
                support_labels,
                query_batch,
                query_labels,
                self.config,
                device=self.device,
            )

            for k, v in metrics.items():
                metrics_accum[k].append(v)

        eval_metrics = {k: np.mean(v) for k, v in metrics_accum.items()}
        return eval_metrics
