"""
multitask.py — Phase 9/10: Multi-Task Learning for DTA.

Implements:
  - MultiTaskHead: predicts affinity (regression), interaction (binary), MoA (classification)
  - MultiTaskLoss: weighted combination with missing-label masking
  - Dynamic loss weighting (uncertainty-based)
  - Per-task metrics (RMSE, AUROC, F1)
  - evaluate_multitask(): comprehensive multi-task evaluation

The shared backbone (pretrained encoders + optional attention) feeds into
separate heads, each with its own loss function and gradient flow.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Prediction Head
# ─────────────────────────────────────────────────────────────────────────────


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head with three branches:
    1. Affinity regression (continuous value)
    2. Interaction classification (binding vs non-binding)
    3. Mechanism of Action classification (N-class, optional)

    Parameters
    ----------
    input_dim : int
        Dimension of shared feature vector (from encoder concatenation).
    num_moa_classes : int
        Number of MoA categories. Set to 0 to disable MoA head.
    hidden_dim : int
        Hidden layer dimension for each head.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_moa_classes: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_moa_classes = num_moa_classes

        # Shared feature transformation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 1. Affinity regression head
        self.affinity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 2. Interaction classification head (binary)
        self.interaction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 3. MoA classification head (optional)
        if num_moa_classes > 0:
            self.moa_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_dim // 2, num_moa_classes),
            )
        else:
            self.moa_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, input_dim) shared feature vector

        Returns
        -------
        dict with:
          'affinity': (B,) predicted binding affinity
          'interaction': (B,) interaction probability (sigmoid applied)
          'moa': (B, num_moa_classes) class logits (if enabled)
        """
        shared_features = self.shared(x)

        outputs = {
            "affinity": self.affinity_head(shared_features).squeeze(-1),
            "interaction": torch.sigmoid(
                self.interaction_head(shared_features).squeeze(-1)
            ),
        }

        if self.moa_head is not None:
            outputs["moa"] = self.moa_head(shared_features)

        return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss
# ─────────────────────────────────────────────────────────────────────────────


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss with missing-label masking.

    Loss = w1 * L_affinity + w2 * L_interaction + w3 * L_moa

    Missing labels (None / NaN) are automatically masked so they
    do not contribute to gradients.

    Parameters
    ----------
    loss_weights : dict
        Weights per task {'affinity': 1.0, 'interaction': 1.0, 'moa': 1.0}
    use_dynamic_weighting : bool
        If True, uses uncertainty-based dynamic weighting (Kendall et al. 2018)
    affinity_threshold : float
        Threshold for converting affinity to binary interaction label
        (used when interaction_label is not provided)
    """

    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        use_dynamic_weighting: bool = False,
        affinity_threshold: float = 7.0,
    ):
        super().__init__()
        self.weights = loss_weights or {
            "affinity": 1.0,
            "interaction": 1.0,
            "moa": 1.0,
        }
        self.use_dynamic_weighting = use_dynamic_weighting
        self.affinity_threshold = affinity_threshold

        if use_dynamic_weighting:
            # Learnable log-variance per task (Kendall multi-task uncertainty)
            self.log_var_affinity = nn.Parameter(torch.tensor(0.0))
            self.log_var_interaction = nn.Parameter(torch.tensor(0.0))
            self.log_var_moa = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss with masking.

        Parameters
        ----------
        predictions : dict from MultiTaskHead.forward()
        targets : dict with optional keys:
            'affinity': (B,) float — true binding affinity
            'interaction_label': (B,) float 0/1 — true interaction label
            'moa_label': (B,) long — MoA class indices (or None)

        Returns
        -------
        dict with 'loss_affinity', 'loss_interaction', 'loss_moa', 'loss_total'
        """
        losses = {}
        device = predictions["affinity"].device

        # 1. Affinity regression loss (MSE)
        if targets.get("affinity") is not None:
            aff_target = targets["affinity"].to(device).float()
            mask = ~torch.isnan(aff_target)
            if mask.any():
                losses["loss_affinity"] = F.mse_loss(
                    predictions["affinity"][mask], aff_target[mask]
                )
            else:
                losses["loss_affinity"] = torch.tensor(0.0, device=device)
        else:
            losses["loss_affinity"] = torch.tensor(0.0, device=device)

        # 2. Interaction classification loss (BCE)
        interaction_target = targets.get("interaction_label")
        if interaction_target is None and targets.get("affinity") is not None:
            # Derive from affinity if not provided
            aff = targets["affinity"].to(device).float()
            interaction_target = (aff >= self.affinity_threshold).float()

        if interaction_target is not None:
            interaction_target = interaction_target.to(device).float()
            mask = ~torch.isnan(interaction_target)
            if mask.any():
                losses["loss_interaction"] = F.binary_cross_entropy(
                    predictions["interaction"][mask].clamp(1e-7, 1 - 1e-7),
                    interaction_target[mask],
                )
            else:
                losses["loss_interaction"] = torch.tensor(0.0, device=device)
        else:
            losses["loss_interaction"] = torch.tensor(0.0, device=device)

        # 3. MoA classification loss (CrossEntropy)
        moa_target = targets.get("moa_label")
        if moa_target is not None and "moa" in predictions:
            moa_target = moa_target.to(device).long()
            # Mask: -1 or negative values indicate missing MoA label
            mask = moa_target >= 0
            if mask.any():
                losses["loss_moa"] = F.cross_entropy(
                    predictions["moa"][mask], moa_target[mask]
                )
            else:
                losses["loss_moa"] = torch.tensor(0.0, device=device)
        else:
            losses["loss_moa"] = torch.tensor(0.0, device=device)

        # Combine losses
        if self.use_dynamic_weighting:
            # Uncertainty-based weighting (Kendall et al.)
            w_aff = torch.exp(-self.log_var_affinity)
            w_int = torch.exp(-self.log_var_interaction)
            w_moa = torch.exp(-self.log_var_moa)

            total = (
                w_aff * losses["loss_affinity"] + self.log_var_affinity
                + w_int * losses["loss_interaction"] + self.log_var_interaction
                + w_moa * losses["loss_moa"] + self.log_var_moa
            )
        else:
            total = (
                self.weights["affinity"] * losses["loss_affinity"]
                + self.weights["interaction"] * losses["loss_interaction"]
                + self.weights["moa"] * losses["loss_moa"]
            )

        losses["loss_total"] = total
        return losses


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Training Utilities
# ─────────────────────────────────────────────────────────────────────────────


def train_multitask_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    grad_clip: float = 5.0,
    scaler=None,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Run one multi-task training epoch.

    Returns dict of average losses per task.
    """
    model.train()
    accum = {"loss_affinity": 0.0, "loss_interaction": 0.0, "loss_moa": 0.0, "loss_total": 0.0}
    cnt = 0
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        smiles = batch["smiles"].to(device, non_blocking=True)
        seq = batch["seq"].to(device, non_blocking=True)

        # Build targets dict
        targets = {}
        if "aff" in batch:
            targets["affinity"] = batch["aff"].squeeze(1)
        if "interaction_label" in batch:
            targets["interaction_label"] = batch["interaction_label"]
        if "moa_label" in batch:
            targets["moa_label"] = batch["moa_label"]

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predictions = model(smiles, seq)
            losses = loss_fn(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses["loss_total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = smiles.size(0)
        for k in accum:
            accum[k] += losses[k].item() * bs
        cnt += bs

    return {k: v / max(cnt, 1) for k, v in accum.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Evaluation
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_multitask(
    model: nn.Module,
    dataloader,
    device: torch.device,
    affinity_threshold: float = 7.0,
) -> Dict[str, float]:
    """
    Comprehensive multi-task evaluation.

    Metrics per task:
    - Affinity: RMSE, Pearson r
    - Interaction: AUROC, AUPRC
    - MoA: Accuracy, F1-score (macro)

    Parameters
    ----------
    model : nn.Module
        Multi-task model
    dataloader : DataLoader
    device : torch.device
    affinity_threshold : float
        For deriving interaction labels from affinity

    Returns
    -------
    dict with all per-task metrics
    """
    model.eval()

    all_aff_true, all_aff_pred = [], []
    all_int_true, all_int_pred = [], []
    all_moa_true, all_moa_pred = [], []

    for batch in dataloader:
        smiles = batch["smiles"].to(device, non_blocking=True)
        seq = batch["seq"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            predictions = model(smiles, seq)

        # Collect predictions
        if isinstance(predictions, dict):
            aff_pred = predictions["affinity"].cpu().numpy()
            int_pred = predictions["interaction"].cpu().numpy()
            moa_pred = predictions.get("moa")
            if moa_pred is not None:
                moa_pred = moa_pred.cpu().numpy()
        else:
            aff_pred = predictions.cpu().numpy()
            int_pred = None
            moa_pred = None

        # Collect true values
        if "aff" in batch:
            aff_true = batch["aff"].squeeze(1).numpy()
            all_aff_true.append(aff_true)
            all_aff_pred.append(aff_pred)

            # Derive interaction labels
            if int_pred is not None:
                int_true = (aff_true >= affinity_threshold).astype(float)
                if "interaction_label" in batch:
                    int_true = batch["interaction_label"].numpy()
                all_int_true.append(int_true)
                all_int_pred.append(int_pred)

        if "moa_label" in batch and moa_pred is not None:
            moa_true = batch["moa_label"].numpy()
            valid = moa_true >= 0
            if valid.any():
                all_moa_true.append(moa_true[valid])
                all_moa_pred.append(moa_pred[valid])

    metrics = {}

    # Affinity metrics
    if all_aff_true:
        y_true = np.concatenate(all_aff_true)
        y_pred = np.concatenate(all_aff_pred)
        metrics["affinity_rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        if len(y_true) > 1 and y_true.std() > 0 and y_pred.std() > 0:
            metrics["affinity_pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            metrics["affinity_pearson"] = 0.0

    # Interaction metrics
    if all_int_true:
        y_true = np.concatenate(all_int_true)
        y_pred = np.concatenate(all_int_pred)

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            if len(np.unique(y_true)) > 1:
                metrics["interaction_auroc"] = float(roc_auc_score(y_true, y_pred))
                metrics["interaction_auprc"] = float(average_precision_score(y_true, y_pred))
            else:
                metrics["interaction_auroc"] = 0.0
                metrics["interaction_auprc"] = 0.0
        except ImportError:
            # Fallback: just compute accuracy
            y_pred_binary = (y_pred >= 0.5).astype(float)
            metrics["interaction_accuracy"] = float((y_pred_binary == y_true).mean())

    # MoA metrics
    if all_moa_true:
        y_true = np.concatenate(all_moa_true)
        y_pred = np.concatenate(all_moa_pred)
        y_pred_classes = y_pred.argmax(axis=1)

        metrics["moa_accuracy"] = float((y_pred_classes == y_true).mean())

        try:
            from sklearn.metrics import f1_score
            metrics["moa_f1_macro"] = float(
                f1_score(y_true, y_pred_classes, average="macro", zero_division=0)
            )
        except ImportError:
            pass

    return metrics
