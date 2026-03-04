"""
train.py — Supervised training loop for DTA models.

Features:
  - MSE regression loss with gradient clipping
  - TensorBoard logging (loss, RMSE, CI, learning rate)
  - Early stopping with patience
  - Freeze-strategy support (frozen / full_finetune / gradual_unfreeze)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from typing import Optional


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    """Run one training epoch. Returns average MSE loss."""
    model.train()
    total_loss = 0.0
    cnt = 0
    for batch in loader:
        smiles = batch["smiles"].to(device)
        seq = batch["seq"].to(device)
        aff = batch["aff"].to(device).squeeze(1)
        pred = model(smiles, seq)
        loss = nn.functional.mse_loss(pred, aff)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * smiles.size(0)
        cnt += smiles.size(0)
    return total_loss / max(cnt, 1)


@torch.no_grad()
def eval_model(model: nn.Module, loader, device: torch.device):
    """Evaluate model. Returns (y_true, y_pred) as numpy arrays."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        smiles = batch["smiles"].to(device)
        seq = batch["seq"].to(device)
        aff = batch["aff"].to(device).squeeze(1)
        pred = model(smiles, seq)
        preds.append(pred.cpu().numpy())
        trues.append(aff.cpu().numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ──────────────────────────────────────────────
# Full training loop with logging
# ──────────────────────────────────────────────

def train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int = 30,
    patience: int = 8,
    grad_clip: float = 5.0,
    freeze_strategy: str = "full_finetune",
    unfreeze_after: int = 5,
    tb_writer: Optional[SummaryWriter] = None,
    experiment_tag: str = "",
) -> dict:
    """
    Full training loop with early stopping and optional TensorBoard logging.

    Parameters
    ----------
    freeze_strategy : 'frozen' | 'full_finetune' | 'gradual_unfreeze'
    unfreeze_after : epochs after which to unfreeze encoders (gradual_unfreeze only)

    Returns
    -------
    dict with 'best_model_state', 'train_losses', 'val_rmses', per-epoch info.
    """
    from .utilities import rmse as calc_rmse, ci_auto

    # Apply freeze strategy
    if freeze_strategy == "frozen" and hasattr(model, "freeze_encoders"):
        model.freeze_encoders()
        print("[train] Encoders frozen — training FC head only.")
    elif freeze_strategy == "gradual_unfreeze" and hasattr(model, "freeze_encoders"):
        model.freeze_encoders()
        print(f"[train] Gradual unfreeze — encoders frozen for first {unfreeze_after} epochs.")

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses = []
    val_rmses = []
    val_cis = []

    for epoch in range(1, epochs + 1):
        # Gradual unfreeze: unfreeze after N epochs
        if (freeze_strategy == "gradual_unfreeze"
                and epoch == unfreeze_after + 1
                and hasattr(model, "unfreeze_encoders")):
            model.unfreeze_encoders()
            print(f"[train] Epoch {epoch}: Unfreezing encoders.")

        train_loss = train_epoch(model, train_loader, optimizer, device, grad_clip)
        trues_val, preds_val = eval_model(model, val_loader, device)
        val_rmse_val = calc_rmse(trues_val, preds_val)
        val_ci_val = ci_auto(trues_val, preds_val)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_rmse_val)
            else:
                scheduler.step()

        train_losses.append(train_loss)
        val_rmses.append(val_rmse_val)
        val_cis.append(val_ci_val)

        # TensorBoard
        if tb_writer is not None:
            prefix = f"{experiment_tag}/" if experiment_tag else ""
            tb_writer.add_scalar(f"{prefix}train/loss", train_loss, epoch)
            tb_writer.add_scalar(f"{prefix}val/rmse", val_rmse_val, epoch)
            tb_writer.add_scalar(f"{prefix}val/ci", val_ci_val, epoch)
            tb_writer.add_scalar(
                f"{prefix}train/lr", optimizer.param_groups[0]["lr"], epoch
            )

        print(
            f"[Epoch {epoch:3d}/{epochs}] "
            f"TrainLoss: {train_loss:.4f}  ValRMSE: {val_rmse_val:.4f}  "
            f"ValCI: {val_ci_val:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Best model
        if val_rmse_val < best_val_rmse:
            best_val_rmse = val_rmse_val
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[train] No improvement for {patience} epochs — early stopping.")
            break

    return {
        "best_model_state": best_state,
        "best_val_rmse": best_val_rmse,
        "train_losses": train_losses,
        "val_rmses": val_rmses,
        "val_cis": val_cis,
        "epochs_trained": len(train_losses),
    }