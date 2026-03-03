"""
pretrain.py — Contrastive pretraining loop for CL-DTA.

Modes:
  drug_only         — Pretrain drug CNN encoder only
  prot_only         — Pretrain protein CNN encoder only
  both_independent  — Pretrain both encoders independently (sequential)
  cross_modal       — Joint pretraining with cross-modal alignment loss

Usage (CLI):
  python -m Implementation_of_DeepDTA_pipeline.pretrain \
      --data data/davis_processed.csv --mode both_independent \
      --epochs 100 --batch 256 --temperature 0.07 \
      --out checkpoints/pretrained/
"""

from __future__ import annotations

import os
import time
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .model import DrugEncoder, ProteinEncoder, ProjectionHead
from .tokenizers_and_datasets import build_vocab
from .contrastive_dataset import (
    ContrastiveDrugDataset,
    ContrastiveProteinDataset,
    ContrastiveCrossModalDataset,
)
from .contrastive_losses import get_contrastive_loss
from .utilities import set_seed

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Single-modality pretraining
# ──────────────────────────────────────────────

def pretrain_encoder(
    encoder: torch.nn.Module,
    proj_head: ProjectionHead,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    tb_writer: SummaryWriter | None = None,
    tag: str = "drug",
    save_dir: str = "checkpoints/pretrained/",
    save_every: int = 25,
) -> float:
    """
    Train a single encoder + projection head with contrastive loss.

    Returns the final epoch loss.
    """
    os.makedirs(save_dir, exist_ok=True)
    encoder.to(device)
    proj_head.to(device)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            v1 = batch["view1"].to(device)
            v2 = batch["view2"].to(device)

            h1 = encoder(v1)
            h2 = encoder(v2)
            z1 = proj_head(h1)
            z2 = proj_head(h2)

            loss = loss_fn(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(proj_head.parameters()), 5.0
            )
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if scheduler is not None:
            scheduler.step()

        if tb_writer is not None:
            tb_writer.add_scalar(f"pretrain/{tag}_loss", avg_loss, epoch)
            tb_writer.add_scalar(
                f"pretrain/{tag}_lr", optimizer.param_groups[0]["lr"], epoch
            )

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "[Pretrain %s] Epoch %d/%d  Loss: %.4f  LR: %.2e",
                tag, epoch, epochs, avg_loss, optimizer.param_groups[0]["lr"],
            )
            print(
                f"[Pretrain {tag}] Epoch {epoch}/{epochs}  "
                f"Loss: {avg_loss:.4f}  LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        # Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
        if epoch % save_every == 0 or epoch == epochs:
            ckpt = {
                "encoder": encoder.state_dict(),
                "proj_head": proj_head.state_dict(),
                "epoch": epoch,
                "loss": avg_loss,
            }
            path = os.path.join(save_dir, f"{tag}_encoder.pt")
            torch.save(ckpt, path)

    return best_loss


# ──────────────────────────────────────────────
# Cross-modal pretraining
# ──────────────────────────────────────────────

def pretrain_cross_modal(
    drug_encoder: DrugEncoder,
    prot_encoder: ProteinEncoder,
    drug_proj: ProjectionHead,
    prot_proj: ProjectionHead,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    tb_writer: SummaryWriter | None = None,
    save_dir: str = "checkpoints/pretrained/",
    save_every: int = 25,
) -> float:
    """Cross-modal alignment: align drug–protein representations."""
    os.makedirs(save_dir, exist_ok=True)
    drug_encoder.to(device)
    prot_encoder.to(device)
    drug_proj.to(device)
    prot_proj.to(device)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        drug_encoder.train()
        prot_encoder.train()
        drug_proj.train()
        prot_proj.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            d = batch["drug_view"].to(device)
            p = batch["prot_view"].to(device)

            hd = drug_encoder(d)
            hp = prot_encoder(p)
            zd = drug_proj(hd)
            zp = prot_proj(hp)

            loss = loss_fn(zd, zp)

            optimizer.zero_grad()
            loss.backward()
            all_params = (
                list(drug_encoder.parameters()) + list(prot_encoder.parameters())
                + list(drug_proj.parameters()) + list(prot_proj.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if scheduler is not None:
            scheduler.step()

        if tb_writer is not None:
            tb_writer.add_scalar("pretrain/cross_modal_loss", avg_loss, epoch)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Pretrain cross-modal] Epoch {epoch}/{epochs}  "
                f"Loss: {avg_loss:.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
        if epoch % save_every == 0 or epoch == epochs:
            torch.save(
                {"encoder": drug_encoder.state_dict(), "proj_head": drug_proj.state_dict(),
                 "epoch": epoch, "loss": avg_loss},
                os.path.join(save_dir, "drug_encoder.pt"),
            )
            torch.save(
                {"encoder": prot_encoder.state_dict(), "proj_head": prot_proj.state_dict(),
                 "epoch": epoch, "loss": avg_loss},
                os.path.join(save_dir, "prot_encoder.pt"),
            )

    return best_loss


# ──────────────────────────────────────────────
# Main pretraining orchestrator
# ──────────────────────────────────────────────

def run_pretraining(
    df: pd.DataFrame,
    mode: str = "both_independent",
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
    loss_name: str = "nt_xent",
    drug_aug_names: list | None = None,
    prot_aug_names: list | None = None,
    mask_ratio: float = 0.15,
    crop_min_ratio: float = 0.7,
    sub_ratio: float = 0.10,
    drop_prob: float = 0.1,
    projection_dim: int = 64,
    emb_dim: int = 128,
    conv_out: int = 128,
    max_sml_len: int = 120,
    max_prot_len: int = 1000,
    device: str = "cuda",
    save_dir: str = "checkpoints/pretrained/",
    tb_dir: str = "runs/pretrain/",
    seed: int = 42,
) -> dict:
    """
    High-level pretraining entry point.

    Returns dict with paths to saved encoder checkpoints.
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # Build vocabularies from full dataset
    sml_stoi, sml_itos = build_vocab(list(df["smiles"].unique()))
    prot_stoi, prot_itos = build_vocab(list(df["sequence"].unique()))

    drug_aug_names = drug_aug_names or ["smiles_enum", "atom_mask"]
    prot_aug_names = prot_aug_names or ["subseq_crop", "residue_mask"]

    loss_kwargs = {"temperature": temperature} if loss_name != "triplet" else {"margin": 1.0}
    loss_fn = get_contrastive_loss(loss_name, **loss_kwargs)

    tb_writer = SummaryWriter(log_dir=tb_dir)
    encoder_dim = conv_out * 3  # 3 kernels

    results = {}

    # ── Drug pretraining ──
    if mode in ("drug_only", "both_independent"):
        print("\n=== Drug Encoder Pretraining ===")
        unique_smiles = df["smiles"].unique().tolist()
        drug_ds = ContrastiveDrugDataset(
            unique_smiles, sml_stoi, max_sml_len,
            aug_names=drug_aug_names, mask_ratio=mask_ratio, drop_prob=drop_prob,
        )
        drug_loader = DataLoader(
            drug_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )

        drug_enc = DrugEncoder(len(sml_itos), emb_dim, conv_out)
        drug_proj = ProjectionHead(encoder_dim, 128, projection_dim)
        drug_opt = torch.optim.AdamW(
            list(drug_enc.parameters()) + list(drug_proj.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        drug_sched = torch.optim.lr_scheduler.CosineAnnealingLR(drug_opt, T_max=epochs)

        pretrain_encoder(
            drug_enc, drug_proj, drug_loader, loss_fn, drug_opt, drug_sched,
            device, epochs, tb_writer, tag="drug", save_dir=save_dir,
        )
        results["drug_ckpt"] = os.path.join(save_dir, "drug_encoder.pt")

    # ── Protein pretraining ──
    if mode in ("prot_only", "both_independent"):
        print("\n=== Protein Encoder Pretraining ===")
        unique_seqs = df["sequence"].unique().tolist()
        prot_ds = ContrastiveProteinDataset(
            unique_seqs, prot_stoi, max_prot_len,
            aug_names=prot_aug_names, mask_ratio=mask_ratio,
            crop_min_ratio=crop_min_ratio, sub_ratio=sub_ratio,
        )
        prot_loader = DataLoader(
            prot_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )

        prot_enc = ProteinEncoder(len(prot_itos), emb_dim, conv_out)
        prot_proj = ProjectionHead(encoder_dim, 128, projection_dim)
        prot_opt = torch.optim.AdamW(
            list(prot_enc.parameters()) + list(prot_proj.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        prot_sched = torch.optim.lr_scheduler.CosineAnnealingLR(prot_opt, T_max=epochs)

        pretrain_encoder(
            prot_enc, prot_proj, prot_loader, loss_fn, prot_opt, prot_sched,
            device, epochs, tb_writer, tag="prot", save_dir=save_dir,
        )
        results["prot_ckpt"] = os.path.join(save_dir, "prot_encoder.pt")

    # ── Cross-modal pretraining ──
    if mode == "cross_modal":
        print("\n=== Cross-Modal Pretraining ===")
        cm_ds = ContrastiveCrossModalDataset(
            df, sml_stoi, prot_stoi, max_sml_len, max_prot_len,
            drug_aug_names=drug_aug_names, prot_aug_names=prot_aug_names,
            mask_ratio=mask_ratio, drop_prob=drop_prob,
            crop_min_ratio=crop_min_ratio, sub_ratio=sub_ratio,
        )
        cm_loader = DataLoader(
            cm_ds, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )

        drug_enc = DrugEncoder(len(sml_itos), emb_dim, conv_out)
        prot_enc = ProteinEncoder(len(prot_itos), emb_dim, conv_out)
        drug_proj = ProjectionHead(encoder_dim, 128, projection_dim)
        prot_proj = ProjectionHead(encoder_dim, 128, projection_dim)

        all_params = (
            list(drug_enc.parameters()) + list(prot_enc.parameters())
            + list(drug_proj.parameters()) + list(prot_proj.parameters())
        )
        cm_opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
        cm_sched = torch.optim.lr_scheduler.CosineAnnealingLR(cm_opt, T_max=epochs)

        pretrain_cross_modal(
            drug_enc, prot_enc, drug_proj, prot_proj,
            cm_loader, loss_fn, cm_opt, cm_sched,
            device, epochs, tb_writer, save_dir=save_dir,
        )
        results["drug_ckpt"] = os.path.join(save_dir, "drug_encoder.pt")
        results["prot_ckpt"] = os.path.join(save_dir, "prot_encoder.pt")

    # Save vocab alongside checkpoints
    import json
    with open(os.path.join(save_dir, "sml_vocab.json"), "w") as f:
        json.dump({"stoi": sml_stoi, "itos": sml_itos}, f)
    with open(os.path.join(save_dir, "prot_vocab.json"), "w") as f:
        json.dump({"stoi": prot_stoi, "itos": prot_itos}, f)

    tb_writer.close()
    print(f"\nPretraining complete. Checkpoints saved to {save_dir}")
    return results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CL-DTA Contrastive Pretraining")
    parser.add_argument("--data", type=str, required=True, help="Processed CSV path")
    parser.add_argument("--mode", type=str, default="both_independent",
                        choices=["drug_only", "prot_only", "both_independent", "cross_modal"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss", type=str, default="nt_xent",
                        choices=["nt_xent", "infonce", "triplet"])
    parser.add_argument("--drug-augs", nargs="+", default=["smiles_enum", "atom_mask"])
    parser.add_argument("--prot-augs", nargs="+", default=["subseq_crop", "residue_mask"])
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--crop-min-ratio", type=float, default=0.7)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--conv-out", type=int, default=128)
    parser.add_argument("--max-sml-len", type=int, default=120)
    parser.add_argument("--max-prot-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="checkpoints/pretrained/")
    parser.add_argument("--tb-dir", type=str, default="runs/pretrain/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(args.data)

    run_pretraining(
        df=df,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        temperature=args.temperature,
        loss_name=args.loss,
        drug_aug_names=args.drug_augs,
        prot_aug_names=args.prot_augs,
        mask_ratio=args.mask_ratio,
        crop_min_ratio=args.crop_min_ratio,
        projection_dim=args.projection_dim,
        emb_dim=args.emb_dim,
        conv_out=args.conv_out,
        max_sml_len=args.max_sml_len,
        max_prot_len=args.max_prot_len,
        device=args.device,
        save_dir=args.out,
        tb_dir=args.tb_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
