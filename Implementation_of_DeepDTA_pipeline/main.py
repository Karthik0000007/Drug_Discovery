"""
main.py — CLI entry point for CL-DTA pipeline.

Supports:
  - Single experiment runs (train + eval)
  - Pretrained encoder loading (CL-DTA)
  - All 4 split types (random, cold_drug, cold_target, cold_both)
  - Config file (YAML) or command-line arguments
  - Full metric reporting (MSE, RMSE, CI, Pearson, Spearman, r²_m)
  - TensorBoard logging

Usage:
  # Train DeepDTA baseline
  python -m Implementation_of_DeepDTA_pipeline.main \
      --data data/davis_processed.csv --model DeepDTA --split cold_drug

  # Train CL-DTA with pretrained encoders
  python -m Implementation_of_DeepDTA_pipeline.main \
      --data data/davis_processed.csv --model CL-DTA \
      --pretrained-drug checkpoints/drug_encoder.pt \
      --pretrained-prot checkpoints/prot_encoder.pt \
      --freeze-strategy gradual_unfreeze

  # Load from YAML config
  python -m Implementation_of_DeepDTA_pipeline.main --config configs/davis_cold_drug.yaml
"""

from __future__ import annotations

import os
import math
import json
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .model import DeepDTAModel
from .data_loading import prepare_data
from .tokenizers_and_datasets import build_vocab, DtaDataset
from .train import train_loop, eval_model
from .utilities import set_seed, compute_all_metrics
from .gpu_config import configure_gpu, get_optimal_num_workers, get_optimal_batch_size


def build_model_from_args(args, vocab_drug: int, vocab_prot: int, device):
    """Instantiate model from CLI args."""
    model_name = args.model

    if model_name in ("DeepDTA", "CL-DTA"):
        model = DeepDTAModel(
            vocab_drug=vocab_drug,
            vocab_prot=vocab_prot,
            emb_dim=args.emb_dim,
            conv_out=args.conv_out,
            dropout=args.dropout,
        )
        if model_name == "CL-DTA":
            model.load_pretrained_encoders(
                args.pretrained_drug, args.pretrained_prot
            )
        return model.to(device)

    elif model_name == "AttentionDTA":
        from .model_attndta import AttentionDTAModel
        model = AttentionDTAModel(
            vocab_drug=vocab_drug,
            vocab_prot=vocab_prot,
            emb_dim=args.emb_dim,
            conv_out=args.conv_out,
            dropout=args.dropout,
        )
        return model.to(device)

    elif model_name == "WideDTA":
        from .model_widedta import WideDTAModel
        model = WideDTAModel(
            vocab_drug=vocab_drug,
            vocab_prot=vocab_prot,
            emb_dim=args.emb_dim,
            conv_out=args.conv_out,
            dropout=args.dropout,
        )
        return model.to(device)

    elif model_name == "GraphDTA":
        from .model_graphdta import GraphDTAModel
        model = GraphDTAModel(vocab_prot=vocab_prot)
        return model.to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="CL-DTA — Drug-Target Affinity Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──
    parser.add_argument("--data", type=str, required=True,
                        help="CSV path with columns: smiles, sequence, affinity (+ drug_id, target_id for cold splits)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file (overrides CLI args where set)")

    # ── Model ──
    parser.add_argument("--model", type=str, default="DeepDTA",
                        choices=["DeepDTA", "CL-DTA", "AttentionDTA", "WideDTA", "GraphDTA"])
    parser.add_argument("--pretrained-drug", type=str, default=None,
                        help="Path to pretrained drug encoder checkpoint")
    parser.add_argument("--pretrained-prot", type=str, default=None,
                        help="Path to pretrained protein encoder checkpoint")
    parser.add_argument("--freeze-strategy", type=str, default="full_finetune",
                        choices=["frozen", "full_finetune", "gradual_unfreeze"])
    parser.add_argument("--unfreeze-after", type=int, default=5,
                        help="Epochs before unfreezing (gradual_unfreeze only)")

    # ── Training ──
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    # ── Architecture ──
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--conv-out", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-sml-len", type=int, default=100)
    parser.add_argument("--max-prot-len", type=int, default=1000)

    # ── Split ──
    parser.add_argument("--split", type=str, default="random",
                        choices=["random", "cold_drug", "cold_target", "cold_both"])
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # ── Output ──
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # ── Optional: IC50 conversion ──
    parser.add_argument("--ic50-nanomolar", action="store_true",
                        help="Convert IC50 (nM) to pIC50 before training")

    args = parser.parse_args()

    # ── Load YAML config if provided ──
    if args.config is not None:
        from .config import load_config
        cfg = load_config(args.config)
        # Override CLI defaults with config values
        if cfg.data.dataset:
            args.data = args.data  # keep CLI data path
        if cfg.train.epochs:
            args.epochs = cfg.train.epochs
        if cfg.train.batch_size:
            args.batch = cfg.train.batch_size
        if cfg.train.lr:
            args.lr = cfg.train.lr
        if cfg.data.split:
            args.split = cfg.data.split
        if cfg.train.freeze_strategy:
            args.freeze_strategy = cfg.train.freeze_strategy
        print(f"[main] Loaded config from {args.config}")

    # ── Setup ──
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # Configure GPU for 80% utilization
    if args.device == "cuda" and torch.cuda.is_available():
        device = configure_gpu(memory_fraction=0.80)
    else:
        device = torch.device(args.device)

    tag = f"{args.model}_{args.split}_seed{args.seed}"
    print(f"\n{'='*60}")
    print(f"[main] CL-DTA Pipeline — {tag}")
    print(f"{'='*60}")

    # ── Load & clean data ──
    print(f"[main] Loading data from {args.data} ...")
    df = pd.read_csv(args.data)
    for col in ("smiles", "sequence", "affinity"):
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'")

    if args.ic50_nanomolar:
        print("[main] Converting IC50 (nM) to pIC50 ...")
        def to_pic50(v):
            try:
                v = float(v)
                return 9.0 - math.log10(v) if v > 0 else np.nan
            except Exception:
                return np.nan
        df["affinity"] = df["affinity"].apply(to_pic50)

    df = df.dropna(subset=["smiles", "sequence", "affinity"]).reset_index(drop=True)
    print(f"[main] Total examples: {len(df)}")

    # ── Split ──
    train_df, val_df, test_df = prepare_data(
        df, split=args.split,
        test_frac=args.test_frac, val_frac=args.val_frac,
        seed=args.seed,
    )
    print(f"[main] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Vocabularies (from training data only) ──
    sml_stoi, sml_itos = build_vocab(train_df["smiles"].tolist())
    prot_stoi, prot_itos = build_vocab(train_df["sequence"].tolist())
    print(f"[main] Vocab: drug={len(sml_stoi)}, protein={len(prot_stoi)}")

    # ── Datasets & loaders (GPU-optimized) ──
    train_ds = DtaDataset(train_df, sml_stoi, prot_stoi, args.max_sml_len, args.max_prot_len)
    val_ds = DtaDataset(val_df, sml_stoi, prot_stoi, args.max_sml_len, args.max_prot_len)
    test_ds = DtaDataset(test_df, sml_stoi, prot_stoi, args.max_sml_len, args.max_prot_len)

    # Optimized DataLoader settings for maximum GPU throughput
    num_workers = get_optimal_num_workers()
    use_persistent = num_workers > 0
    loader_kwargs = dict(
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    print(f"[main] DataLoader: workers={num_workers}, pin_memory={device.type == 'cuda'}, prefetch=4")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, **loader_kwargs)

    # ── Model ──
    model = build_model_from_args(args, len(sml_stoi), len(prot_stoi), device)
    if hasattr(model, "parameter_count"):
        print(f"[main] Parameters: {model.parameter_count()}")

    # ── Optimizer + scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is not None:
        tb_dir = os.path.join(args.out, "tensorboard", tag)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)

    # ── Train ──
    t0 = time.time()
    result = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        freeze_strategy=args.freeze_strategy,
        unfreeze_after=args.unfreeze_after,
        tb_writer=writer,
        experiment_tag=tag,
    )
    elapsed = time.time() - t0

    # ── Load best model & final test eval ──
    if result["best_model_state"] is not None:
        model.load_state_dict(result["best_model_state"])
    model.to(device)

    y_true_test, y_pred_test = eval_model(model, test_loader, device)
    test_metrics = compute_all_metrics(y_true_test, y_pred_test)

    # Log test metrics to TensorBoard
    if writer is not None:
        for k, v in test_metrics.items():
            writer.add_scalar(f"{tag}/test/{k}", v, 0)
        writer.close()

    # ── Print results ──
    print(f"\n{'='*60}")
    print(f"[main] FINAL TEST RESULTS — {tag}")
    print(f"{'='*60}")
    for k, v in test_metrics.items():
        print(f"  {k:>15s}: {v:.4f}")
    print(f"  {'train_time':>15s}: {elapsed:.1f}s")
    print(f"  {'epochs_trained':>15s}: {result['epochs_trained']}")

    # ── Save results JSON ──
    experiment_result = {
        "model": args.model,
        "dataset": os.path.basename(args.data).replace("_processed.csv", ""),
        "split": args.split,
        "seed": args.seed,
        "metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_losses": [float(x) for x in result["train_losses"]],
        "val_rmses": [float(x) for x in result["val_rmses"]],
        "val_cis": [float(x) for x in result["val_cis"]],
        "epochs_trained": result["epochs_trained"],
        "best_val_rmse": float(result["best_val_rmse"]),
        "train_time_sec": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    json_dir = os.path.join(args.out, "json")
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f"{tag}.json")
    with open(json_path, "w") as f:
        json.dump(experiment_result, f, indent=2)
    print(f"[main] Results saved to {json_path}")

    # ── Save best model checkpoint ──
    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}_best.pt")
    torch.save({
        "model_state": result["best_model_state"],
        "args": vars(args),
        "test_metrics": test_metrics,
    }, ckpt_path)
    print(f"[main] Best model saved to {ckpt_path}")

    # ── Save test predictions ──
    preds_df = pd.DataFrame({"true": y_true_test, "pred": y_pred_test})
    preds_path = os.path.join(args.out, f"{tag}_test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"[main] Predictions saved to {preds_path}")


if __name__ == "__main__":
    main()