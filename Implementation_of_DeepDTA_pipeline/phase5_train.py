"""
Phase 5: MAML Meta-Learning Training Script

This script integrates with Phase 4 pretrained encoders and trains a MAML
meta-learner on few-shot DTA tasks for rapid cold-start adaptation.

Usage:
    python -m Implementation_of_DeepDTA_pipeline.phase5_train \
        --data data/davis_processed.csv \
        --model-path checkpoints/pretrained_phase4/ \
        --meta-epochs 10 \
        --meta-batch-size 4 \
        --num-inner-steps 3
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from .meta_dataset import MetaDTADataset
from .meta_train import MetaTrainer
from .meta_eval import (
    evaluate_few_shot_performance,
    evaluate_by_task_type,
    ablation_adaptation_scope,
    print_few_shot_results,
)
from .config import MetaLearningConfig, ExperimentConfig
from .model import DeepDTAModel
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from .gpu_config import configure_gpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_or_create_model(df, model_path=None, device="cuda"):
    import json
    from .tokenizers_and_datasets import build_vocab

    # Try to load vocab from checkpoint, otherwise compute from data
    if model_path:
        vocab_path_sml = Path(model_path) / "sml_vocab.json"
        vocab_path_prot = Path(model_path) / "prot_vocab.json"
        if vocab_path_sml.exists() and vocab_path_prot.exists():
            with open(vocab_path_sml) as f:
                sml_vocab = json.load(f)
                vocab_drug = len(sml_vocab.get("stoi", {}))
            with open(vocab_path_prot) as f:
                prot_vocab = json.load(f)
                vocab_prot = len(prot_vocab.get("stoi", {}))
        else:
            smiles_list = df['smiles'].unique().tolist()
            seq_list = df['sequence'].unique().tolist()
            sml_stoi, _ = build_vocab(smiles_list)
            prot_stoi, _ = build_vocab(seq_list)
            vocab_drug = len(sml_stoi) + 2
            vocab_prot = len(prot_stoi) + 2
    else:
        smiles_list = df['smiles'].unique().tolist()
        seq_list = df['sequence'].unique().tolist()
        sml_stoi, _ = build_vocab(smiles_list)
        prot_stoi, _ = build_vocab(seq_list)
        vocab_drug = len(sml_stoi) + 2
        vocab_prot = len(prot_stoi) + 2

    model = DeepDTAModel(
        vocab_drug=vocab_drug,
        vocab_prot=vocab_prot,
        emb_dim=128,
        conv_out=128,
        sml_kernels=(4, 6, 8),
        prot_kernels=(4, 8, 12),
        dropout=0.2,
        use_pretrained_embeddings=False,
    )
    model = model.to(device)
    if model_path:
        logger.info(f"Loading pretrained encoders from {model_path}")
        drug_ckpt = Path(model_path) / "drug_encoder.pt"
        prot_ckpt = Path(model_path) / "prot_encoder.pt"
        if drug_ckpt.exists() and prot_ckpt.exists():
            model.load_pretrained_encoders(str(drug_ckpt), str(prot_ckpt))
        else:
            logger.warning(f"Pretrained checkpoints not found at {model_path}. Using random initialization.")
    return model


def main(args):
    """Main meta-training pipeline."""
    # Configure GPU for 80% utilization
    if torch.cuda.is_available():
        device = configure_gpu(memory_fraction=0.80)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load data
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} samples")

    # ─────────────────────────────────────────────────────────────────────────
    # Create meta-dataset
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Creating meta-dataset with split_type={args.split_type}")
    meta_dataset = MetaDTADataset(
        df,
        drug_col="drug_id",
        target_col="target_id",
        affinity_col="affinity",
        split_type=args.split_type,
        seed=args.seed,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Create meta-learning config
    # ─────────────────────────────────────────────────────────────────────────
    meta_config = MetaLearningConfig(
        enabled=True,
        num_inner_steps=args.num_inner_steps,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        meta_batch_size=args.meta_batch_size,
        adaptation_scope=args.adaptation_scope,
        task_type=args.split_type,
        k_support=args.k_support,
        k_query=args.k_query,
        meta_epochs=args.meta_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    logger.info(f"Meta-Learning Config: {meta_config}")

    # ─────────────────────────────────────────────────────────────────────────
    # Load or create model
    # ─────────────────────────────────────────────────────────────────────────
    model = load_or_create_model(df, args.model_path, device=device)
    logger.info("Model initialized")

    # ─────────────────────────────────────────────────────────────────────────
    # Initialize trainer
    # ─────────────────────────────────────────────────────────────────────────
    trainer = MetaTrainer(model, meta_config, meta_lr=meta_config.meta_lr, device=device)
    logger.info("Trainer initialized")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup TensorBoard logging
    # ─────────────────────────────────────────────────────────────────────────
    writer = SummaryWriter(f"runs/meta_learning/phase5_{args.seed}/") if SummaryWriter is not None else None
    if writer is None:
        logger.warning("TensorBoard not installed. Install via: pip install tensorboard")

    # ─────────────────────────────────────────────────────────────────────────
    # Meta-training loop
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Starting meta-training for {meta_config.meta_epochs} epochs...")

    for epoch in range(meta_config.meta_epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{meta_config.meta_epochs} ---")

        # Training step
        epoch_metrics = trainer.train_epoch(
            meta_dataset, num_tasks_per_epoch=args.num_tasks_per_epoch
        )

        logger.info(f"Meta-Loss: {epoch_metrics['meta_loss']:.4f}")
        logger.info(f"Query-Loss: {epoch_metrics['query_loss']:.4f}")
        logger.info(f"Adaptation Improvement: {epoch_metrics['adaptation_improvement']:.4f}")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("meta_train/meta_loss", epoch_metrics["meta_loss"], epoch)
            writer.add_scalar("meta_train/support_loss", epoch_metrics["support_loss"], epoch)
            writer.add_scalar("meta_train/query_loss", epoch_metrics["query_loss"], epoch)
            writer.add_scalar(
                "meta_train/adaptation_improvement",
                epoch_metrics["adaptation_improvement"],
                epoch,
            )

        # Periodic evaluation
        if (epoch + 1) % args.eval_interval == 0:
            logger.info(f"\n[Epoch {epoch+1}] Running evaluation...")
            eval_metrics = trainer.evaluate_epoch(
                meta_dataset, num_tasks=args.num_eval_tasks
            )
            logger.info(f"Eval Loss (before): {eval_metrics['loss_before']:.4f}")
            logger.info(f"Eval Loss (after): {eval_metrics['loss_after']:.4f}")
            logger.info(f"Eval Improvement: {eval_metrics['improvement_pct']:.2f}%")

            if writer is not None:
                writer.add_scalar("meta_eval/loss_before", eval_metrics["loss_before"], epoch)
                writer.add_scalar("meta_eval/loss_after", eval_metrics["loss_after"], epoch)
                writer.add_scalar(
                    "meta_eval/improvement_pct", eval_metrics["improvement_pct"], epoch
                )

    if writer is not None:
        writer.flush()
        writer.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Save checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "meta_learned_model.pt"

    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Model saved to {ckpt_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Few-shot evaluation (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if args.run_few_shot_eval:
        logger.info("\n\n=== FINAL FEW-SHOT EVALUATION ===")
        results = evaluate_few_shot_performance(
            model,
            meta_dataset,
            meta_config,
            k_shots=args.k_shots,
            k_query=args.k_query,
            num_tasks_per_k=args.num_few_shot_tasks,
            device=device,
        )
        print_few_shot_results(results)

    logger.info("\n✓ Meta-training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: MAML Meta-Learning for Few-Shot DTA"
    )

    # Data
    parser.add_argument(
        "--data", type=str, default="data/davis_processed.csv", help="Path to DTA data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Phase 4 pretrained model checkpoint",
    )

    # Meta-learning config
    parser.add_argument(
        "--meta-epochs", type=int, default=20, help="Number of meta-training epochs"
    )
    parser.add_argument(
        "--meta-batch-size", type=int, default=4, help="Tasks per meta-batch"
    )
    parser.add_argument(
        "--num-inner-steps", type=int, default=3, help="Gradient steps on support set"
    )
    parser.add_argument(
        "--inner-lr", type=float, default=1e-3, help="Inner loop learning rate"
    )
    parser.add_argument(
        "--meta-lr", type=float, default=1e-4, help="Meta-model learning rate"
    )

    # Adaptation scope
    parser.add_argument(
        "--adaptation-scope",
        type=str,
        default="head_only",
        choices=["head_only", "partial_encoder", "full"],
        help="Which parameters to adapt",
    )

    # Task sampling
    parser.add_argument(
        "--split-type",
        type=str,
        default="mixed",
        choices=["cold_drug", "cold_target", "cold_both", "mixed"],
        help="Task type to sample",
    )
    parser.add_argument(
        "--k-support", type=int, default=5, help="Support set size (few-shot)"
    )
    parser.add_argument("--k-query", type=int, default=10, help="Query set size")

    # Training
    parser.add_argument(
        "--num-tasks-per-epoch", type=int, default=100, help="Tasks per training epoch"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=5, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--num-eval-tasks", type=int, default=50, help="Tasks for evaluation"
    )

    # Few-shot evaluation
    parser.add_argument(
        "--run-few-shot-eval",
        action="store_true",
        help="Run few-shot evaluation at end",
    )
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="k-shot scenarios to evaluate",
    )
    parser.add_argument(
        "--num-few-shot-tasks",
        type=int,
        default=50,
        help="Tasks per k-shot scenario",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/meta_learning/",
        help="Checkpoint directory",
    )

    args = parser.parse_args()
    main(args)
