"""
Phase 5: Few-Shot Evaluation Script

Evaluates a trained meta-learner on few-shot scenarios.

Usage:
    python -m Implementation_of_DeepDTA_pipeline.phase5_eval \
        --data data/davis_processed.csv \
        --model-path checkpoints/meta_learning/meta_learned_model.pt \
        --k-shots 1 5 10
"""

from __future__ import annotations

import argparse
import torch
import pandas as pd
from pathlib import Path
import logging

from .meta_dataset import MetaDTADataset
from .meta_eval import (
    evaluate_few_shot_performance,
    evaluate_by_task_type,
    ablation_adaptation_scope,
    print_few_shot_results,
)
from .config import MetaLearningConfig
from .model import DeepDTAModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, df: pd.DataFrame = None, device: str = "cuda") -> torch.nn.Module:
    """Load meta-learned model."""
    import json
    from pathlib import Path
    from .tokenizers_and_datasets import build_vocab

    # Try to load vocab from pretrained_test checkpoint (where encoders came from)
    pretrained_dir = Path("checkpoints/pretrained_test")
    vocab_path_sml = pretrained_dir / "sml_vocab.json"
    vocab_path_prot = pretrained_dir / "prot_vocab.json"

    if vocab_path_sml.exists() and vocab_path_prot.exists():
        with open(vocab_path_sml) as f:
            vocab_drug = len(json.load(f).get("stoi", {}))
        with open(vocab_path_prot) as f:
            vocab_prot = len(json.load(f).get("stoi", {}))
        logger.info(f"Loaded vocab from pretrained: drug={vocab_drug}, prot={vocab_prot}")
    elif df is not None:
        smiles_list = df['smiles'].unique().tolist()
        seq_list = df['sequence'].unique().tolist()
        sml_stoi, _ = build_vocab(smiles_list)
        prot_stoi, _ = build_vocab(seq_list)
        vocab_drug = len(sml_stoi) + 2
        vocab_prot = len(prot_stoi) + 2
        logger.info(f"Computed vocab from data: drug={vocab_drug}, prot={vocab_prot}")
    else:
        vocab_drug = 256
        vocab_prot = 256
        logger.warning("Using default vocab sizes")

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

    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)  # Use strict=False to allow size mismatches
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model


def main(args):
    """Run few-shot evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)

    # Create meta-dataset
    logger.info(f"Creating meta-dataset with split_type={args.split_type}")
    meta_dataset = MetaDTADataset(
        df,
        drug_col="drug_id",
        target_col="target_id",
        affinity_col="affinity",
        split_type=args.split_type,
        seed=args.seed,
    )

    # Meta-learning config
    meta_config = MetaLearningConfig(
        enabled=True,
        num_inner_steps=args.num_inner_steps,
        inner_lr=args.inner_lr,
        adaptation_scope=args.adaptation_scope,
        k_support=max(args.k_shots),  # Use largest k for config
        k_query=args.k_query,
    )

    # Load model
    model = load_model(args.model_path, df=df, device=device)

    # ─────────────────────────────────────────────────────────────────────────
    # Few-shot evaluation
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n=== FEW-SHOT EVALUATION ===")
    results = evaluate_few_shot_performance(
        model,
        meta_dataset,
        meta_config,
        k_shots=args.k_shots,
        k_query=args.k_query,
        num_tasks_per_k=args.num_tasks_per_k,
        device=device,
    )
    print_few_shot_results(results)

    # ─────────────────────────────────────────────────────────────────────────
    # Task-type specific evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if args.eval_by_task_type:
        logger.info("\n=== TASK-TYPE SPECIFIC EVALUATION ===")
        results_by_type = evaluate_by_task_type(
            model,
            meta_dataset,
            meta_config,
            task_types=["cold_drug", "cold_target", "cold_both"],
            k_support=args.k_shots[len(args.k_shots) // 2],  # Use middle k
            k_query=args.k_query,
            num_tasks_per_type=args.num_tasks_per_k,
            device=device,
        )

        print("\n" + "="*80)
        print("TASK-TYPE BREAKDOWN")
        print("="*80)
        for task_type, metrics in results_by_type.items():
            print(f"\n{task_type}:")
            print(
                f"  Loss (before): {metrics['loss_before']['mean']:.4f} ± "
                f"{metrics['loss_before']['std']:.4f}"
            )
            print(
                f"  Loss (after):  {metrics['loss_after']['mean']:.4f} ± "
                f"{metrics['loss_after']['std']:.4f}"
            )
            print(
                f"  Improvement:   {metrics['improvement_pct']['mean']:.2f}% ± "
                f"{metrics['improvement_pct']['std']:.2f}%"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Ablation study
    # ─────────────────────────────────────────────────────────────────────────
    if args.run_ablation:
        logger.info("\n=== ABLATION: ADAPTATION SCOPE ===")
        results_ablation = ablation_adaptation_scope(
            model,
            meta_dataset,
            meta_config,
            scopes=["head_only", "partial_encoder", "full"],
            k_support=args.k_shots[len(args.k_shots) // 2],
            k_query=args.k_query,
            num_tasks=args.num_tasks_per_k,
            device=device,
        )

        print("\n" + "="*80)
        print("ADAPTATION SCOPE ABLATION")
        print("="*80)
        for scope, metrics in results_ablation.items():
            print(
                f"{scope}: {metrics['improvement_pct']['mean']:.2f}% ± "
                f"{metrics['improvement_pct']['std']:.2f}%"
            )

    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 5: Few-Shot Evaluation for Meta-Learned DTA"
    )

    parser.add_argument(
        "--data", type=str, default="data/davis_processed.csv", help="Path to DTA data"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained meta-learned model",
    )

    # Evaluation config
    parser.add_argument(
        "--k-shots",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="k-shot scenarios to evaluate",
    )
    parser.add_argument("--k-query", type=int, default=10, help="Query set size")
    parser.add_argument(
        "--num-tasks-per-k",
        type=int,
        default=50,
        help="Number of tasks per k-shot scenario",
    )

    # Meta-learning config
    parser.add_argument(
        "--num-inner-steps", type=int, default=3, help="Gradient steps on support set"
    )
    parser.add_argument(
        "--inner-lr", type=float, default=1e-3, help="Inner loop learning rate"
    )
    parser.add_argument(
        "--adaptation-scope",
        type=str,
        default="head_only",
        choices=["head_only", "partial_encoder", "full"],
    )

    # Dataset config
    parser.add_argument(
        "--split-type",
        type=str,
        default="mixed",
        choices=["cold_drug", "cold_target", "cold_both", "mixed"],
    )

    # Optional evaluations
    parser.add_argument(
        "--eval-by-task-type",
        action="store_true",
        help="Evaluate per task type",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation study on adaptation scopes",
    )

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
