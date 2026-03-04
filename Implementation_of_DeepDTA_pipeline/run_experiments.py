"""
run_experiments.py — Batch experiment orchestrator for CL-DTA.

Runs the full experiment matrix:
  2 datasets (DAVIS, KIBA) × 4 splits × 5 models × 3 seeds = 120 experiments.

Usage:
  python -m Implementation_of_DeepDTA_pipeline.run_experiments \
      --datasets davis kiba \
      --splits random cold_drug cold_target cold_both \
      --models DeepDTA AttentionDTA CL-DTA \
      --seeds 42 123 456 \
      --output-dir results
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from .config import ExperimentConfig, DataConfig, TrainConfig, PretrainConfig
from .data_loading import prepare_data
from .tokenizers_and_datasets import build_vocab, tokenize_seq, DtaDataset
from .utilities import set_seed, compute_all_metrics
from .train import train_loop, eval_model


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def build_model(
    model_name: str,
    vocab_drug: int,
    vocab_prot: int,
    device: torch.device,
    pretrained_drug: Optional[str] = None,
    pretrained_prot: Optional[str] = None,
    **kwargs,
) -> torch.nn.Module:
    """Instantiate a model by name."""

    if model_name in ("DeepDTA", "CL-DTA"):
        from .model import DeepDTAModel
        model = DeepDTAModel(vocab_drug, vocab_prot)
        if model_name == "CL-DTA":
            model.load_pretrained_encoders(pretrained_drug, pretrained_prot)
        return model.to(device)

    elif model_name == "AttentionDTA":
        from .model_attndta import AttentionDTAModel
        model = AttentionDTAModel(vocab_drug, vocab_prot)
        return model.to(device)

    elif model_name == "WideDTA":
        from .model_widedta import WideDTAModel
        # WideDTA uses same char-level tokenization for simplicity in this pipeline
        model = WideDTAModel(vocab_drug, vocab_prot)
        return model.to(device)

    elif model_name == "GraphDTA":
        try:
            from .model_graphdta import GraphDTAModel
            model = GraphDTAModel(vocab_prot)
            return model.to(device)
        except ImportError:
            print(f"[run] GraphDTA requires torch_geometric — skipping.")
            return None

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ──────────────────────────────────────────────
# Single experiment runner
# ──────────────────────────────────────────────

def run_single_experiment(
    dataset_name: str,
    split_type: str,
    model_name: str,
    seed: int,
    data_dir: str = "data",
    output_dir: str = "results",
    pretrained_drug: Optional[str] = None,
    pretrained_prot: Optional[str] = None,
    freeze_strategy: str = "full_finetune",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-4,
    patience: int = 8,
) -> Optional[Dict]:
    """
    Run one experiment: load data → split → train → evaluate → save.

    Returns dict with metrics and training history.
    """
    tag = f"{dataset_name}_{split_type}_{model_name}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"[run] Starting: {tag}")
    print(f"{'='*60}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ──
    csv_path = os.path.join(data_dir, f"{dataset_name}_processed.csv")
    if not os.path.exists(csv_path):
        print(f"[run] Dataset not found: {csv_path} — skipping.")
        return None

    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = prepare_data(df, split=split_type, seed=seed)
    print(f"[run] Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Build vocabularies (from training data only) ──
    drug_stoi, _ = build_vocab(train_df["smiles"].tolist())
    prot_stoi, _ = build_vocab(train_df["sequence"].tolist())

    max_drug = 100
    max_prot = 1000

    # ── Create datasets ──
    train_ds = DtaDataset(train_df, drug_stoi, prot_stoi, max_drug, max_prot)
    val_ds = DtaDataset(val_df, drug_stoi, prot_stoi, max_drug, max_prot)
    test_ds = DtaDataset(test_df, drug_stoi, prot_stoi, max_drug, max_prot)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Build model ──
    model = build_model(
        model_name, len(drug_stoi), len(prot_stoi), device,
        pretrained_drug=pretrained_drug,
        pretrained_prot=pretrained_prot,
    )
    if model is None:
        return None

    if hasattr(model, "parameter_count"):
        print(f"[run] Parameters: {model.parameter_count()}")

    # ── Optimizer + Scheduler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5,
    )

    # ── TensorBoard ──
    writer = None
    if SummaryWriter is not None:
        tb_dir = os.path.join(output_dir, "tensorboard", tag)
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
        epochs=epochs,
        patience=patience,
        freeze_strategy=freeze_strategy if model_name == "CL-DTA" else "full_finetune",
        tb_writer=writer,
        experiment_tag=tag,
    )
    train_time = time.time() - t0

    # ── Test evaluation ──
    if result["best_model_state"] is not None:
        model.load_state_dict(result["best_model_state"])
    model.to(device)

    y_true, y_pred = eval_model(model, test_loader, device)
    test_metrics = compute_all_metrics(y_true, y_pred)

    # Log test metrics
    if writer is not None:
        for k, v in test_metrics.items():
            writer.add_scalar(f"{tag}/test/{k}", v, 0)
        writer.close()

    print(f"[run] Test metrics: {test_metrics}")
    print(f"[run] Training time: {train_time:.1f}s")

    # ── Save results ──
    experiment_result = {
        "model": model_name,
        "dataset": dataset_name,
        "split": split_type,
        "seed": seed,
        "metrics": {k: float(v) for k, v in test_metrics.items()},
        "train_losses": [float(x) for x in result["train_losses"]],
        "val_rmses": [float(x) for x in result["val_rmses"]],
        "val_cis": [float(x) for x in result["val_cis"]],
        "epochs_trained": result["epochs_trained"],
        "best_val_rmse": float(result["best_val_rmse"]),
        "train_time_sec": round(train_time, 1),
        "timestamp": datetime.now().isoformat(),
    }

    result_dir = os.path.join(output_dir, "json")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{tag}.json")
    with open(result_path, "w") as f:
        json.dump(experiment_result, f, indent=2)
    print(f"[run] Saved results to {result_path}")

    return experiment_result


# ──────────────────────────────────────────────
# Batch orchestrator
# ──────────────────────────────────────────────

def run_all_experiments(
    datasets: List[str] = ("davis", "kiba"),
    splits: List[str] = ("random", "cold_drug", "cold_target", "cold_both"),
    models: List[str] = ("DeepDTA", "AttentionDTA", "CL-DTA"),
    seeds: List[int] = (42, 123, 456),
    data_dir: str = "data",
    output_dir: str = "results",
    pretrained_drug: Optional[str] = None,
    pretrained_prot: Optional[str] = None,
    freeze_strategy: str = "full_finetune",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-4,
) -> List[Dict]:
    """Run the full experiment matrix and aggregate results."""

    total = len(datasets) * len(splits) * len(models) * len(seeds)
    print(f"[run] Total experiments: {total}")
    print(f"[run] Datasets: {datasets}")
    print(f"[run] Splits: {splits}")
    print(f"[run] Models: {models}")
    print(f"[run] Seeds: {seeds}")

    all_results = []
    completed = 0
    failed = 0

    for ds in datasets:
        for split in splits:
            for model_name in models:
                for seed in seeds:
                    try:
                        result = run_single_experiment(
                            dataset_name=ds,
                            split_type=split,
                            model_name=model_name,
                            seed=seed,
                            data_dir=data_dir,
                            output_dir=output_dir,
                            pretrained_drug=pretrained_drug,
                            pretrained_prot=pretrained_prot,
                            freeze_strategy=freeze_strategy,
                            epochs=epochs,
                            batch_size=batch_size,
                            lr=lr,
                        )
                        if result is not None:
                            all_results.append(result)
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        print(f"[run] FAILED: {ds}_{split}_{model_name}_seed{seed}")
                        traceback.print_exc()
                        failed += 1

                    print(f"[run] Progress: {completed + failed}/{total} "
                          f"(completed={completed}, failed={failed})")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"[run] ALL EXPERIMENTS COMPLETE")
    print(f"[run] Completed: {completed}, Failed: {failed}")
    print(f"{'='*60}")

    if all_results:
        _print_summary_table(all_results)

    return all_results


def _print_summary_table(results: List[Dict]):
    """Print aggregated mean±std results table."""
    from collections import defaultdict

    agg = defaultdict(list)
    for r in results:
        key = (r["dataset"], r["split"], r["model"])
        agg[key].append(r["metrics"])

    print(f"\n{'Dataset':<10} {'Split':<15} {'Model':<15} {'CI':>12} {'RMSE':>12} {'Pearson':>12}")
    print("-" * 80)
    for (ds, split, model), metric_list in sorted(agg.items()):
        cis = [m.get("ci", 0) for m in metric_list]
        rmses = [m.get("rmse", 0) for m in metric_list]
        rs = [m.get("pearson_r", 0) for m in metric_list]
        ci_str = f"{np.mean(cis):.3f}±{np.std(cis):.3f}"
        rmse_str = f"{np.mean(rmses):.3f}±{np.std(rmses):.3f}"
        r_str = f"{np.mean(rs):.3f}±{np.std(rs):.3f}"
        print(f"{ds:<10} {split:<15} {model:<15} {ci_str:>12} {rmse_str:>12} {r_str:>12}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CL-DTA Experiment Runner")
    parser.add_argument("--datasets", nargs="+", default=["davis", "kiba"])
    parser.add_argument("--splits", nargs="+",
                        default=["random", "cold_drug", "cold_target", "cold_both"])
    parser.add_argument("--models", nargs="+",
                        default=["DeepDTA", "AttentionDTA", "CL-DTA"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--pretrained-drug", default=None)
    parser.add_argument("--pretrained-prot", default=None)
    parser.add_argument("--freeze-strategy", default="full_finetune",
                        choices=["frozen", "full_finetune", "gradual_unfreeze"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    run_all_experiments(
        datasets=args.datasets,
        splits=args.splits,
        models=args.models,
        seeds=args.seeds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_drug=args.pretrained_drug,
        pretrained_prot=args.pretrained_prot,
        freeze_strategy=args.freeze_strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
