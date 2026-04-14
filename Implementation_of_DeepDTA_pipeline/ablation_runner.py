"""
ablation_runner.py — Phase 11: Systematic Ablation & Experiment Orchestration.

Implements:
  - AblationRunner: configurable ablation matrix execution
  - Grid search / combinatorial runs with config overrides
  - Result aggregation and leaderboard generation
  - LaTeX / Markdown table export
  - Automated artifact generation hooks

Key ablations:
  1. Cross-modal alignment weight (0.0 vs 0.1-1.0)
  2. Pretraining scale (DAVIS/KIBA vs BindingDB)
  3. LLM initialization (none vs ESM/ChemBERTa)
  4. Meta-learning (standard vs MAML)
  5. Attention module (off vs pocket-guided)
  6. Uncertainty head (deterministic vs evidential)
  7. Multi-task (affinity-only vs full)
"""

from __future__ import annotations

import copy
import json
import os
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Ablation Runner
# ─────────────────────────────────────────────────────────────────────────────


class AblationRunner:
    """
    Flexible ablation experiment orchestrator.

    Runs experiment matrices by systematically varying configuration
    parameters while keeping others fixed.

    Usage:
        runner = AblationRunner(base_config)
        runner.add_ablation("cross_modal_weight", {"pretrain.align_loss_weight": [0.0, 0.1, 0.5, 1.0]})
        runner.add_ablation("llm_init", {"pretrain.use_pretrained_embeddings": [False, True]})
        results = runner.run_all(run_fn=run_single_experiment)

    Parameters
    ----------
    base_config : dict or ExperimentConfig
        Base configuration to ablate from.
    output_dir : str
        Directory for ablation results.
    seeds : list of int
        Seeds for repeated runs.
    """

    def __init__(
        self,
        base_config: Any,
        output_dir: str = "results/ablations",
        seeds: List[int] = None,
    ):
        if hasattr(base_config, "__dataclass_fields__"):
            from .config import config_to_dict
            self.base_config = config_to_dict(base_config)
        elif isinstance(base_config, dict):
            self.base_config = copy.deepcopy(base_config)
        else:
            raise TypeError("base_config must be a dict or dataclass")

        self.output_dir = output_dir
        self.seeds = seeds or [42, 123, 456]
        self.ablations: Dict[str, Dict[str, List]] = {}
        self.results: List[Dict] = []

        os.makedirs(output_dir, exist_ok=True)

    def add_ablation(
        self,
        name: str,
        variants: Dict[str, List],
    ) -> None:
        """
        Register an ablation study.

        Parameters
        ----------
        name : str
            Name of the ablation (e.g. "cross_modal_weight")
        variants : dict
            Mapping of dotted config paths to lists of values to try.
            e.g. {"pretrain.align_loss_weight": [0.0, 0.1, 0.5, 1.0]}
        """
        self.ablations[name] = variants

    def _set_nested(self, config: dict, dotted_key: str, value: Any) -> dict:
        """Set a value in a nested dict using dotted path notation."""
        keys = dotted_key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        return config

    def _get_nested(self, config: dict, dotted_key: str) -> Any:
        """Get a value from a nested dict using dotted path notation."""
        keys = dotted_key.split(".")
        d = config
        for k in keys:
            d = d.get(k, {})
        return d

    def generate_ablation_configs(
        self,
        name: str,
    ) -> List[Tuple[str, dict]]:
        """Generate all config variants for a named ablation."""
        variants = self.ablations[name]
        keys = list(variants.keys())
        value_lists = [variants[k] for k in keys]

        configs = []
        for combo in product(*value_lists):
            cfg = copy.deepcopy(self.base_config)
            label_parts = []
            for key, val in zip(keys, combo):
                self._set_nested(cfg, key, val)
                short_key = key.split(".")[-1]
                label_parts.append(f"{short_key}={val}")
            label = f"{name}_{'_'.join(label_parts)}"
            configs.append((label, cfg))

        return configs

    def run_ablation(
        self,
        name: str,
        run_fn,
        **run_kwargs,
    ) -> List[Dict]:
        """
        Execute a single ablation study.

        Parameters
        ----------
        name : str
            Ablation name (must be previously added via add_ablation)
        run_fn : callable
            Function that takes a config dict and returns a result dict.
            Signature: run_fn(config, seed, **kwargs) -> dict
        **run_kwargs : dict
            Additional keyword arguments passed to run_fn

        Returns
        -------
        list of result dicts
        """
        if name not in self.ablations:
            raise ValueError(f"Ablation '{name}' not registered. Use add_ablation() first.")

        configs = self.generate_ablation_configs(name)
        results = []

        print(f"\n{'='*70}")
        print(f"[Ablation] Running: {name}")
        print(f"[Ablation] Variants: {len(configs)}, Seeds: {len(self.seeds)}")
        print(f"[Ablation] Total runs: {len(configs) * len(self.seeds)}")
        print(f"{'='*70}\n")

        for variant_label, config in configs:
            for seed in self.seeds:
                tag = f"{variant_label}_seed{seed}"
                print(f"\n[Ablation] Running: {tag}")
                t0 = time.time()

                try:
                    result = run_fn(config=config, seed=seed, **run_kwargs)
                    if result is not None:
                        result["ablation_name"] = name
                        result["variant"] = variant_label
                        result["seed"] = seed
                        result["config_diff"] = self._config_diff(config)
                        result["run_time_sec"] = round(time.time() - t0, 1)
                        results.append(result)
                        self.results.append(result)
                except Exception as e:
                    print(f"[Ablation] FAILED: {tag}")
                    traceback.print_exc()
                    results.append({
                        "ablation_name": name,
                        "variant": variant_label,
                        "seed": seed,
                        "error": str(e),
                    })

        # Save ablation results
        self._save_ablation_results(name, results)
        return results

    def run_all(self, run_fn, **run_kwargs) -> List[Dict]:
        """Run all registered ablations."""
        all_results = []
        for name in self.ablations:
            results = self.run_ablation(name, run_fn, **run_kwargs)
            all_results.extend(results)
        return all_results

    def _config_diff(self, config: dict) -> Dict[str, Any]:
        """Compute the diff between a config and the base config."""
        diff = {}
        for name, variants in self.ablations.items():
            for key in variants:
                val = self._get_nested(config, key)
                base_val = self._get_nested(self.base_config, key)
                if val != base_val:
                    diff[key] = {"base": base_val, "current": val}
        return diff

    def _save_ablation_results(self, name: str, results: List[Dict]) -> None:
        """Save ablation results to JSON."""
        path = os.path.join(self.output_dir, f"ablation_{name}.json")

        # Make JSON-serializable
        def serialize(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean = json.loads(json.dumps(results, default=serialize))
        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"[Ablation] Results saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Result Aggregation & Leaderboard
# ─────────────────────────────────────────────────────────────────────────────


class ResultAggregator:
    """
    Aggregates experiment results and generates leaderboards.

    Supports mean ± std reporting across seeds, per-dataset/split breakdowns,
    and export to LaTeX/Markdown.
    """

    def __init__(self, results: Optional[List[Dict]] = None):
        self.results = results or []

    def add_results(self, results: List[Dict]) -> None:
        self.results.extend(results)

    def load_from_dir(self, results_dir: str) -> None:
        """Load all JSON result files from a directory."""
        for fname in os.listdir(results_dir):
            if fname.endswith(".json"):
                path = os.path.join(results_dir, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self.results.extend(data)
                    elif isinstance(data, dict):
                        self.results.append(data)
                except Exception as e:
                    print(f"[Warning] Could not load {path}: {e}")

    def aggregate(
        self,
        group_by: List[str] = None,
        metric_keys: List[str] = None,
    ) -> pd.DataFrame:
        """
        Aggregate results: mean ± std per group.

        Parameters
        ----------
        group_by : list of str
            Keys to group by (e.g. ['model', 'dataset', 'split'])
        metric_keys : list of str
            Metric names to aggregate (e.g. ['ci', 'rmse', 'pearson_r'])

        Returns
        -------
        pd.DataFrame with aggregated results
        """
        group_by = group_by or ["model", "dataset", "split"]
        metric_keys = metric_keys or ["ci", "rmse", "pearson_r"]

        # Filter valid results (those with metrics)
        valid = [r for r in self.results if "metrics" in r and not r.get("error")]

        if not valid:
            return pd.DataFrame()

        rows = []
        groups = defaultdict(list)

        for r in valid:
            key = tuple(r.get(k, "unknown") for k in group_by)
            groups[key].append(r["metrics"])

        for key, metric_list in sorted(groups.items()):
            row = dict(zip(group_by, key))
            row["n_runs"] = len(metric_list)
            for mk in metric_keys:
                values = [m.get(mk, 0.0) for m in metric_list if mk in m]
                if values:
                    row[f"{mk}_mean"] = np.mean(values)
                    row[f"{mk}_std"] = np.std(values)
                    row[f"{mk}_str"] = f"{np.mean(values):.4f}±{np.std(values):.4f}"
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_leaderboard(
        self,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        sort_by: str = "ci_mean",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Generate a leaderboard sorted by a metric."""
        df = self.aggregate()
        if df.empty:
            return df

        if dataset:
            df = df[df["dataset"] == dataset]
        if split:
            df = df[df["split"] == split]

        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)

        return df.reset_index(drop=True)

    def to_markdown(self, df: Optional[pd.DataFrame] = None) -> str:
        """Export DataFrame to Markdown table."""
        if df is None:
            df = self.aggregate()
        if df.empty:
            return "No results to display."

        # Select display columns
        display_cols = [c for c in df.columns if not c.endswith("_mean") and not c.endswith("_std")]
        str_cols = [c for c in df.columns if c.endswith("_str")]
        cols = [c for c in display_cols if c not in [s.replace("_str", "") for s in str_cols]] + str_cols

        return df[cols].to_markdown(index=False)

    def to_latex(self, df: Optional[pd.DataFrame] = None) -> str:
        """Export DataFrame to LaTeX table."""
        if df is None:
            df = self.aggregate()
        if df.empty:
            return "% No results"

        str_cols = [c for c in df.columns if c.endswith("_str")]
        base_cols = ["model", "dataset", "split", "n_runs"]
        cols = [c for c in base_cols if c in df.columns] + str_cols

        header = " & ".join(c.replace("_", r"\_") for c in cols) + r" \\"
        rows = []
        for _, row in df.iterrows():
            vals = [str(row.get(c, "")) for c in cols]
            rows.append(" & ".join(vals) + r" \\")

        table = [
            r"\begin{tabular}{" + "l" * len(cols) + "}",
            r"\toprule",
            header,
            r"\midrule",
        ] + rows + [
            r"\bottomrule",
            r"\end{tabular}",
        ]
        return "\n".join(table)


# ─────────────────────────────────────────────────────────────────────────────
# Predefined Ablation Configurations
# ─────────────────────────────────────────────────────────────────────────────


def setup_standard_ablations(runner: AblationRunner) -> None:
    """
    Register the standard ablation studies from the CL-DTA paper.

    Covers all key components for a comprehensive ablation table.
    """
    # 1. Cross-modal alignment weight
    runner.add_ablation("cross_modal_weight", {
        "pretrain.align_loss_weight": [0.0, 0.1, 0.3, 0.5, 1.0],
    })

    # 2. LLM initialization
    runner.add_ablation("llm_init", {
        "pretrain.use_pretrained_embeddings": [False, True],
    })

    # 3. Attention module
    runner.add_ablation("attention_module", {
        "train.use_attention_module": [False, True],
    })

    # 4. Uncertainty head
    runner.add_ablation("uncertainty_head", {
        "train.use_evidential": [False, True],
    })

    # 5. Temperature sensitivity
    runner.add_ablation("temperature", {
        "pretrain.temperature": [0.01, 0.05, 0.07, 0.1, 0.5],
    })

    # 6. Freeze strategy
    runner.add_ablation("freeze_strategy", {
        "train.freeze_strategy": ["frozen", "full_finetune", "gradual_unfreeze"],
    })

    print(f"[Ablation] Registered {len(runner.ablations)} ablation studies")
