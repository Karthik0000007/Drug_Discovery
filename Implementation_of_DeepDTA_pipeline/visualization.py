"""
visualization.py — Publication-quality plots for CL-DTA results.

Figures:
  Fig 2: CI drop comparison (grouped bar chart)
  Fig 3: Embedding t-SNE / UMAP (scatter plot)
  Fig 4: Augmentation ablation heatmap
  Fig 5: Training curves (loss / CI vs. epoch)
  Fig 6: Predicted vs. true affinity (scatter)
"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Optional UMAP / t-SNE
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ──────────────────────────────────────────────
# Style defaults
# ──────────────────────────────────────────────

STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

MODEL_COLORS = {
    "DeepDTA": "#3498db",
    "GraphDTA": "#2ecc71",
    "WideDTA": "#e67e22",
    "AttentionDTA": "#9b59b6",
    "CL-DTA": "#e74c3c",
}


def apply_style():
    plt.rcParams.update(STYLE)


# ──────────────────────────────────────────────
# Fig 2: CI drop comparison — grouped bar chart
# ──────────────────────────────────────────────

def plot_ci_comparison(
    results: Dict[str, Dict[str, float]],
    dataset: str = "DAVIS",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart of CI across split types for each model.

    Parameters
    ----------
    results : {model_name: {split_type: ci_value}}
        e.g. {"DeepDTA": {"random": 0.878, "cold_drug": 0.780, ...}, ...}
    dataset : dataset name for title.
    save_path : if provided, saves figure.
    """
    apply_style()
    splits = ["random", "cold_drug", "cold_target", "cold_both"]
    models = list(results.keys())
    n_models = len(models)
    n_splits = len(splits)

    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.15
    x = np.arange(n_splits)

    for i, model in enumerate(models):
        vals = [results[model].get(s, 0) for s in splits]
        color = MODEL_COLORS.get(model, "#95a5a6")
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, edgecolor="white")
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Split Type")
    ax.set_ylabel("Concordance Index (CI)")
    ax.set_title(f"CI Comparison — {dataset}")
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([s.replace("_", "\n") for s in splits])
    ax.legend(loc="upper right")
    ax.set_ylim(0.65, 1.0)
    ax.grid(axis="y", alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved CI comparison to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Fig 3: Embedding t-SNE / UMAP
# ──────────────────────────────────────────────

def plot_embedding_scatter(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None,
    perplexity: int = 30,
) -> plt.Figure:
    """
    2-D scatter plot of drug/protein embeddings.

    Parameters
    ----------
    embeddings : (N, D) array of encoder outputs.
    labels : (N,) array of cluster ids or scaffold labels.
    method : 'tsne' or 'umap'.
    """
    apply_style()

    if method == "umap" and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
    elif TSNE_AVAILABLE:
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords = reducer.fit_transform(embeddings)
    else:
        print("[viz] Neither t-SNE nor UMAP available; skipping embedding plot.")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter_kwargs = dict(s=15, alpha=0.7)
    if labels is not None:
        unique = np.unique(labels)
        cmap = plt.cm.get_cmap("tab20", len(unique))
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=cmap(i), label=f"{lbl}", **scatter_kwargs)
        if len(unique) <= 20:
            ax.legend(fontsize=7, markerscale=2, ncol=2)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], color="#3498db", **scatter_kwargs)

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    ax.grid(alpha=0.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved embedding scatter to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Fig 4: Augmentation ablation heatmap
# ──────────────────────────────────────────────

def plot_ablation_heatmap(
    data: Dict[str, Dict[str, float]],
    title: str = "Augmentation Ablation — CI",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of augmentation configs vs. split-type CI.

    Parameters
    ----------
    data : {aug_config_name: {split: ci}}
    """
    apply_style()
    configs = list(data.keys())
    splits = sorted({s for d in data.values() for s in d})
    matrix = np.zeros((len(configs), len(splits)))
    for i, cfg in enumerate(configs):
        for j, s in enumerate(splits):
            matrix[i, j] = data[cfg].get(s, 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(configs) * 0.4)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([s.replace("_", "\n") for s in splits])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CI")

    # Annotate cells
    for i in range(len(configs)):
        for j in range(len(splits)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved ablation heatmap to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Fig 5: Training curves
# ──────────────────────────────────────────────

def plot_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metrics: Tuple[str, ...] = ("train_loss", "val_rmse", "val_ci"),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plots of training metrics across epochs.

    Parameters
    ----------
    histories : {model_name: {"train_loss": [...], "val_rmse": [...], "val_ci": [...]}}
    """
    apply_style()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model_name, hist in histories.items():
            if metric in hist:
                vals = hist[metric]
                color = MODEL_COLORS.get(model_name, None)
                ax.plot(range(1, len(vals) + 1), vals, label=model_name, color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Training Curves", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved training curves to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Fig 6: Predicted vs. true scatter
# ──────────────────────────────────────────────

def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "CL-DTA",
    dataset: str = "DAVIS",
    split: str = "cold_drug",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot of predicted vs. true affinities with R² annotation."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, s=10, alpha=0.4, color=MODEL_COLORS.get(model_name, "#3498db"))

    # Diagonal line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", linewidth=1, alpha=0.6)

    # Statistics
    from scipy.stats import pearsonr
    r_val, _ = pearsonr(y_true, y_pred)
    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ax.text(0.05, 0.92, f"Pearson r = {r_val:.3f}\nRMSE = {rmse_val:.3f}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("True Affinity")
    ax.set_ylabel("Predicted Affinity")
    ax.set_title(f"{model_name} — {dataset} ({split})")
    ax.grid(alpha=0.2)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved pred vs true to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Utility: load results JSON and produce all figs
# ──────────────────────────────────────────────

def generate_all_figures(results_dir: str, output_dir: str = "figures"):
    """
    Scan a results directory for experiment JSONs and generate all plots.

    Expects JSON files with structure:
      {"model": str, "dataset": str, "split": str, "seed": int,
       "metrics": {"ci": float, "rmse": float, ...},
       "train_losses": [...], "val_rmses": [...], "val_cis": [...]}
    """
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    if not json_files:
        print(f"[viz] No JSON result files found in {results_dir}")
        return

    # Aggregate results
    all_results = []
    for fname in json_files:
        with open(os.path.join(results_dir, fname)) as f:
            all_results.append(json.load(f))

    # Group by dataset
    datasets = set(r.get("dataset", "unknown") for r in all_results)
    for ds in datasets:
        ds_results = [r for r in all_results if r.get("dataset") == ds]

        # Build CI comparison data
        ci_data: Dict[str, Dict[str, List[float]]] = {}
        for r in ds_results:
            model = r.get("model", "unknown")
            split = r.get("split", "unknown")
            ci = r.get("metrics", {}).get("ci", 0)
            ci_data.setdefault(model, {}).setdefault(split, []).append(ci)

        # Average over seeds
        ci_avg = {
            m: {s: np.mean(vals) for s, vals in splits.items()}
            for m, splits in ci_data.items()
        }
        plot_ci_comparison(ci_avg, dataset=ds,
                           save_path=os.path.join(output_dir, f"ci_comparison_{ds}.png"))

        # Training curves (use first seed)
        histories: Dict[str, Dict[str, List[float]]] = {}
        for r in ds_results:
            model = r.get("model", "unknown")
            if model not in histories and "train_losses" in r:
                histories[model] = {
                    "train_loss": r["train_losses"],
                    "val_rmse": r.get("val_rmses", []),
                    "val_ci": r.get("val_cis", []),
                }
        if histories:
            plot_training_curves(
                histories,
                save_path=os.path.join(output_dir, f"training_curves_{ds}.png"),
            )

    print(f"[viz] All figures saved to {output_dir}/")
