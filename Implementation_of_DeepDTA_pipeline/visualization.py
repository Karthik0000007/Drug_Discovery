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
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 200,
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


# ──────────────────────────────────────────────
# Phase 8: Attention Heatmap
# ──────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: np.ndarray,
    sequence: Optional[str] = None,
    title: str = "Attention Weights per Residue",
    save_path: Optional[str] = None,
    max_residues: int = 100,
    show_sequence_chars: bool = False,
    x_tick_step: Optional[int] = None,
) -> plt.Figure:
    """
    Plot attention heatmap over protein residue positions.

    Parameters
    ----------
    attn_weights : (H, L) or (L,) attention weights
        H = num_heads, L = protein sequence length
    sequence : str, optional
        Protein sequence for x-axis labels
    max_residues : int
        Maximum residues to display (truncate for readability)
    """
    apply_style()

    if attn_weights.ndim == 1:
        attn_weights = attn_weights[np.newaxis, :]

    L = min(attn_weights.shape[-1], max_residues)
    attn_weights = attn_weights[:, :L]

    fig_width = max(12, L * 0.22)
    fig_height = max(4.5, attn_weights.shape[0] * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(attn_weights, cmap="viridis", aspect="auto")

    ax.set_xlabel("Protein Residue Position")
    ax.set_ylabel("Attention Head")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.03, label="Attention Weight")
    cbar.ax.tick_params(labelsize=13)

    if x_tick_step is None:
        if L <= 25:
            x_tick_step = 1
        elif L <= 50:
            x_tick_step = 5
        else:
            x_tick_step = 10

    tick_positions = np.arange(0, L, x_tick_step)
    if len(tick_positions) == 0 or tick_positions[-1] != L - 1:
        tick_positions = np.append(tick_positions, L - 1)
    ax.set_xticks(tick_positions)

    if sequence and show_sequence_chars and L <= 30:
        tick_labels = [sequence[pos] for pos in tick_positions]
    else:
        tick_labels = [str(pos + 1) for pos in tick_positions]
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=12)

    ax.set_yticks(range(attn_weights.shape[0]))
    ax.set_yticklabels([f"Head {i}" for i in range(attn_weights.shape[0])], fontsize=14)
    ax.tick_params(axis="x", pad=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved attention heatmap to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Phase 7/8: Uncertainty Calibration Curve
# ──────────────────────────────────────────────

def plot_uncertainty_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
    title: str = "Uncertainty Calibration",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Calibration plot: predicted uncertainty vs actual error.

    Parameters
    ----------
    y_true : (N,) true values
    y_pred : (N,) predictions
    uncertainties : (N,) predicted uncertainties
    """
    apply_style()
    errors = np.abs(y_true - y_pred)
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))

    bin_errors = []
    bin_uncerts = []
    for i in range(n_bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1] + 1e-8)
        if mask.sum() > 0:
            bin_errors.append(errors[mask].mean())
            bin_uncerts.append(uncertainties[mask].mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: calibration curve
    ax = axes[0]
    if bin_uncerts:
        ax.scatter(bin_uncerts, bin_errors, s=50, color="#e74c3c", zorder=5)
        lo = min(min(bin_uncerts), min(bin_errors))
        hi = max(max(bin_uncerts), max(bin_errors))
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Predicted Uncertainty")
    ax.set_ylabel("Actual Error")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: uncertainty vs error scatter
    ax2 = axes[1]
    ax2.scatter(uncertainties, errors, s=5, alpha=0.3, color="#3498db")
    ax2.set_xlabel("Predicted Uncertainty")
    ax2.set_ylabel("|y_true - y_pred|")
    ax2.set_title("Uncertainty vs. Error")
    ax2.grid(alpha=0.3)

    if errors.std() > 0 and uncertainties.std() > 0:
        r = np.corrcoef(uncertainties, errors)[0, 1]
        ax2.text(0.05, 0.92, f"Pearson r = {r:.3f}",
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved calibration plot to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Phase 8: Embedding Comparison (Before / After)
# ──────────────────────────────────────────────

def plot_embedding_comparison(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Space: Before vs After Contrastive Learning",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of embedding spaces before and after training.

    Parameters
    ----------
    embeddings_before : (N, D) embeddings before contrastive pretraining
    embeddings_after : (N, D) embeddings after contrastive pretraining
    labels : (N,) optional category labels for coloring
    method : 'tsne' or 'umap'
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, emb, subtitle in zip(
        axes,
        [embeddings_before, embeddings_after],
        ["Before Pretraining", "After Pretraining"],
    ):
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(emb)
        elif TSNE_AVAILABLE:
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            coords = reducer.fit_transform(emb)
        else:
            ax.text(0.5, 0.5, "t-SNE/UMAP not available",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        if labels is not None:
            unique = np.unique(labels)
            cmap = plt.cm.get_cmap("tab20", len(unique))
            for i, lbl in enumerate(unique):
                mask = labels == lbl
                ax.scatter(coords[mask, 0], coords[mask, 1],
                           color=cmap(i), s=10, alpha=0.6, label=str(lbl))
            if len(unique) <= 15:
                ax.legend(fontsize=6, markerscale=2)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6, color="#3498db")

        ax.set_title(subtitle)
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved embedding comparison to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Phase 8: Mutual Information Computation
# ──────────────────────────────────────────────

def compute_mutual_information(
    drug_embeddings: np.ndarray,
    prot_embeddings: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Compute mutual information between drug and protein embeddings.
    
    Uses histogram-based estimation with discretization.
    
    Parameters
    ----------
    drug_embeddings : (N, D_drug) array
    prot_embeddings : (N, D_prot) array
    n_bins : int
        Number of bins for discretization
    
    Returns
    -------
    mi : float
        Average mutual information across dimension pairs
    """
    from sklearn.metrics import mutual_info_score
    
    # Discretize embeddings
    drug_discrete = np.zeros_like(drug_embeddings, dtype=int)
    prot_discrete = np.zeros_like(prot_embeddings, dtype=int)
    
    for d in range(drug_embeddings.shape[1]):
        drug_discrete[:, d] = np.digitize(
            drug_embeddings[:, d],
            bins=np.linspace(drug_embeddings[:, d].min(), drug_embeddings[:, d].max(), n_bins)
        )
    
    for p in range(prot_embeddings.shape[1]):
        prot_discrete[:, p] = np.digitize(
            prot_embeddings[:, p],
            bins=np.linspace(prot_embeddings[:, p].min(), prot_embeddings[:, p].max(), n_bins)
        )
    
    # Compute MI for each dimension pair and average
    mi_scores = []
    for d in range(min(drug_embeddings.shape[1], 10)):  # Sample first 10 dims for efficiency
        for p in range(min(prot_embeddings.shape[1], 10)):
            mi = mutual_info_score(drug_discrete[:, d], prot_discrete[:, p])
            mi_scores.append(mi)
    
    return float(np.mean(mi_scores))


def plot_mi_evolution(
    mi_scores: List[float],
    epochs: Optional[List[int]] = None,
    title: str = "Mutual Information Evolution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mutual information evolution during training.
    
    Parameters
    ----------
    mi_scores : list of MI values at different checkpoints
    epochs : list of epoch numbers (optional)
    """
    apply_style()
    
    if epochs is None:
        epochs = list(range(1, len(mi_scores) + 1))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, mi_scores, marker='o', linewidth=2, markersize=6, color='#e74c3c')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mutual Information (bits)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    
    # Annotate first and last
    if len(mi_scores) > 0:
        ax.annotate(f'{mi_scores[0]:.3f}', xy=(epochs[0], mi_scores[0]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.annotate(f'{mi_scores[-1]:.3f}', xy=(epochs[-1], mi_scores[-1]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved MI evolution plot to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Phase 8: Multi-Head Attention Comparison
# ──────────────────────────────────────────────

def plot_multihead_attention_comparison(
    attention_weights: np.ndarray,
    sequence: Optional[str] = None,
    pocket_mask: Optional[np.ndarray] = None,
    title: str = "Multi-Head Attention Analysis",
    save_path: Optional[str] = None,
    max_residues: int = 100,
) -> plt.Figure:
    """
    Compare attention patterns across multiple heads with pocket highlighting.
    
    Parameters
    ----------
    attention_weights : (num_heads, seq_len) array
    sequence : str, optional protein sequence
    pocket_mask : (seq_len,) binary mask for pocket residues
    """
    apply_style()
    
    num_heads = attention_weights.shape[0]
    L = min(attention_weights.shape[1], max_residues)
    attention_weights = attention_weights[:, :L]
    
    if pocket_mask is not None:
        pocket_mask = pocket_mask[:L]
    
    fig, axes = plt.subplots(num_heads, 1, figsize=(14, 3 * num_heads))
    if num_heads == 1:
        axes = [axes]
    
    for head_idx, ax in enumerate(axes):
        weights = attention_weights[head_idx]
        positions = np.arange(L)
        
        # Color by pocket/non-pocket
        if pocket_mask is not None:
            pocket_positions = positions[pocket_mask == 1]
            non_pocket_positions = positions[pocket_mask == 0]
            
            ax.bar(pocket_positions, weights[pocket_mask == 1],
                  color='#e74c3c', alpha=0.7, label='Pocket')
            ax.bar(non_pocket_positions, weights[pocket_mask == 0],
                  color='#3498db', alpha=0.7, label='Non-pocket')
            ax.legend(loc='upper right')
        else:
            ax.bar(positions, weights, color='#3498db', alpha=0.7)
        
        ax.set_ylabel(f'Head {head_idx}')
        ax.set_ylim(0, weights.max() * 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Compute entropy
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        ax.text(0.98, 0.95, f'Entropy: {entropy:.3f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Protein Residue Position')
    fig.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        print(f"[viz] Saved multi-head attention comparison to {save_path}")
    return fig


# ──────────────────────────────────────────────
# Phase 11: Publication Artifact Generation
# ──────────────────────────────────────────────

def generate_paper_artifacts(
    results_dir: str,
    output_dir: str = "paper/figures",
) -> None:
    """
    Generate all publication-ready figures from experiment results.

    Produces:
    - Main results table (all models × splits × metrics)
    - Ablation heatmaps / bar charts
    - Training curves comparison
    - Uncertainty calibration curves
    """
    os.makedirs(output_dir, exist_ok=True)
    generate_all_figures(results_dir, output_dir)
    print(f"[viz] Publication artifacts saved to {output_dir}/")
