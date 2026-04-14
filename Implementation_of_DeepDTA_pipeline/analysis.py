"""
analysis.py — Phase 8: Interpretability and Theoretical Analysis.

Implements:
  - Mutual Information estimation between embeddings and structural labels
  - Embedding similarity analysis (intra-class vs inter-class)
  - Contrastive learning effect analysis (temperature, variance, collapse detection)
  - Attention map extraction and biological correlation
  - Full analysis pipeline from checkpoint

Produces publication-quality quantitative analysis outputs.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mutual Information Analysis
# ─────────────────────────────────────────────────────────────────────────────


def compute_mutual_information(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 30,
    method: str = "knn",
) -> float:
    """
    Estimate Mutual Information between embeddings and categorical labels.

    Parameters
    ----------
    embeddings : (N, D) embedding matrix
    labels : (N,) categorical labels (e.g. protein family, drug scaffold, affinity bin)
    n_bins : int
        Number of bins for discretization (histogram method)
    method : str
        'knn' (uses sklearn MI estimator) or 'histogram' (binned estimate)

    Returns
    -------
    mi_score : float
        Mutual information in nats
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)

    if method == "knn":
        try:
            from sklearn.feature_selection import mutual_info_classif
            # Use first few PCA components to avoid curse of dimensionality
            from sklearn.decomposition import PCA
            n_components = min(10, embeddings.shape[1], embeddings.shape[0] - 1)
            if n_components < 1:
                return 0.0
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
            mi_scores = mutual_info_classif(reduced, labels, discrete_features=False, n_neighbors=5)
            return float(mi_scores.mean())
        except ImportError:
            method = "histogram"  # fallback

    if method == "histogram":
        # Simple binned MI estimate using PCA projection
        from sklearn.decomposition import PCA
        n_components = min(3, embeddings.shape[1], embeddings.shape[0] - 1)
        if n_components < 1:
            return 0.0
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)

        # Discretize continuous embeddings
        mi_total = 0.0
        for dim in range(n_components):
            x = reduced[:, dim]
            bins = np.linspace(x.min(), x.max() + 1e-8, n_bins + 1)
            x_binned = np.digitize(x, bins)

            # Joint and marginal probabilities
            unique_labels = np.unique(labels)
            unique_bins = np.unique(x_binned)
            N = len(labels)

            for lb in unique_labels:
                for bn in unique_bins:
                    p_joint = ((labels == lb) & (x_binned == bn)).sum() / N
                    p_label = (labels == lb).sum() / N
                    p_bin = (x_binned == bn).sum() / N
                    if p_joint > 0 and p_label > 0 and p_bin > 0:
                        mi_total += p_joint * np.log(p_joint / (p_label * p_bin))

        return float(mi_total / n_components)

    raise ValueError(f"Unknown method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Similarity Analysis
# ─────────────────────────────────────────────────────────────────────────────


def compare_embedding_similarity(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute intra-class and inter-class cosine similarity.

    After contrastive learning:
    - Intra-class similarity should INCREASE (same family → closer)
    - Inter-class similarity should DECREASE (different families → farther)

    Parameters
    ----------
    embeddings : (N, D) embedding matrix
    labels : (N,) categorical labels

    Returns
    -------
    dict with 'intra_class_sim', 'inter_class_sim', 'separation_ratio'
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms

    unique_labels = np.unique(labels)

    # Sample for efficiency
    max_samples = min(2000, len(embeddings))
    if len(embeddings) > max_samples:
        idx = np.random.choice(len(embeddings), max_samples, replace=False)
        normalized = normalized[idx]
        labels = labels[idx]

    # Compute similarity matrix
    sim_matrix = normalized @ normalized.T

    intra_sims = []
    inter_sims = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])

    intra_mean = float(np.mean(intra_sims)) if intra_sims else 0.0
    inter_mean = float(np.mean(inter_sims)) if inter_sims else 0.0

    separation = intra_mean - inter_mean

    return {
        "intra_class_sim": intra_mean,
        "inter_class_sim": inter_mean,
        "separation": separation,
        "separation_ratio": float(intra_mean / (inter_mean + 1e-8)),
        "num_intra_pairs": len(intra_sims),
        "num_inter_pairs": len(inter_sims),
    }


def compare_before_after(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compare embedding quality before and after contrastive pretraining.

    Parameters
    ----------
    embeddings_before : (N, D) embeddings before contrastive learning
    embeddings_after : (N, D) embeddings after contrastive learning
    labels : (N,) categorical labels

    Returns
    -------
    dict with 'before' and 'after' similarity metrics, and 'improvement' deltas
    """
    before = compare_embedding_similarity(embeddings_before, labels)
    after = compare_embedding_similarity(embeddings_after, labels)

    improvement = {
        k: after[k] - before[k]
        for k in ["intra_class_sim", "inter_class_sim", "separation"]
    }

    return {
        "before": before,
        "after": after,
        "improvement": improvement,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Learning Behavior Analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_contrastive_behavior(
    embeddings: np.ndarray,
    temperature: float = 0.07,
) -> Dict[str, float]:
    """
    Analyze properties of learned contrastive embeddings.

    Checks for:
    - Embedding spread (variance)
    - Collapse detection (if all embeddings converge to same point)
    - Temperature sensitivity
    - Uniformity on the hypersphere

    Parameters
    ----------
    embeddings : (N, D) embedding matrix (should be L2-normalized)
    temperature : float
        Temperature used during training

    Returns
    -------
    dict with analysis metrics
    """
    embeddings = np.asarray(embeddings, dtype=float)

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms

    # 1. Variance per dimension
    var_per_dim = normalized.var(axis=0)
    mean_var = float(var_per_dim.mean())
    min_var = float(var_per_dim.min())

    # 2. Collapse detection: if variance drops below threshold
    is_collapsed = mean_var < 1e-4
    collapse_ratio = float((var_per_dim < 1e-5).sum() / len(var_per_dim))

    # 3. Embedding norms (should be ~1 if normalized)
    original_norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = float(original_norms.mean())
    std_norm = float(original_norms.std())

    # 4. Uniformity on hypersphere (Wang & Isola 2020)
    # Measures how uniformly distributed embeddings are
    # Lower = more uniform (better)
    max_samples = min(1000, len(normalized))
    if len(normalized) > max_samples:
        idx = np.random.choice(len(normalized), max_samples, replace=False)
        sample = normalized[idx]
    else:
        sample = normalized
    # Pairwise squared distances
    sq_dist = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    uniformity = float(np.log(np.exp(-2 * sq_dist).mean()))

    # 5. Alignment (average pairwise similarity — lower diversity = higher alignment)
    sim_mean = float((sample @ sample.T).mean())

    # 6. Temperature effect estimate
    scaled_sims = (sample @ sample.T) / temperature
    max_logit_range = float(scaled_sims.max() - scaled_sims.min())

    return {
        "mean_variance": mean_var,
        "min_variance": min_var,
        "is_collapsed": bool(is_collapsed),
        "collapse_ratio": collapse_ratio,
        "mean_norm": mean_norm,
        "std_norm": std_norm,
        "uniformity": uniformity,
        "mean_similarity": sim_mean,
        "temperature": temperature,
        "logit_range_at_temp": max_logit_range,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Attention Map Analysis (Phase 6 integration)
# ─────────────────────────────────────────────────────────────────────────────


def extract_attention_maps(
    model,
    dataloader,
    device: str = "cuda",
    max_samples: int = 100,
) -> Dict[str, np.ndarray]:
    """
    Extract attention weights from PocketGuidedAttention module.

    Parameters
    ----------
    model : nn.Module
        Model with attention module
    dataloader : DataLoader
        Data to extract attention from
    device : str
        Device
    max_samples : int
        Maximum number of samples to process

    Returns
    -------
    dict with 'attention_weights', 'sequences', 'predictions'
    """
    import torch

    model.eval()
    all_weights = []
    all_preds = []
    all_trues = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if count >= max_samples:
                break

            smiles = batch["smiles"].to(device)
            seq = batch["seq"].to(device)
            aff = batch.get("aff", None)

            # Check if model has attention module
            if hasattr(model, "attention_module") and model.attention_module is not None:
                # Get intermediate features
                d = model.drug_encoder(smiles)
                # Need sequence-level protein features
                if hasattr(model, "prot_seq_features"):
                    p_seq = model.prot_seq_features(seq)
                    enhanced_d, attn_w = model.attention_module(d, p_seq)
                    all_weights.append(attn_w.cpu().numpy())

            if aff is not None:
                all_trues.append(aff.cpu().numpy())

            count += smiles.size(0)

    result = {}
    if all_weights:
        result["attention_weights"] = np.concatenate(all_weights, axis=0)
    if all_trues:
        result["true_affinities"] = np.concatenate(all_trues, axis=0)

    return result


def compute_attention_entropy(attn_weights: np.ndarray) -> np.ndarray:
    """
    Compute entropy of attention distribution per sample.

    Higher entropy = more dispersed attention (less focused).
    Lower entropy = more focused attention.

    Parameters
    ----------
    attn_weights : (N, H, L) attention weights

    Returns
    -------
    entropy : (N,) average entropy per sample
    """
    # Clamp for numerical stability
    attn_clamped = np.clip(attn_weights, 1e-10, 1.0)
    entropy = -(attn_clamped * np.log(attn_clamped)).sum(axis=-1)  # (N, H)
    return entropy.mean(axis=-1)  # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Full Analysis Pipeline
# ─────────────────────────────────────────────────────────────────────────────


def run_analysis_pipeline(
    checkpoint_path: str,
    output_dir: str = "analysis_outputs",
    device: str = "cuda",
) -> Dict[str, any]:
    """
    Run complete analysis pipeline from a saved checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint (.pt)
    output_dir : str
        Directory to save analysis outputs
    device : str
        Device

    Returns
    -------
    dict with all analysis results
    """
    import torch

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return results

    ckpt = torch.load(checkpoint_path, map_location=device)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # If embeddings are saved in checkpoint, analyze them
    if "embeddings" in ckpt:
        embeddings = ckpt["embeddings"]
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        # Contrastive behavior
        behavior = analyze_contrastive_behavior(embeddings)
        results["contrastive_behavior"] = behavior
        logger.info(f"Contrastive behavior: {behavior}")

        # If labels available
        if "labels" in ckpt:
            labels = ckpt["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            mi_score = compute_mutual_information(embeddings, labels)
            results["mutual_information"] = mi_score
            logger.info(f"Mutual Information: {mi_score:.4f}")

            sim_analysis = compare_embedding_similarity(embeddings, labels)
            results["similarity_analysis"] = sim_analysis
            logger.info(f"Similarity analysis: {sim_analysis}")

    # Save results
    results_path = os.path.join(output_dir, "analysis_results.json")

    # Make JSON serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    logger.info(f"Analysis results saved to {results_path}")

    return results
