"""
contrastive_losses.py — Contrastive loss functions for pretraining.

Implements:
  - NT-Xent (Normalized Temperature-scaled Cross-Entropy) — primary
  - InfoNCE — ablation variant
  - Triplet loss — ablation variant
  - Embedding Alignment Loss (Phase 4) — aligns pretrained LLM embeddings with learned representations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (SimCLR).

    Given a batch of N positive pairs (2N total views), treats all other
    2N-2 views per anchor as negatives.

    L = -1/(2N) Σ [ log(exp(sim(z_i, z_i+)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
                   + log(exp(sim(z_i+, z_i)/τ) / Σ_{k≠i} exp(sim(z_i+, z_k)/τ)) ]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (N, D) ℓ₂-normalized projection vectors for the two views.

        Returns
        -------
        Scalar NT-Xent loss.
        """
        z1 = z1.float()
        z2 = z2.float()
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)       # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature   # (2N, 2N)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, torch.finfo(sim.dtype).min)

        # Positive pair indices: (i, i+N) and (i+N, i) for i in 0..N-1
        pos_idx = torch.cat([
            torch.arange(N, 2 * N, device=z.device),
            torch.arange(0, N, device=z.device),
        ])  # (2N,)

        # Gather positive similarities
        pos_sim = sim[torch.arange(2 * N, device=z.device), pos_idx]   # (2N,)

        # NT-Xent: cross-entropy where positive pair is the correct class
        # log_softmax over dim=1, pick the positive index
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (asymmetric variant).

    Only uses z1 as anchor, z2 as positive (N-1 negatives from other z2 entries).

    L = -1/N Σ log( exp(sim(z1_i, z2_i)/τ) / Σ_j exp(sim(z1_i, z2_j)/τ) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1.float()
        z2 = z2.float()
        N = z1.size(0)
        # Similarity matrix: (N, N) — each row is z1_i vs all z2_j
        sim = torch.mm(z1, z2.t()) / self.temperature
        # Labels: diagonal entries are positives
        labels = torch.arange(N, device=z1.device)
        loss = F.cross_entropy(sim, labels)
        return loss


class TripletLoss(nn.Module):
    """
    Triplet margin loss for contrastive pretraining (ablation).

    For each positive pair (z1_i, z2_i), hardest negative is selected from
    other z2 entries. Loss = max(0, sim(anchor, neg) - sim(anchor, pos) + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = z1.float()
        z2 = z2.float()
        N = z1.size(0)
        # Cosine similarity: (N, N)
        sim = torch.mm(z1, z2.t())

        # Positive similarities (diagonal)
        pos_sim = sim.diag()

        # Mask out positive from negatives
        mask = torch.eye(N, device=z1.device, dtype=torch.bool)
        sim_neg = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

        # Hard negative: max similarity to non-positive
        hard_neg_sim, _ = sim_neg.max(dim=1)

        # Triplet loss: max(0, neg_sim - pos_sim + margin)
        loss = F.relu(hard_neg_sim - pos_sim + self.margin)
        return loss.mean()


# ──────────────────────────────────────────────
# Phase 1: Cross-Modal Alignment Loss
# ──────────────────────────────────────────────

def cross_modal_alignment_loss(
    drug_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Cross-modal alignment loss for drug-protein binding pairs.
    
    Applies NT-Xent-style contrastive loss across modalities where:
    - Positive pairs: (drug_i, protein_i) — known binding pairs (diagonal)
    - Negative pairs: all other (drug_i, protein_j) combinations in batch
    
    Parameters
    ----------
    drug_embeddings : (B, D) ℓ₂-normalized drug projection vectors
    protein_embeddings : (B, D) ℓ₂-normalized protein projection vectors
    temperature : temperature parameter for scaling similarities
    
    Returns
    -------
    Scalar cross-modal alignment loss
    
    Notes
    -----
    Each index i corresponds to a known binding pair (drug_i, protein_i).
    The loss encourages binding pairs to have high similarity while
    pushing apart non-binding combinations.
    """
    B = drug_embeddings.size(0)
    
    # Normalize embeddings (ensure unit norm)
    drug_embeddings = F.normalize(drug_embeddings.float(), p=2, dim=1)
    protein_embeddings = F.normalize(protein_embeddings.float(), p=2, dim=1)
    
    # Compute similarity matrix: (B, B)
    # sim[i, j] = cosine_similarity(drug_i, protein_j)
    sim_matrix = torch.mm(drug_embeddings, protein_embeddings.t()) / temperature
    
    # Positive pairs are on the diagonal
    # Labels for cross-entropy: each drug_i should match protein_i
    labels = torch.arange(B, device=drug_embeddings.device)
    
    # Compute loss in both directions (drug→protein and protein→drug)
    # This is symmetric and ensures both modalities learn aligned representations
    loss_d2p = F.cross_entropy(sim_matrix, labels)  # drug as query, protein as key
    loss_p2d = F.cross_entropy(sim_matrix.t(), labels)  # protein as query, drug as key
    
    # Average bidirectional loss
    loss = (loss_d2p + loss_p2d) / 2.0
    
    return loss


# ──────────────────────────────────────────────
# Phase 4: Embedding Alignment Loss (LLM ↔ Learned)
# ──────────────────────────────────────────────

def embedding_alignment_loss(
    llm_embeddings: torch.Tensor,
    learned_embeddings: torch.Tensor,
    distance_metric: str = "cosine",
) -> torch.Tensor:
    """
    Alignment loss between pretrained LLM embeddings and learned representations.

    Encourages the learned embeddings to be consistent with and leverage
    the knowledge encoded in pretrained language models.

    Parameters
    ----------
    llm_embeddings : (B, D) ℓ₂-normalized pretrained LLM embeddings
    learned_embeddings : (B, D) ℓ₂-normalized learned encoder embeddings
    distance_metric : 'cosine' | 'mse'
        Metric for computing distance between embeddings

    Returns
    -------
    Scalar alignment loss

    Notes
    -----
    Both embeddings should be normalized to unit norm. The loss measures
    the average cosine distance or MSE between paired embeddings, pushing
    learned representations towards the pretrained knowledge space while
    allowing task-specific adaptation.
    """
    # Ensure unit norm
    llm_emb = F.normalize(llm_embeddings.float(), p=2, dim=1)
    learned_emb = F.normalize(learned_embeddings.float(), p=2, dim=1)

    if distance_metric == "cosine":
        # Cosine distance: 1 - cosine_similarity
        # For unit norm: 1 - dot product
        cos_sim = (llm_emb * learned_emb).sum(dim=1)  # (B,)
        loss = (1.0 - cos_sim).mean()
    elif distance_metric == "mse":
        # Euclidean distance for unit norm vectors
        loss = torch.norm(llm_emb - learned_emb, p=2, dim=1).mean()
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")

    return loss


def compute_contrastive_losses_with_alignment(
    drug_view1: torch.Tensor,
    drug_view2: torch.Tensor,
    prot_view1: torch.Tensor,
    prot_view2: torch.Tensor,
    paired_drug_emb: torch.Tensor,
    paired_prot_emb: torch.Tensor,
    drug_align_emb: torch.Tensor | None = None,
    prot_align_emb: torch.Tensor | None = None,
    drug_llm_emb: torch.Tensor | None = None,
    prot_llm_emb: torch.Tensor | None = None,
    temperature: float = 0.07,
    align_loss_weight: float = 0.5,
    llm_align_weight: float = 0.3,
    loss_fn_name: str = "nt_xent",
    distance_metric: str = "cosine",
) -> dict:
    """
    Unified contrastive loss computation with optional LLM embedding alignment.

    Computes:
    1. Intra-modal drug loss (drug_view1 vs drug_view2)
    2. Intra-modal protein loss (prot_view1 vs prot_view2)
    3. Cross-modal alignment loss (paired drug-protein embeddings)
    4. (Optional) LLM alignment loss (learned representations → pretrained embeddings)

    Parameters
    ----------
    drug_view1, drug_view2 : (B, D) drug augmentation views
    prot_view1, prot_view2 : (B, D) protein augmentation views
    paired_drug_emb : (B, D) drug embeddings for binding pairs
    paired_prot_emb : (B, D) protein embeddings for binding pairs
    drug_llm_emb : (B, D) or None, pretrained drug LLM embeddings
    prot_llm_emb : (B, D) or None, pretrained protein LLM embeddings
    drug_align_emb : (B, D) or None, learned embeddings used for LLM alignment
    prot_align_emb : (B, D) or None, learned embeddings used for LLM alignment
    temperature : temperature for contrastive losses
    align_loss_weight : weight for cross-modal alignment loss
    llm_align_weight : weight for LLM alignment loss
    loss_fn_name : 'nt_xent' | 'infonce' | 'triplet'
    distance_metric : 'cosine' | 'mse' for LLM alignment

    Returns
    -------
    dict with keys: 'loss_drug', 'loss_protein', 'loss_align', 'loss_llm_drug',
                    'loss_llm_prot', 'loss_total'
    """
    # Get intra-modal loss function
    loss_kwargs = {"temperature": temperature} if loss_fn_name != "triplet" else {"margin": 1.0}
    intra_loss_fn = get_contrastive_loss(loss_fn_name, **loss_kwargs)

    # Compute intra-modal losses
    loss_drug = intra_loss_fn(drug_view1, drug_view2)
    loss_protein = intra_loss_fn(prot_view1, prot_view2)

    # Compute cross-modal alignment loss
    loss_align = cross_modal_alignment_loss(
        paired_drug_emb, paired_prot_emb, temperature=temperature
    )

    # Compute LLM embedding alignment losses (Phase 4)
    loss_llm_drug = torch.tensor(0.0, device=drug_view1.device)
    loss_llm_prot = torch.tensor(0.0, device=prot_view1.device)

    # Alignment targets (default to paired embeddings if not provided)
    align_drug_target = drug_align_emb if drug_align_emb is not None else paired_drug_emb
    align_prot_target = prot_align_emb if prot_align_emb is not None else paired_prot_emb

    if drug_llm_emb is not None:
        loss_llm_drug = embedding_alignment_loss(
            drug_llm_emb, align_drug_target, distance_metric=distance_metric
        )

    if prot_llm_emb is not None:
        loss_llm_prot = embedding_alignment_loss(
            prot_llm_emb, align_prot_target, distance_metric=distance_metric
        )

    # Combined loss
    loss_total = (
        loss_drug + loss_protein
        + align_loss_weight * loss_align
        + llm_align_weight * (loss_llm_drug + loss_llm_prot) / 2.0
    )

    return {
        "loss_drug": loss_drug,
        "loss_protein": loss_protein,
        "loss_align": loss_align,
        "loss_llm_drug": loss_llm_drug,
        "loss_llm_prot": loss_llm_prot,
        "loss_total": loss_total,
    }


def compute_contrastive_losses(
    drug_view1: torch.Tensor,
    drug_view2: torch.Tensor,
    prot_view1: torch.Tensor,
    prot_view2: torch.Tensor,
    paired_drug_emb: torch.Tensor,
    paired_prot_emb: torch.Tensor,
    temperature: float = 0.07,
    align_loss_weight: float = 0.5,
    loss_fn_name: str = "nt_xent",
) -> dict:
    """Original compute_contrastive_losses (backward compatible)."""
    return compute_contrastive_losses_with_alignment(
        drug_view1, drug_view2, prot_view1, prot_view2,
        paired_drug_emb, paired_prot_emb,
        drug_llm_emb=None, prot_llm_emb=None,
        temperature=temperature, align_loss_weight=align_loss_weight,
        llm_align_weight=0.0, loss_fn_name=loss_fn_name,
    )


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

LOSS_REGISTRY = {
    "nt_xent": NTXentLoss,
    "infonce": InfoNCELoss,
    "triplet": TripletLoss,
}


def get_contrastive_loss(name: str, **kwargs) -> nn.Module:
    """Instantiate a contrastive loss by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Choose from {list(LOSS_REGISTRY.keys())}.")
    return LOSS_REGISTRY[name](**kwargs)
