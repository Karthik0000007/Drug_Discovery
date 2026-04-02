"""
contrastive_losses.py — Contrastive loss functions for pretraining.

Implements:
  - NT-Xent (Normalized Temperature-scaled Cross-Entropy) — primary
  - InfoNCE — ablation variant
  - Triplet loss — ablation variant
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
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)       # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature   # (2N, 2N)

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

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
        N = z1.size(0)
        # Cosine similarity: (N, N)
        sim = torch.mm(z1, z2.t())

        # Positive similarities (diagonal)
        pos_sim = sim.diag()

        # Mask out positive from negatives
        mask = torch.eye(N, device=z1.device, dtype=torch.bool)
        sim_neg = sim.masked_fill(mask, -1e9)

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
    drug_embeddings = F.normalize(drug_embeddings, p=2, dim=1)
    protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)
    
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
    """
    Unified contrastive loss computation for cross-modal pretraining.
    
    Computes:
    1. Intra-modal drug loss (drug_view1 vs drug_view2)
    2. Intra-modal protein loss (prot_view1 vs prot_view2)
    3. Cross-modal alignment loss (paired drug-protein embeddings)
    
    Parameters
    ----------
    drug_view1, drug_view2 : (B, D) drug augmentation views
    prot_view1, prot_view2 : (B, D) protein augmentation views
    paired_drug_emb : (B, D) drug embeddings for binding pairs
    paired_prot_emb : (B, D) protein embeddings for binding pairs
    temperature : temperature for contrastive losses
    align_loss_weight : weight for cross-modal alignment loss
    loss_fn_name : 'nt_xent' | 'infonce' | 'triplet'
    
    Returns
    -------
    dict with keys: 'loss_drug', 'loss_protein', 'loss_align', 'loss_total'
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
    
    # Combined loss
    loss_total = loss_drug + loss_protein + align_loss_weight * loss_align
    
    return {
        "loss_drug": loss_drug,
        "loss_protein": loss_protein,
        "loss_align": loss_align,
        "loss_total": loss_total,
    }


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
