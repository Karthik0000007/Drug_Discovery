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
