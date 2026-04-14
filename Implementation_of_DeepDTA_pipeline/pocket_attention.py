"""
pocket_attention.py — Phase 6: Pocket-Guided Cross-Attention for DTA.

Implements a lightweight structural-awareness module using cross-attention
between drug and protein embeddings. Optional pocket masking guides
attention to binding-relevant protein regions.

Architecture:
  Query  = drug_embedding (B, 1, D)
  Key    = protein_sequence_embedding (B, L, D)
  Value  = protein_sequence_embedding (B, L, D)
  Output = context-aware drug representation (B, D)

This module is plug-and-play: enabled only during fine-tuning (not pretraining).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Pocket-Guided Cross-Attention Module
# ─────────────────────────────────────────────────────────────────────────────


class PocketGuidedAttention(nn.Module):
    """
    Cross-attention between drug and protein embeddings with optional
    pocket-guided masking for binding-site awareness.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must match drug and protein encoder output).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate for attention weights.
    max_seq_len : int
        Maximum protein sequence length (for efficiency, truncation).
    use_residual : bool
        If True, adds residual connection (drug_emb + attn_output).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1200,
        use_residual: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_residual = use_residual

        # Query projection (drug → Q)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        # Key projection (protein → K)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        # Value projection (protein → V)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Pocket bias: learnable bias added to attention logits when pocket mask provided
        self.pocket_bias = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        drug_embedding: torch.Tensor,
        protein_embedding: torch.Tensor,
        pocket_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention: drug queries attend to protein sequence.

        Parameters
        ----------
        drug_embedding : (B, D)
            Pooled drug representation.
        protein_embedding : (B, L, D)
            Sequence-level protein features. If (B, D), expanded to (B, 1, D).
        pocket_mask : (B, L) or None
            Binary mask indicating likely binding residues (1 = pocket, 0 = non-pocket).
            If None, full attention is used (no masking).

        Returns
        -------
        enhanced_drug : (B, D)
            Context-aware drug representation.
        attn_weights : (B, num_heads, L)
            Attention weights over protein positions (for interpretability).
        """
        B = drug_embedding.size(0)

        # Ensure protein has sequence dimension
        if protein_embedding.dim() == 2:
            protein_embedding = protein_embedding.unsqueeze(1)  # (B, 1, D)

        # Truncate long sequences for efficiency
        L = min(protein_embedding.size(1), self.max_seq_len)
        protein_embedding = protein_embedding[:, :L, :]
        if pocket_mask is not None:
            pocket_mask = pocket_mask[:, :L]

        # Drug as query (B, 1, D)
        drug_query = drug_embedding.unsqueeze(1)

        # Multi-head projections
        Q = self.W_q(drug_query)   # (B, 1, D)
        K = self.W_k(protein_embedding)  # (B, L, D)
        V = self.W_v(protein_embedding)  # (B, L, D)

        # Reshape to (B, num_heads, seq_len, d_k)
        Q = Q.view(B, 1, self.num_heads, self.d_k).transpose(1, 2)        # (B, H, 1, d_k)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)        # (B, H, L, d_k)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)        # (B, H, L, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, 1, L)

        # Pocket guidance: add learnable bias to pocket regions
        if pocket_mask is not None:
            # pocket_mask: (B, L) → (B, 1, 1, L) for broadcasting
            pocket_bias = pocket_mask.unsqueeze(1).unsqueeze(1).float() * self.pocket_bias
            scores = scores + pocket_bias

        # Softmax + dropout
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, 1, L)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, V)  # (B, H, 1, d_k)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)  # (B, 1, D)
        context = self.W_o(context).squeeze(1)  # (B, D)

        # Residual connection
        if self.use_residual:
            enhanced_drug = self.layer_norm(drug_embedding + context)
        else:
            enhanced_drug = self.layer_norm(context)

        # Return attention weights for interpretability: (B, H, L)
        attn_out = attn_weights.squeeze(2)  # (B, H, L)

        return enhanced_drug, attn_out

    def get_attention_entropy(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute average attention entropy across heads for interpretability.

        Parameters
        ----------
        attn_weights : (B, H, L) attention weights.

        Returns
        -------
        entropy : (B,) scalar entropy per sample.
        """
        # Avoid log(0) with clamping
        attn_clamped = attn_weights.clamp(min=1e-10)
        entropy = -(attn_clamped * torch.log(attn_clamped)).sum(dim=-1)  # (B, H)
        return entropy.mean(dim=-1)  # (B,) average across heads

    def get_pocket_attention_ratio(
        self, attn_weights: torch.Tensor, pocket_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fraction of attention on pocket vs non-pocket regions.

        Parameters
        ----------
        attn_weights : (B, H, L)
        pocket_mask : (B, L)

        Returns
        -------
        ratio : (B,) fraction of attention on pocket residues.
        """
        # Average across heads: (B, L)
        avg_attn = attn_weights.mean(dim=1)
        L = avg_attn.size(1)
        mask = pocket_mask[:, :L].float()
        pocket_attn = (avg_attn * mask).sum(dim=-1)  # (B,)
        return pocket_attn


# ─────────────────────────────────────────────────────────────────────────────
# Protein Sequence Feature Extractor (for providing sequence-level embeddings)
# ─────────────────────────────────────────────────────────────────────────────


class ProteinSequenceFeatures(nn.Module):
    """
    Extract sequence-level protein features (B, L, D) from character-level
    token embeddings, suitable as input to PocketGuidedAttention.

    Uses the same Conv1D architecture as the protein encoder but returns
    intermediate features instead of pooled output.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        kernels: tuple = (4, 8, 12),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, conv_out, k, padding=k // 2)
            for k in kernels
        ])
        self.out_dim = conv_out * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) LongTensor of protein token ids.
        Returns: (B, L', out_dim) sequence-level features.
        """
        emb = self.embedding(x).permute(0, 2, 1)  # (B, emb_dim, L)
        conv_feats = [F.relu(c(emb)) for c in self.convs]
        # Concat along channel axis → (B, out_dim, L')
        cat = torch.cat(conv_feats, dim=1)
        # Transpose to (B, L', out_dim) for attention
        return cat.permute(0, 2, 1)
