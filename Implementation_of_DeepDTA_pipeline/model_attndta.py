"""
model_attndta.py — AttentionDTA baseline.

Architecture (He et al., IEEE/ACM TCBB 2020):
  Drug Branch:   Embedding → Conv1D×3 → Self-Attention (4 heads) → Attn-pool → ℝ^384
  Protein Branch: Same structure → ℝ^384
  FC Head:       concat → 1024 → 256 → 1

Parameters: ~1.8M  (attention weights added on top of DeepDTA)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Self-attention module
# ──────────────────────────────────────────────

class SelfAttention(nn.Module):
    """Multi-head self-attention with attention-weighted pooling."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        Returns: (B, D)  — attention-weighted pooled representation.
        """
        B, L, D = x.shape
        residual = x

        # Projections → (B, n_heads, L, d_k)
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (B, n_heads, L, d_k) → (B, L, D)
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        context = self.W_o(context)
        context = self.layer_norm(context + residual)

        # Attention-weighted pooling: compute importances and average
        # Use mean attention weight across heads per position
        weights = attn.mean(dim=1).mean(dim=1)  # (B, L)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        pooled = torch.bmm(weights.unsqueeze(1), context).squeeze(1)  # (B, D)
        return pooled


# ──────────────────────────────────────────────
# Encoder with attention
# ──────────────────────────────────────────────

class AttentionEncoder(nn.Module):
    """Embedding → Conv1D×3 → Self-Attention → pooled features."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        kernels: tuple[int, ...] = (4, 6, 8),
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, conv_out, k, padding=k // 2) for k in kernels
        ])
        self.conv_out_dim = conv_out * len(kernels)
        self.attention = SelfAttention(self.conv_out_dim, n_heads, dropout)
        self.out_dim = self.conv_out_dim  # 384 by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, out_dim)"""
        e = self.embedding(x).permute(0, 2, 1)  # (B, emb, L)
        conv_feats = [F.relu(c(e)) for c in self.convs]
        # Concat along channel axis, then transpose to (B, L, C)
        cat = torch.cat(conv_feats, dim=1).permute(0, 2, 1)  # (B, L', C)
        pooled = self.attention(cat)  # (B, C)
        return pooled


# ──────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────

class AttentionDTAModel(nn.Module):
    """
    AttentionDTA: CNN + Self-Attention encoder for DTA prediction.

    Parameters
    ----------
    vocab_drug, vocab_prot : vocabulary sizes.
    emb_dim : embedding dimension (default 128).
    conv_out : channels per conv filter (default 128).
    sml_kernels, prot_kernels : kernel sizes for drug / protein branches.
    n_heads : number of attention heads (default 4).
    dropout : dropout rate.
    """

    def __init__(
        self,
        vocab_drug: int,
        vocab_prot: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        sml_kernels: tuple[int, ...] = (4, 6, 8),
        prot_kernels: tuple[int, ...] = (4, 8, 12),
        n_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_encoder = AttentionEncoder(
            vocab_drug, emb_dim, conv_out, sml_kernels, n_heads, dropout * 0.5
        )
        self.prot_encoder = AttentionEncoder(
            vocab_prot, emb_dim, conv_out, prot_kernels, n_heads, dropout * 0.5
        )

        total = self.drug_encoder.out_dim + self.prot_encoder.out_dim
        self.fc = nn.Sequential(
            nn.Linear(total, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, smiles: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        d = self.drug_encoder(smiles)
        p = self.prot_encoder(seq)
        x = torch.cat([d, p], dim=1)
        return self.fc(x).squeeze(-1)

    def parameter_count(self) -> dict:
        def count(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "drug_encoder": count(self.drug_encoder),
            "prot_encoder": count(self.prot_encoder),
            "fc_head": count(self.fc),
            "total": count(self),
        }
