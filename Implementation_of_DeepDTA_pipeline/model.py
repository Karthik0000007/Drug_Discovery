"""
model.py — DeepDTA architecture with contrastive pretraining support.

Architecture:
  Drug Branch:   Embedding → 3×Conv1D → AdaptiveMaxPool → concat → ℝ^(3×conv_out)
  Protein Branch: Embedding → 3×Conv1D → AdaptiveMaxPool → concat → ℝ^(3×conv_out)
  FC Head:       concat(drug, prot) → 1024 → 256 → 1

Adds ProjectionHead for contrastive pretraining and weight transfer utilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────

class ConvBlock(nn.Module):
    """1-D convolution → ReLU → AdaptiveMaxPool1d(1)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive pretraining.
    Maps encoder features → ℓ₂-normalised projection.
    Discarded after pretraining.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ──────────────────────────────────────────────
# Encoder wrappers (for pretraining checkpoint saving)
# ──────────────────────────────────────────────

class DrugEncoder(nn.Module):
    """Drug embedding + multi-kernel Conv1D encoder."""

    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 conv_out: int = 128, kernels=(4, 6, 8)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in kernels])
        self.out_dim = conv_out * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) → (B, out_dim)"""
        e = self.embedding(x).permute(0, 2, 1)  # (B, emb, L)
        feats = [c(e) for c in self.convs]
        return torch.cat(feats, dim=1)


class ProteinEncoder(nn.Module):
    """Protein embedding + multi-kernel Conv1D encoder."""

    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 conv_out: int = 128, kernels=(4, 8, 12)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in kernels])
        self.out_dim = conv_out * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x).permute(0, 2, 1)
        feats = [c(e) for c in self.convs]
        return torch.cat(feats, dim=1)


# ──────────────────────────────────────────────
# DeepDTA model
# ──────────────────────────────────────────────

class DeepDTAModel(nn.Module):
    """
    DeepDTA-like CNN model for drug–target affinity prediction.

    Parameters
    ----------
    vocab_drug, vocab_prot : vocabulary sizes (including special tokens).
    emb_dim : embedding dimension.
    conv_out : output channels per conv filter.
    sml_kernels, prot_kernels : kernel sizes for drug / protein branches.
    dropout : dropout probability in FC head.
    """

    def __init__(
        self,
        vocab_drug: int,
        vocab_prot: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        sml_kernels=(4, 6, 8),
        prot_kernels=(4, 8, 12),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_encoder = DrugEncoder(vocab_drug, emb_dim, conv_out, sml_kernels)
        self.prot_encoder = ProteinEncoder(vocab_prot, emb_dim, conv_out, prot_kernels)

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

    # ── Convenience properties (backward compat) ──

    @property
    def embed_drug(self):
        return self.drug_encoder.embedding

    @property
    def embed_prot(self):
        return self.prot_encoder.embedding

    @property
    def drug_convs(self):
        return self.drug_encoder.convs

    @property
    def prot_convs(self):
        return self.prot_encoder.convs

    # ── Forward ──

    def forward(self, smiles: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        smiles, seq : LongTensor (B, L)
        Returns : (B,) predicted affinity
        """
        d = self.drug_encoder(smiles)
        p = self.prot_encoder(seq)
        x = torch.cat([d, p], dim=1)
        return self.fc(x).squeeze(-1)

    # ── Pretrained weight transfer ──

    def load_pretrained_encoders(
        self,
        drug_ckpt: str | None = None,
        prot_ckpt: str | None = None,
    ) -> None:
        """
        Load pretrained weights into encoder branches.
        Reinitialises the FC head so it trains from scratch.
        """
        if drug_ckpt is not None:
            state = torch.load(drug_ckpt, map_location="cpu")
            self.drug_encoder.load_state_dict(state["encoder"], strict=False)
            print(f"[model] Loaded pretrained drug encoder from {drug_ckpt}")

        if prot_ckpt is not None:
            state = torch.load(prot_ckpt, map_location="cpu")
            self.prot_encoder.load_state_dict(state["encoder"], strict=False)
            print(f"[model] Loaded pretrained protein encoder from {prot_ckpt}")

        # Reinitialise FC head
        for layer in self.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def freeze_encoders(self) -> None:
        """Freeze drug & protein encoder weights (train FC head only)."""
        for param in self.drug_encoder.parameters():
            param.requires_grad = False
        for param in self.prot_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze all encoder weights."""
        for param in self.drug_encoder.parameters():
            param.requires_grad = True
        for param in self.prot_encoder.parameters():
            param.requires_grad = True

    def parameter_count(self) -> dict:
        """Return parameter counts by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "drug_encoder": count(self.drug_encoder),
            "prot_encoder": count(self.prot_encoder),
            "fc_head": count(self.fc),
            "total": count(self),
        }