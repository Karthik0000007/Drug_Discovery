"""
model_widedta.py — WideDTA baseline.

Architecture (Ozturk et al., Bioinformatics 2019):
  Drug Branch:   SMILES → character-level words → Embedding → Conv1D×3 → Pool → ℝ^384
  Protein Branch: Sequence → character-level words → Embedding → Conv1D×3 → Pool → ℝ^384

  In the original paper, LMCS (Ligand Max Common Substructure) words and
  Pfam/PROSITE motif words are used. Here we approximate with n-gram tokenization
  on character-level SMILES and protein sequences to keep the pipeline self-contained,
  while preserving the architectural difference (word-level vs character-level).

  FC Head: concat(384, 384) → 1024 → 256 → 1
  Parameters: ~1.5M (larger vocabularies from n-gram wordpieces)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────
# N-gram word extraction
# ──────────────────────────────────────────────

def extract_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from a string."""
    if len(text) < n:
        return [text]
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def build_ngram_vocab(
    texts: List[str],
    n: int = 3,
    max_vocab: int = 8000,
    min_freq: int = 2,
) -> Dict[str, int]:
    """
    Build n-gram vocabulary from a list of strings.

    Returns
    -------
    stoi : dict mapping n-gram → index (0 = PAD, 1 = UNK)
    """
    freq: Counter = Counter()
    for t in texts:
        freq.update(extract_ngrams(t, n))
    # Filter and sort
    kept = [(w, c) for w, c in freq.most_common() if c >= min_freq]
    kept = kept[:max_vocab - 2]  # reserve PAD, UNK
    stoi = {"<PAD>": 0, "<UNK>": 1}
    for i, (w, _) in enumerate(kept):
        stoi[w] = i + 2
    return stoi


def tokenize_ngrams(
    text: str,
    stoi: Dict[str, int],
    n: int = 3,
    max_words: int = 100,
) -> List[int]:
    """Tokenize text into n-gram indices with padding/truncation."""
    grams = extract_ngrams(text, n)
    indices = [stoi.get(g, 1) for g in grams]  # UNK = 1
    if len(indices) > max_words:
        indices = indices[:max_words]
    else:
        indices += [0] * (max_words - len(indices))  # PAD = 0
    return indices


# ──────────────────────────────────────────────
# Word-level CNN encoder
# ──────────────────────────────────────────────

class WordEncoder(nn.Module):
    """Word/n-gram embedding → Conv1D×3 → AdaptiveMaxPool → concat."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        kernels: Tuple[int, ...] = (4, 6, 8),
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, conv_out, k) for k in kernels
        ])
        self.pools = nn.ModuleList([
            nn.AdaptiveMaxPool1d(1) for _ in kernels
        ])
        self.out_dim = conv_out * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L_words) → (B, out_dim)"""
        e = self.embedding(x).permute(0, 2, 1)  # (B, emb, L)
        feats = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(e))
            h = pool(h).squeeze(-1)
            feats.append(h)
        return torch.cat(feats, dim=1)


# ──────────────────────────────────────────────
# WideDTA model
# ──────────────────────────────────────────────

class WideDTAModel(nn.Module):
    """
    WideDTA: Domain-specific word (n-gram) tokenization + CNN encoder.

    Parameters
    ----------
    vocab_drug : drug n-gram vocabulary size.
    vocab_prot : protein n-gram vocabulary size.
    emb_dim : embedding dimension.
    conv_out : output channels per conv.
    dropout : dropout probability.
    """

    def __init__(
        self,
        vocab_drug: int,
        vocab_prot: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        sml_kernels: Tuple[int, ...] = (4, 6, 8),
        prot_kernels: Tuple[int, ...] = (4, 8, 12),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_encoder = WordEncoder(vocab_drug, emb_dim, conv_out, sml_kernels)
        self.prot_encoder = WordEncoder(vocab_prot, emb_dim, conv_out, prot_kernels)

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


# ──────────────────────────────────────────────
# Helper: build WideDTA from raw data
# ──────────────────────────────────────────────

def build_widedta_from_data(
    train_smiles: List[str],
    train_sequences: List[str],
    drug_ngram: int = 3,
    prot_ngram: int = 3,
    max_drug_vocab: int = 8000,
    max_prot_vocab: int = 8000,
    max_drug_words: int = 100,
    max_prot_words: int = 1000,
    **model_kwargs,
) -> Tuple[WideDTAModel, Dict[str, int], Dict[str, int], int, int, int, int]:
    """
    Convenience: build vocabs and model from training data.

    Returns
    -------
    (model, drug_stoi, prot_stoi, drug_ngram, prot_ngram,
     max_drug_words, max_prot_words)
    """
    drug_stoi = build_ngram_vocab(train_smiles, drug_ngram, max_drug_vocab)
    prot_stoi = build_ngram_vocab(train_sequences, prot_ngram, max_prot_vocab)
    model = WideDTAModel(len(drug_stoi), len(prot_stoi), **model_kwargs)
    return (model, drug_stoi, prot_stoi,
            drug_ngram, prot_ngram, max_drug_words, max_prot_words)
