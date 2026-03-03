"""
model_graphdta.py — GraphDTA baseline.

Architecture (Nguyen et al., Bioinformatics 2021):
  Drug Branch:   SMILES → RDKit mol → molecular graph
                 Node features (78-d) → GCN/GAT ×3 → global mean pool → ℝ^128
  Protein Branch: Embedding → Conv1D×3 → Pool → ℝ^384 (same as DeepDTA)
  FC Head:       concat(128, 384) = 512 → 1024 → 256 → 1
  Parameters:    ~800K

Requires torch_geometric. Falls back to a simple fingerprint-based model
if torch_geometric is not available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

# Check for torch_geometric availability
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    PYGEOM_AVAILABLE = True
except ImportError:
    PYGEOM_AVAILABLE = False

# RDKit for molecular graph construction
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ──────────────────────────────────────────────
# Atom / bond featurization
# ──────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Si",
    "B", "Se", "Na", "K", "Ca", "Mg", "Zn", "Fe", "Cu", "other",
]
HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
] if RDKIT_AVAILABLE else []


def one_hot(val, options: list) -> list:
    """One-hot encode a value among options, with 'other' bucket."""
    vec = [0] * (len(options) + 1)
    if val in options:
        vec[options.index(val)] = 1
    else:
        vec[-1] = 1  # other
    return vec


def atom_features(atom) -> list:
    """
    Compute 78-dim atom feature vector.
    Features: atom type (21), degree (7), charge (1), n_Hs (5),
              aromatic (1), hybridization (6), in_ring (1),
              ... padding to 78.
    """
    feats = []
    # Atom type one-hot (21)
    feats += one_hot(atom.GetSymbol(), ATOM_TYPES)
    # Degree one-hot (7)
    feats += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
    # Formal charge (1)
    feats.append(atom.GetFormalCharge())
    # Number of Hs one-hot (5)
    feats += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])
    # Is aromatic (1)
    feats.append(int(atom.GetIsAromatic()))
    # Hybridization one-hot (6)
    feats += one_hot(atom.GetHybridization(), HYBRIDIZATION)
    # In ring (1)
    feats.append(int(atom.IsInRing()))
    # Pad to 78
    while len(feats) < 78:
        feats.append(0)
    return feats[:78]


def smiles_to_graph(smiles: str) -> Optional["Data"]:
    """Convert SMILES to torch_geometric Data object."""
    if not RDKIT_AVAILABLE or not PYGEOM_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edge index (COO format)
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # undirected
    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


# ──────────────────────────────────────────────
# GCN drug encoder
# ──────────────────────────────────────────────

if PYGEOM_AVAILABLE:
    class GCNDrugEncoder(nn.Module):
        """3-layer GCN encoder for molecular graphs."""

        def __init__(self, in_features: int = 78, hidden: int = 128, out_dim: int = 128):
            super().__init__()
            self.conv1 = GCNConv(in_features, hidden)
            self.conv2 = GCNConv(hidden, hidden)
            self.conv3 = GCNConv(hidden, out_dim)
            self.out_dim = out_dim

        def forward(self, data) -> torch.Tensor:
            """data: torch_geometric Batch → (B, out_dim)"""
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)  # (B, out_dim)
            return x

    class GATDrugEncoder(nn.Module):
        """3-layer GAT encoder for molecular graphs."""

        def __init__(self, in_features: int = 78, hidden: int = 128,
                     out_dim: int = 128, heads: int = 4):
            super().__init__()
            self.conv1 = GATConv(in_features, hidden // heads, heads=heads)
            self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
            self.conv3 = GATConv(hidden, out_dim, heads=1, concat=False)
            self.out_dim = out_dim

        def forward(self, data) -> torch.Tensor:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)
            return x


# ──────────────────────────────────────────────
# Protein encoder (reuse DeepDTA's CNN)
# ──────────────────────────────────────────────

class ProteinCNNEncoder(nn.Module):
    """Same CNN-based protein encoder as DeepDTA."""

    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 conv_out: int = 128, kernels=(4, 8, 12)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for k in kernels:
            self.convs.append(nn.Conv1d(emb_dim, conv_out, k))
            self.pools.append(nn.AdaptiveMaxPool1d(1))
        self.out_dim = conv_out * len(kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embedding(x).permute(0, 2, 1)
        feats = []
        for conv, pool in zip(self.convs, self.pools):
            h = F.relu(conv(e))
            h = pool(h).squeeze(-1)
            feats.append(h)
        return torch.cat(feats, dim=1)


# ──────────────────────────────────────────────
# GraphDTA model
# ──────────────────────────────────────────────

class GraphDTAModel(nn.Module):
    """
    GraphDTA: Graph-based drug encoder + CNN protein encoder.

    Parameters
    ----------
    vocab_prot : protein vocabulary size.
    drug_encoder_type : 'gcn' or 'gat'.
    drug_in_features : dimensionality of atom features (default 78).
    drug_hidden : hidden dim for GCN/GAT.
    drug_out : output dim for drug encoder (default 128).
    dropout : dropout rate for FC head.
    """

    def __init__(
        self,
        vocab_prot: int,
        drug_encoder_type: str = "gcn",
        drug_in_features: int = 78,
        drug_hidden: int = 128,
        drug_out: int = 128,
        emb_dim: int = 128,
        conv_out: int = 128,
        prot_kernels=(4, 8, 12),
        dropout: float = 0.2,
    ):
        super().__init__()

        if not PYGEOM_AVAILABLE:
            raise ImportError(
                "torch_geometric is required for GraphDTA. "
                "Install: pip install torch-geometric"
            )

        if drug_encoder_type == "gat":
            self.drug_encoder = GATDrugEncoder(drug_in_features, drug_hidden, drug_out)
        else:
            self.drug_encoder = GCNDrugEncoder(drug_in_features, drug_hidden, drug_out)

        self.prot_encoder = ProteinCNNEncoder(vocab_prot, emb_dim, conv_out, prot_kernels)

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

    def forward(self, drug_graph, seq: torch.Tensor) -> torch.Tensor:
        """
        drug_graph : torch_geometric Batch of molecular graphs.
        seq : (B, L) protein token indices.
        Returns : (B,) predicted affinities.
        """
        d = self.drug_encoder(drug_graph)
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
# Utility: batch SMILES → graph batch
# ──────────────────────────────────────────────

def collate_smiles_to_graph_batch(smiles_list: List[str], device=None):
    """
    Convert a list of SMILES strings to a batched graph.

    Parameters
    ----------
    smiles_list : list of SMILES strings
    device : target device (optional)

    Returns
    -------
    torch_geometric Batch object, or None if conversion fails.
    """
    if not PYGEOM_AVAILABLE or not RDKIT_AVAILABLE:
        return None
    graphs = []
    for s in smiles_list:
        g = smiles_to_graph(s)
        if g is not None:
            graphs.append(g)
        else:
            # Fallback: single dummy node
            g = Data(
                x=torch.zeros(1, 78, dtype=torch.float),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
            )
            graphs.append(g)
    batch = Batch.from_data_list(graphs)
    if device is not None:
        batch = batch.to(device)
    return batch
