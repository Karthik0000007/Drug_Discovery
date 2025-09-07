import torch
import torch.nn as nn

# ---------------------------
# Model: DeepDTA-like
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        # 1D conv expects (batch, channel, length)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x

class DeepDTAModel(nn.Module):
    def __init__(self, vocab_drug:int, vocab_prot:int,
                 emb_dim:int=128, conv_out=128, sml_kernels=(4,6,8),
                 prot_kernels=(4,8,12), dropout=0.2):
        super().__init__()
        self.embed_drug = nn.Embedding(num_embeddings=vocab_drug, embedding_dim=emb_dim, padding_idx=0)
        self.embed_prot = nn.Embedding(num_embeddings=vocab_prot, embedding_dim=emb_dim, padding_idx=0)

        self.drug_convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in sml_kernels])
        self.prot_convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in prot_kernels])

        total = conv_out * (len(sml_kernels) + len(prot_kernels))
        self.fc = nn.Sequential(
            nn.Linear(total, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, smiles, seq):
        # smiles, seq: LongTensor (batch, L)
        d = self.embed_drug(smiles).permute(0,2,1)   # (B, emb, L)
        p = self.embed_prot(seq).permute(0,2,1)
        d_feats = [c(d) for c in self.drug_convs]   # list of (B, conv_out)
        p_feats = [c(p) for c in self.prot_convs]
        x = torch.cat(d_feats + p_feats, dim=1)
        out = self.fc(x).squeeze(-1)
        return out