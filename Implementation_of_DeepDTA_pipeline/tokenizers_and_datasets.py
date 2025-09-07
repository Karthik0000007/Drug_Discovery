from typing import List, Dict
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------
# Tokenizers & Dataset
# ---------------------------
def build_vocab(sequences: List[str], min_freq=1):
    # char-level vocab: index 0 reserved for PAD.
    freq = {}
    for s in sequences:
        for ch in s:
            freq[ch] = freq.get(ch, 0) + 1
    items = [ch for ch,cnt in freq.items() if cnt >= min_freq]
    items = sorted(items)
    # mapping: PAD=0, UNK=1, then chars
    itos = ['<PAD>', '<UNK>'] + items
    stoi = {ch:i for i,ch in enumerate(itos)}
    return stoi, itos

def tokenize_seq(s: str, stoi: Dict[str,int], max_len: int):
    ids = []
    for ch in s:
        ids.append(stoi.get(ch, stoi.get('<UNK>',1)))
    if len(ids) >= max_len:
        return ids[:max_len]
    else:
        ids = ids + [0] * (max_len - len(ids))
        return ids

class DtaDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 sml_stoi: Dict[str,int], prot_stoi: Dict[str,int],
                 max_sml_len:int=120, max_prot_len:int=1000):
        self.df = df.reset_index(drop=True)
        self.sml_stoi = sml_stoi
        self.prot_stoi = prot_stoi
        self.max_sml_len = max_sml_len
        self.max_prot_len = max_prot_len

        # required columns expected: 'smiles', 'sequence', 'affinity'
        assert 'smiles' in df.columns and 'sequence' in df.columns and 'affinity' in df.columns

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        s = row['smiles']
        p = row['sequence']
        a = float(row['affinity'])
        s_ids = tokenize_seq(s, self.sml_stoi, self.max_sml_len)
        p_ids = tokenize_seq(p, self.prot_stoi, self.max_prot_len)
        return {
            'smiles': torch.LongTensor(s_ids),
            'seq': torch.LongTensor(p_ids),
            'aff': torch.FloatTensor([a])
        }