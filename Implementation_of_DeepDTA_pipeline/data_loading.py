import numpy as np
import pandas as pd

# ---------------------------
# Data loading / splitting
# ---------------------------
def prepare_data(df: pd.DataFrame,
                 split: str = 'random',
                 test_frac: float = 0.1,
                 val_frac: float = 0.1,
                 seed: int = 42):
    """
    split: 'random', 'cold_drug', 'cold_target'
    For cold splits columns drug_id and target_id must be present.
    """
    assert split in ('random','cold_drug','cold_target')
    np.random.seed(seed)
    if split == 'random':
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        n_test = int(len(idx) * test_frac)
        n_val = int(len(idx) * val_frac)
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test+n_val]
        train_idx = idx[n_test+n_val:]
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    else:
        if split == 'cold_drug':
            key = 'drug_id'
        else:
            key = 'target_id'
        if key not in df.columns:
            raise ValueError(f"To perform {split} split provide column '{key}' in CSV.")
        groups = df[key].unique().tolist()
        np.random.shuffle(groups)
        n_test = max(1, int(len(groups) * test_frac))
        n_val = max(1, int(len(groups) * val_frac))
        test_groups = set(groups[:n_test])
        val_groups = set(groups[n_test:n_test+n_val])
        train_mask = ~df[key].isin(test_groups) & ~df[key].isin(val_groups)
        val_mask = df[key].isin(val_groups)
        test_mask = df[key].isin(test_groups)
        return df[train_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)