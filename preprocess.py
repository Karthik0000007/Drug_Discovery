import json
import pickle
import numpy as np
import pandas as pd
import os

def convert_deepdta_format(dataset: str, data_dir: str, out_path: str, log_transform: bool = True):
    """Convert DeepDTA JSON + matrix format to flat CSV."""
    with open(os.path.join(data_dir, 'ligands_can.txt')) as f:
        drugs = json.load(f)          # {drug_name: smiles}
    with open(os.path.join(data_dir, 'proteins.txt')) as f:
        proteins = json.load(f)       # {target_name: sequence}

    # Y is a Python-2-pickled numpy array; requires latin1 encoding in Python 3
    with open(os.path.join(data_dir, 'Y'), 'rb') as f:
        Y = pickle.load(f, encoding='latin1')
    Y = np.array(Y, dtype=float)

    drug_names = list(drugs.keys())
    prot_names = list(proteins.keys())

    rows = []
    for i, d in enumerate(drug_names):
        for j, p in enumerate(prot_names):
            aff = Y[i, j]
            if np.isnan(aff):
                continue
            if log_transform and dataset == 'davis':
                # Convert Kd (nM) → pKd
                aff = -np.log10(aff / 1e9)
            rows.append({
                'drug_id':  d,
                'target_id': p,
                'smiles':   drugs[d],
                'sequence': proteins[p],
                'affinity': aff
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} pairs to {out_path}")

convert_deepdta_format('davis', 'data/davis/', 'data/davis_processed.csv')
convert_deepdta_format('kiba',  'data/kiba/',  'data/kiba_processed.csv', log_transform=False)