"""
Extract embeddings from trained model for visualization.

Usage:
    python extract_embeddings.py \
        --checkpoint checkpoints/meta_learning/meta_learned_model.pt \
        --dataset davis \
        --split test \
        --output embeddings/
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from Implementation_of_DeepDTA_pipeline.model import DeepDTAModel
from Implementation_of_DeepDTA_pipeline.data_loading import load_davis, load_kiba
from Implementation_of_DeepDTA_pipeline.tokenizers_and_datasets import DTADataset


def extract_embeddings(model, dataloader, device='cuda'):
    """
    Extract drug and protein embeddings from trained model.
    
    Returns
    -------
    drug_embeddings : (N, D_drug) array
    prot_embeddings : (N, D_prot) array
    affinities : (N,) array
    smiles_list : list of SMILES strings
    seq_list : list of protein sequences
    """
    model.eval()
    model.to(device)
    
    drug_embs = []
    prot_embs = []
    affinities = []
    smiles_list = []
    seq_list = []
    
    print("[extract] Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}")
            
            smiles = batch['smiles'].to(device)
            seq = batch['seq'].to(device)
            aff = batch['aff'].cpu().numpy()
            
            # Extract encoder outputs
            d_emb = model.drug_encoder(smiles)
            p_emb = model.prot_encoder(seq)
            
            drug_embs.append(d_emb.cpu().numpy())
            prot_embs.append(p_emb.cpu().numpy())
            affinities.append(aff)
            
            # Store raw sequences if available
            if 'smiles_raw' in batch:
                smiles_list.extend(batch['smiles_raw'])
            if 'seq_raw' in batch:
                seq_list.extend(batch['seq_raw'])
    
    drug_embeddings = np.vstack(drug_embs)
    prot_embeddings = np.vstack(prot_embs)
    affinities = np.concatenate(affinities)
    
    print(f"[extract] Extracted {len(affinities)} samples")
    print(f"  Drug embeddings: {drug_embeddings.shape}")
    print(f"  Protein embeddings: {prot_embeddings.shape}")
    
    return drug_embeddings, prot_embeddings, affinities, smiles_list, seq_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='davis',
                       choices=['davis', 'kiba'],
                       help='Dataset to extract from')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to use')
    parser.add_argument('--output', type=str, default='embeddings/',
                       help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to extract (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load checkpoint
    print(f"[extract] Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use defaults
        from Implementation_of_DeepDTA_pipeline.config import ExperimentConfig
        config = ExperimentConfig()
    
    # Load dataset
    print(f"[extract] Loading {args.dataset} dataset ({args.split} split)")
    if args.dataset == 'davis':
        df = load_davis()
    else:
        df = load_kiba()
    
    # Simple train/val/test split
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    if args.split == 'train':
        df = df.iloc[:train_end]
    elif args.split == 'val':
        df = df.iloc[train_end:val_end]
    else:
        df = df.iloc[val_end:]
    
    if args.max_samples:
        df = df.iloc[:args.max_samples]
    
    print(f"  Loaded {len(df)} samples")
    
    # Create dataset and dataloader
    dataset = DTADataset(df, max_sml_len=120, max_prot_len=1000)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    vocab_drug = 65
    vocab_prot = 26
    
    model = DeepDTAModel(
        vocab_drug=vocab_drug,
        vocab_prot=vocab_prot,
        emb_dim=128,
        conv_out=128,
    )
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("[extract] Model loaded successfully")
    
    # Extract embeddings
    drug_embs, prot_embs, affinities, smiles_list, seq_list = extract_embeddings(
        model, dataloader, device=args.device
    )
    
    # Save embeddings
    output_prefix = f"{args.dataset}_{args.split}"
    
    np.save(os.path.join(args.output, f'{output_prefix}_drug_embeddings.npy'), drug_embs)
    np.save(os.path.join(args.output, f'{output_prefix}_prot_embeddings.npy'), prot_embs)
    np.save(os.path.join(args.output, f'{output_prefix}_affinities.npy'), affinities)
    
    if smiles_list:
        with open(os.path.join(args.output, f'{output_prefix}_smiles.txt'), 'w') as f:
            f.write('\n'.join(smiles_list))
    
    if seq_list:
        with open(os.path.join(args.output, f'{output_prefix}_sequences.txt'), 'w') as f:
            f.write('\n'.join(seq_list))
    
    print(f"\n[extract] Embeddings saved to {args.output}/")
    print(f"  - {output_prefix}_drug_embeddings.npy")
    print(f"  - {output_prefix}_prot_embeddings.npy")
    print(f"  - {output_prefix}_affinities.npy")
    
    # Compute mutual information
    from Implementation_of_DeepDTA_pipeline.visualization import compute_mutual_information
    
    print("\n[extract] Computing mutual information...")
    mi = compute_mutual_information(drug_embs, prot_embs, n_bins=20)
    print(f"  Mutual Information: {mi:.4f} bits")
    
    # Save MI
    with open(os.path.join(args.output, f'{output_prefix}_mi.txt'), 'w') as f:
        f.write(f"Mutual Information: {mi:.4f} bits\n")
    
    print("\n✅ Extraction complete!")


if __name__ == '__main__':
    main()
