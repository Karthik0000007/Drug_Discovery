"""
example_phase3_usage.py — Example usage of Phase 3 large-scale dataset features.

Demonstrates:
1. Loading and preprocessing BindingDB
2. Loading Pharos dark proteins
3. Creating cold_pharos split
4. Using LazyDtaDataset for memory-efficient training
5. Computing and logging dataset statistics
"""

import logging
import pandas as pd
from pathlib import Path

from large_scale_datasets import (
    load_bindingdb,
    load_pharos,
    BindingDBConfig,
    LazyDtaDataset,
    log_dataset_stats,
)
from data_loading import prepare_data
from tokenizers_and_datasets import build_vocab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_bindingdb_loading():
    """Example: Load and preprocess BindingDB dataset."""
    print("\n" + "="*80)
    print("Example 1: Loading BindingDB")
    print("="*80)
    
    # Configure preprocessing
    config = BindingDBConfig(
        min_affinity=3.0,           # Minimum pKd (weaker than 1 mM)
        max_affinity=12.0,          # Maximum pKd (stronger than 1 pM)
        min_sequence_length=50,
        max_sequence_length=5000,
        min_smiles_length=5,
        max_smiles_length=200,
        allowed_affinity_types=["Kd", "Ki", "IC50"],
        remove_duplicates=True,
        keep_highest_quality=True,
    )
    
    # Load BindingDB (with caching)
    bindingdb_df = load_bindingdb(
        path="data/BindingDB_All.tsv",
        config=config,
        cache_path="data/bindingdb_processed.parquet",
        force_reprocess=False,  # Use cache if available
    )
    
    # Log statistics
    log_dataset_stats(bindingdb_df, name="BindingDB")
    
    return bindingdb_df


def example_pharos_loading():
    """Example: Load Pharos dark protein dataset."""
    print("\n" + "="*80)
    print("Example 2: Loading Pharos Dark Proteins")
    print("="*80)
    
    # Load Pharos dataset
    pharos_df = load_pharos(
        path="data/pharos_interactions.tsv",
        interaction_threshold=10,  # Proteins with < 10 interactions
        cache_path="data/pharos_processed.parquet",
        force_reprocess=False,
    )
    
    # Log statistics
    log_dataset_stats(pharos_df, name="Pharos Dark Proteins")
    
    return pharos_df


def example_cold_pharos_split():
    """Example: Create cold_pharos split for zero-shot evaluation."""
    print("\n" + "="*80)
    print("Example 3: Cold-Pharos Split")
    print("="*80)
    
    # Load datasets
    # For this example, we'll use DAVIS + simulated Pharos proteins
    davis_df = pd.read_csv("data/davis_processed.csv")
    
    # Simulate dark proteins (in practice, load from Pharos)
    # Select proteins with fewest interactions as "dark"
    protein_counts = davis_df["target_id"].value_counts()
    dark_proteins = set(protein_counts[protein_counts < 50].index)
    
    logger.info(f"Identified {len(dark_proteins)} simulated dark proteins")
    
    # Create cold_pharos split
    train_df, val_df, test_df, split_info = prepare_data(
        davis_df,
        split="cold_pharos",
        val_frac=0.1,
        seed=42,
        pharos_proteins=dark_proteins,
    )
    
    print("\n" + split_info.summary())
    
    # Verify zero protein overlap
    train_proteins = set(train_df["target_id"].unique())
    test_proteins = set(test_df["target_id"].unique())
    overlap = train_proteins & test_proteins
    
    print(f"\nProtein overlap check: {len(overlap)} proteins")
    assert len(overlap) == 0, "Protein leakage detected!"
    print("✓ Zero protein leakage verified")
    
    return train_df, val_df, test_df, split_info


def example_lazy_dataset():
    """Example: Use LazyDtaDataset for memory-efficient training."""
    print("\n" + "="*80)
    print("Example 4: Memory-Efficient Lazy Loading")
    print("="*80)
    
    # First, save a dataset as parquet (simulated large dataset)
    davis_df = pd.read_csv("data/davis_processed.csv")
    parquet_path = "data/davis_lazy.parquet"
    davis_df.to_parquet(parquet_path, index=False)
    
    # Build vocabularies
    sml_stoi, sml_itos = build_vocab(davis_df["smiles"].unique().tolist())
    prot_stoi, prot_itos = build_vocab(davis_df["sequence"].unique().tolist())
    
    # Create lazy dataset
    lazy_dataset = LazyDtaDataset(
        parquet_path=parquet_path,
        sml_stoi=sml_stoi,
        prot_stoi=prot_stoi,
        max_sml_len=120,
        max_prot_len=1000,
        cache_tokenized=True,
        cache_dir="data/tokenized_cache/",
    )
    
    print(f"Lazy dataset length: {len(lazy_dataset)}")
    
    # Access a sample (loads only that row from parquet)
    sample = lazy_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"SMILES shape: {sample['smiles'].shape}")
    print(f"Sequence shape: {sample['seq'].shape}")
    print(f"Affinity: {sample['aff'].item():.2f}")
    
    # Save tokenized cache for future runs
    lazy_dataset.save_cache()
    print("✓ Tokenized cache saved")
    
    return lazy_dataset


def example_combined_workflow():
    """Example: Complete workflow combining BindingDB pretraining + DAVIS fine-tuning."""
    print("\n" + "="*80)
    print("Example 5: Combined Workflow")
    print("="*80)
    
    # Step 1: Load large-scale BindingDB for pretraining
    print("\n[Step 1] Loading BindingDB for pretraining...")
    # bindingdb_df = load_bindingdb(...)  # Uncomment when BindingDB is available
    
    # Step 2: Pretrain on BindingDB (cross-modal contrastive learning)
    print("[Step 2] Pretraining on BindingDB...")
    print("  → Use pretrain.py with mode='cross_modal'")
    print("  → Save encoder checkpoints")
    
    # Step 3: Load DAVIS for fine-tuning
    print("\n[Step 3] Loading DAVIS for fine-tuning...")
    davis_df = pd.read_csv("data/davis_processed.csv")
    log_dataset_stats(davis_df, name="DAVIS")
    
    # Step 4: Create cold_both split
    print("\n[Step 4] Creating cold_both split...")
    train_df, val_df, test_df, split_info = prepare_data(
        davis_df,
        split="cold_both",
        test_frac=0.1,
        val_frac=0.1,
        seed=42,
    )
    print(split_info.summary())
    
    # Step 5: Fine-tune with pretrained encoders
    print("\n[Step 5] Fine-tuning with pretrained encoders...")
    print("  → Load pretrained drug/protein encoders")
    print("  → Fine-tune on DAVIS cold_both split")
    print("  → Evaluate on test set")
    
    # Step 6: Zero-shot evaluation on Pharos
    print("\n[Step 6] Zero-shot evaluation on Pharos dark proteins...")
    print("  → Load Pharos dataset")
    print("  → Evaluate without fine-tuning on dark proteins")
    print("  → Compare with fine-tuned performance")
    
    print("\n✓ Complete workflow demonstrated")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("Phase 3: Large-Scale Dataset Integration Examples")
    print("="*80)
    
    # Note: These examples assume data files exist
    # Uncomment and run individual examples as needed
    
    # Example 1: BindingDB loading
    # bindingdb_df = example_bindingdb_loading()
    
    # Example 2: Pharos loading
    # pharos_df = example_pharos_loading()
    
    # Example 3: Cold-Pharos split
    try:
        train_df, val_df, test_df, split_info = example_cold_pharos_split()
    except FileNotFoundError:
        print("DAVIS dataset not found. Skipping cold_pharos example.")
    
    # Example 4: Lazy dataset
    try:
        lazy_dataset = example_lazy_dataset()
    except FileNotFoundError:
        print("DAVIS dataset not found. Skipping lazy dataset example.")
    
    # Example 5: Combined workflow
    example_combined_workflow()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)


if __name__ == "__main__":
    main()
