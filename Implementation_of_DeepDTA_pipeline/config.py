"""
config.py — Dataclass-based hierarchical configuration for CL-DTA experiments.

Supports YAML loading for reproducible experiment configs. Replaces scattered
argparse defaults with a single, version-controlled configuration object.
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


# ──────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────

@dataclass
class DataConfig:
    """Data loading and splitting configuration."""
    dataset: str = "davis"                     # 'davis' | 'kiba'
    data_path: str = "data/"
    max_sml_len: int = 120
    max_prot_len: int = 1000
    split: str = "random"                      # 'random' | 'cold_drug' | 'cold_target' | 'cold_both' | 'cold_pharos'
    test_frac: float = 0.1
    val_frac: float = 0.1
    n_folds: int = 5                           # For k-fold entity-group CV on cold splits
    
    # Phase 2: Leakage verification and edge case handling
    verify_no_leakage: bool = True             # Programmatic leakage verification
    min_samples_threshold: int = 100           # Minimum samples per split
    max_retry_attempts: int = 5                # Retry split if constraints violated


@dataclass
class PretrainConfig:
    """Contrastive pretraining configuration."""
    enabled: bool = True
    mode: str = "cross_modal"                  # 'drug_only' | 'prot_only' | 'both_independent' | 'cross_modal'
    epochs: int = 100
    batch_size: int = 256
    lr: float = 5e-4
    temperature: float = 0.07
    loss: str = "nt_xent"                      # 'nt_xent' | 'infonce' | 'triplet'

    # Cross-modal alignment (Phase 1)
    use_cross_modal: bool = True               # Enable cross-modal alignment loss
    align_loss_weight: float = 0.5             # Weight for cross-modal alignment loss (0.1-1.0)

    # Phase 4: Pretrained Models (LLM Integration)
    use_pretrained_embeddings: bool = False    # Enable pretrained LLM embeddings
    pretrained_drug_model: Optional[str] = None  # 'chemberta' | 'molformer' | None
    pretrained_prot_model: Optional[str] = None  # 'esm2_t33' | 'esm2_t6' | 'protbert' | None
    freeze_pretrained: bool = True             # Freeze pretrained model weights
    unfreeze_last_k_layers: int = 0            # Number of final layers to unfreeze (0 = keep frozen)
    cache_llm_embeddings: bool = True          # Cache pretrained embeddings to avoid recomputation
    embedding_alignment_weight: float = 0.3    # Weight for LLM↔learned embedding alignment loss

    # Phase 4: Tokenization
    use_pretrained_tokenizers: bool = False    # Use Hugging Face tokenizers (auto-enabled if using pretrained models)

    drug_augmentations: List[str] = field(
        default_factory=lambda: ["smiles_enum", "atom_mask"]
    )
    prot_augmentations: List[str] = field(
        default_factory=lambda: ["subseq_crop", "residue_mask"]
    )
    mask_ratio: float = 0.15
    crop_min_ratio: float = 0.7
    sub_ratio: float = 0.10
    drop_prob: float = 0.1
    projection_dim: int = 64
    weight_decay: float = 1e-5
    checkpoint_dir: str = "checkpoints/pretrained/"


@dataclass
class TrainConfig:
    """Supervised fine-tuning configuration."""
    epochs: int = 30
    batch_size: int = 256                      # Increased for better GPU utilization
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 8
    dropout: float = 0.2
    emb_dim: int = 128
    conv_out: int = 128
    grad_clip: float = 5.0
    freeze_strategy: str = "full_finetune"     # 'frozen' | 'full_finetune' | 'gradual_unfreeze'
    unfreeze_after: int = 5                    # epochs for gradual unfreeze

    # Phase 6: Pocket-Guided Attention
    use_attention_module: bool = False         # Enable pocket-guided cross-attention
    attention_heads: int = 4                   # Number of attention heads
    attention_max_seq_len: int = 1200          # Max protein sequence length for attention

    # Phase 7: Evidential Regression / Uncertainty
    use_evidential: bool = False               # Enable evidential regression head
    evidential_reg_weight: float = 0.01        # Regularization weight for evidential loss


@dataclass
class MetaLearningConfig:
    """Phase 5: Meta-learning configuration for few-shot DTA."""
    enabled: bool = False                      # Enable meta-learning training

    # Inner loop (task-specific adaptation)
    num_inner_steps: int = 3                   # Number of gradient steps on support set
    inner_lr: float = 1e-3                     # Learning rate for inner loop adaptation

    # Outer loop (meta-update)
    meta_lr: float = 1e-4                      # Learning rate for meta-update
    meta_batch_size: int = 4                   # Number of tasks per meta-batch

    # Adaptation scope
    adaptation_scope: str = "head_only"        # 'head_only' | 'partial_encoder' | 'full'
    unfreeze_layers: int = 2                   # For partial_encoder: num layers to unfreeze

    # Task sampling
    task_type: str = "mixed"                   # 'cold_drug' | 'cold_target' | 'cold_both' | 'mixed'
    k_support: int = 5                         # Support set size (few-shot)
    k_query: int = 10                          # Query set size

    # Efficiency
    use_functional_model: bool = True          # Use higher library for efficient MAML
    cache_task_embeddings: bool = False        # Cache embeddings per task

    # Training
    meta_epochs: int = 10                      # Number of meta-training epochs
    checkpoint_dir: str = "checkpoints/meta_learning/"


@dataclass
class MultiTaskConfig:
    """Phase 9/10: Multi-task learning configuration."""
    enabled: bool = False                      # Enable multi-task learning
    num_moa_classes: int = 0                   # Number of MoA classes (0 = disabled)
    affinity_threshold: float = 7.0            # pKd threshold for binary interaction label
    use_dynamic_weighting: bool = False        # Uncertainty-based dynamic loss weighting
    loss_weights: Dict = field(default_factory=lambda: {
        "affinity": 1.0,
        "interaction": 1.0,
        "moa": 1.0,
    })


# ──────────────────────────────────────────────
# Top-level experiment config
# ──────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    multitask: MultiTaskConfig = field(default_factory=MultiTaskConfig)
    model: str = "cl_dta"                      # 'deepdta' | 'graphdta' | 'widedta' | 'attndta' | 'cl_dta'
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    device: str = "cuda"
    results_dir: str = "results/"
    tensorboard_dir: str = "runs/"
    checkpoint_dir: str = "checkpoints/"
    ic50_nanomolar: bool = False

    # Pretrained encoder paths (set after pretraining or loaded from config)
    pretrained_drug_ckpt: Optional[str] = None
    pretrained_prot_ckpt: Optional[str] = None


# ──────────────────────────────────────────────
# YAML I/O
# ──────────────────────────────────────────────

def _nested_update(base: dict, overrides: dict) -> dict:
    """Recursively update *base* with *overrides*."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _nested_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(yaml_path: str) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    data_cfg = DataConfig(**raw.get("data", {}))
    pretrain_cfg = PretrainConfig(**raw.get("pretrain", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    meta_learning_cfg = MetaLearningConfig(**raw.get("meta_learning", {}))
    multitask_cfg = MultiTaskConfig(**raw.get("multitask", {}))

    sub_keys = ("data", "pretrain", "train", "meta_learning", "multitask")
    top = {k: v for k, v in raw.items() if k not in sub_keys}
    # Map 'experiment' sub-key to top-level
    if "experiment" in raw:
        top.update(raw["experiment"])

    return ExperimentConfig(
        data=data_cfg,
        pretrain=pretrain_cfg,
        train=train_cfg,
        meta_learning=meta_learning_cfg,
        multitask=multitask_cfg,
        **{k: v for k, v in top.items() if k in ExperimentConfig.__dataclass_fields__},
    )


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Save an ExperimentConfig to a YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    """Convert config to a plain dict (for JSON logging)."""
    return asdict(cfg)
