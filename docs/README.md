# CL-DTA: Contrastive Self-Supervised Learning for Cold-Start Drug–Target Affinity Prediction

A research-grade framework for predicting drug–target binding affinity (DTA) with a focus on the **cold-start problem** — reliably predicting affinity for entirely unseen drugs and/or protein targets at test time.

CL-DTA introduces a **two-phase training paradigm** that combines contrastive self-supervised pretraining with supervised fine-tuning, producing molecular representations that generalize beyond the training distribution.

---

## Overview

Standard supervised DTA models suffer large performance drops under cold-start evaluation because their representations are optimized for training entities rather than transferable structural properties. CL-DTA addresses this with:

**Phase 1 — Contrastive Pretraining:** Drug SMILES encoders and protein sequence encoders are independently pretrained using augmentation-based contrastive learning (NT-Xent loss). Domain-specific augmentations force encoders to learn augmentation-invariant, structurally meaningful representations.

**Phase 2 — Supervised Fine-Tuning:** Pretrained encoder weights are loaded into the regression architecture and fine-tuned end-to-end with MSE loss on drug–target affinity labels.

The core hypothesis is that **contrastive pretraining produces representations that generalize better to unseen chemical/biological entities**, evaluated across four split protocols on two standard benchmarks.

---

## Project Structure

```
Drug_Discovery/
├── preprocess.py                          # Raw dataset parsing → standardized CSV
├── data/
│   ├── davis_processed.csv                # Parsed DAVIS dataset
│   ├── kiba_processed.csv                 # Parsed KIBA dataset
│   ├── davis/                             # Raw DAVIS files
│   └── kiba/                              # Raw KIBA files
├── configs/                               # YAML experiment configurations
│   ├── davis_random.yaml
│   ├── davis_cold_drug.yaml
│   ├── davis_cold_target.yaml
│   ├── davis_cold_both.yaml
│   ├── kiba_random.yaml
│   ├── kiba_cold_drug.yaml
│   ├── kiba_cold_target.yaml
│   └── kiba_cold_both.yaml
└── Implementation_of_DeepDTA_pipeline/
    ├── config.py                          # Dataclass-based hierarchical configuration
    ├── data_loading.py                    # Train/val/test splitting (4 protocols)
    ├── tokenizers_and_datasets.py         # Character-level tokenization, DtaDataset
    ├── augmentations.py                   # Domain-specific SMILES & sequence augmentations
    ├── contrastive_dataset.py             # Positive pair generation for contrastive learning
    ├── contrastive_losses.py              # NT-Xent, InfoNCE, Triplet loss implementations
    ├── pretrain.py                        # Contrastive pretraining loop
    ├── model.py                           # DeepDTA CNN architecture (primary model)
    ├── model_attndta.py                   # AttentionDTA (CNN + multi-head self-attention)
    ├── model_widedta.py                   # WideDTA (n-gram tokenization + CNN)
    ├── model_graphdta.py                  # GraphDTA (GCN/GAT + CNN)
    ├── train.py                           # Supervised training loop with early stopping
    ├── utilities.py                       # Metrics: MSE, RMSE, CI, Pearson, Spearman, rm²
    ├── visualization.py                   # Publication-quality plots
    ├── run_experiments.py                 # Batch experiment orchestrator (120 experiments)
    └── main.py                            # Primary CLI entry point
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.x
- pandas, NumPy, scikit-learn, SciPy, matplotlib, PyYAML

```bash
pip install -r requirements.txt
```

### Optional Dependencies

| Package | Purpose |
|---|---|
| `rdkit-pypi` | SMILES enumeration, molecular graph construction, validity checking |
| `tensorboard` | Training curve visualization |
| `torch-geometric` | GNN layers for GraphDTA baseline |
| `umap-learn` | UMAP embedding visualization |

The codebase runs without any optional dependency — features that require them degrade gracefully with informative warnings.

---

## Datasets

| Dataset | Pairs | Unique Drugs | Unique Targets | Affinity Type | Scale |
|---|---|---|---|---|---|
| DAVIS | ~30,056 | 68 | 442 | $K_d$ (nM) | Converted to $pK_d = -\log_{10}(K_d / 10^{-9})$ |
| KIBA | ~118,254 | 2,111 | 229 | KIBA score | Already log-scale |

### Preprocessing

```bash
python preprocess.py
```

Produces `data/davis_processed.csv` and `data/kiba_processed.csv` with columns: `drug_id`, `target_id`, `smiles`, `sequence`, `affinity`.

---

## Usage

### 1. Contrastive Pretraining

Pretrain drug and protein encoders independently using NT-Xent contrastive loss:

```bash
# Drug + protein encoders (both_independent mode)
python -m Implementation_of_DeepDTA_pipeline.pretrain \
    --data data/davis_processed.csv \
    --mode both_independent \
    --epochs 100 \
    --batch-size 256 \
    --lr 5e-4 \
    --temperature 0.07 \
    --checkpoint-dir checkpoints/davis/
```

Pretrained encoder checkpoints are saved to `checkpoints/davis/drug_encoder.pt` and `checkpoints/davis/prot_encoder.pt`.

### 2. Supervised Fine-Tuning

Train a DTA model, optionally loading pretrained encoders:

```bash
# CL-DTA with pretrained encoders, cold-drug split
python -m Implementation_of_DeepDTA_pipeline.main \
    --data data/davis_processed.csv \
    --model cl_dta \
    --split cold_drug \
    --pretrained-drug checkpoints/davis/drug_encoder.pt \
    --pretrained-prot checkpoints/davis/prot_encoder.pt \
    --freeze-strategy gradual_unfreeze \
    --epochs 30 \
    --output results/davis_cold_drug/

# DeepDTA baseline, random split
python -m Implementation_of_DeepDTA_pipeline.main \
    --data data/davis_processed.csv \
    --model deepdta \
    --split random \
    --epochs 30
```

### 3. Config-File-Based Runs

All hyperparameters can be specified via YAML config (see `configs/`):

```bash
python -m Implementation_of_DeepDTA_pipeline.main \
    --config configs/davis_cold_drug.yaml
```

### 4. Full Experiment Matrix

Run all 120 experiments (2 datasets × 4 splits × 5 models × 3 seeds):

```bash
python -m Implementation_of_DeepDTA_pipeline.run_experiments \
    --data-dir data/ \
    --output-dir results/ \
    --seeds 42 123 456
```

---

## Model Architectures

### DeepDTA (Primary)

Parallel 1D CNN branches independently encode drug SMILES and protein sequences.

```
Drug:    Embedding(vocab, 128) → Conv1D(k=4,6,8) → AdaptiveMaxPool → concat → ℝ^384
Protein: Embedding(vocab, 128) → Conv1D(k=4,8,12) → AdaptiveMaxPool → concat → ℝ^384
FC Head: Linear(768→1024) → ReLU → Dropout → Linear(1024→256) → ReLU → Dropout → Linear(256→1)
Total parameters: ~1.75M
```

### AttentionDTA

Extends DeepDTA with multi-head self-attention (4 heads) pooling over CNN features.

```
Total parameters: ~2.93M
```

### WideDTA

Uses character n-gram tokenization instead of single-character tokens, enabling the model to capture local sequence motifs directly.

### GraphDTA

Represents drug molecules as molecular graphs processed by GCN or GAT layers (requires `torch-geometric` and `rdkit`). Protein branch uses the same CNN encoder as DeepDTA.

### CL-DTA

DeepDTA architecture with encoders initialized from contrastive pretraining. The pretrained encoders provide generalizable representations for cold-start scenarios.

---

## Contrastive Pretraining

### Augmentation Engine

**Drug (SMILES) augmentations:**

| Augmentation | Description |
|---|---|
| SMILES Enumeration | RDKit non-canonical SMILES generation — same molecule, different string representation |
| Atom Masking | Replace $k\%$ of SMILES tokens with `<MASK>` |
| Substructure Dropout | Remove atoms/bonds to create molecular analogs (RDKit validity-checked) |

**Protein (sequence) augmentations:**

| Augmentation | Description |
|---|---|
| Subsequence Cropping | Random contiguous window of 70–100% of sequence length |
| Residue Masking | Replace $k\%$ of amino acid characters with `<MASK>` |
| Residue Substitution | Biochemically conservative substitutions via BLOSUM62 probabilities |

### NT-Xent Loss

For a minibatch of $N$ entities, each augmented twice to views $(z_i, z_i^+)$:

$$\mathcal{L}_{\text{NT-Xent}} = -\frac{1}{2N}\sum_{i=1}^{N}\Bigl[\log\frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{k \neq i}\exp(\text{sim}(z_i, z_k)/\tau)} + \log\frac{\exp(\text{sim}(z_i^+, z_i)/\tau)}{\sum_{k \neq i}\exp(\text{sim}(z_i^+, z_k)/\tau)}\Bigr]$$

where $\text{sim}(u,v) = u^\top v / (\|u\|\|v\|)$ is cosine similarity and $\tau = 0.07$ is the temperature.

Pretraining optimizer: AdamW, lr $5\times10^{-4}$, cosine annealing LR schedule, weight decay $10^{-5}$.

---

## Evaluation Protocol

### Split Strategies

| Split | Train Entities in Test? | Difficulty |
|---|---|---|
| `random` | Yes (same drugs & targets) | Easiest — standard i.i.d. |
| `cold_drug` | No (unseen drugs at test) | Hard |
| `cold_target` | No (unseen targets at test) | Hard |
| `cold_both` | No (unseen drugs AND targets) | Hardest |

Cold splits partition entity IDs (not sample indices) ensuring zero leakage between train and test entity sets. This is verified programmatically after every split.

### Metrics

| Metric | Description |
|---|---|
| MSE | Mean squared error — primary regression metric |
| RMSE | Root mean squared error — interpretable error scale |
| CI | Concordance index — standard DTA ranking metric |
| Pearson $r$ | Linear correlation between predicted and true affinities |
| Spearman $\rho$ | Rank correlation |
| $r_m^2$ | Modified $r^2$ — KIBA literature standard |

### Experimental Seeds

Three seeds `{42, 123, 456}` are used per experiment. `random`, `numpy`, `torch`, and `torch.cuda` RNGs are all seeded for reproducibility.

---

## Configuration

Experiments are fully specified by hierarchical YAML configs (see `configs/`). Example structure:

```yaml
data:
  dataset: davis
  data_path: data/davis_processed.csv
  split: cold_drug          # random | cold_drug | cold_target | cold_both
  max_sml_len: 100
  max_prot_len: 1000

pretrain:
  mode: both_independent
  epochs: 100
  lr: 0.0005
  temperature: 0.07
  loss: nt_xent             # nt_xent | infonce | triplet
  drug_augmentations: [smiles_enumeration, atom_masking, substructure_dropout]
  prot_augmentations: [subsequence_crop, residue_masking, residue_substitution]

train:
  epochs: 30
  lr: 0.0001
  patience: 8
  freeze_strategy: gradual_unfreeze   # frozen | full_finetune | gradual_unfreeze

experiment:
  name: davis_cold_drug
  seeds: [42, 123, 456]
  device: auto
```

Eight pre-built configs covering all dataset × split combinations are provided in `configs/`.

---

## Visualization

`visualization.py` generates publication-quality figures from experiment JSON outputs:

- **CI Comparison:** Grouped bar chart comparing all models across split types
- **Embedding Scatter:** t-SNE / UMAP of learned drug/protein representations
- **Ablation Heatmap:** Component contribution analysis
- **Training Curves:** Multi-metric line plots per epoch
- **Prediction vs. Ground Truth:** Scatter plot with Pearson $r$ annotation

```bash
python -m Implementation_of_DeepDTA_pipeline.visualization --results-dir results/ --output-dir figures/
```

---

## References

- Chen et al. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML 2020.
- Oord et al. *Representation Learning with Contrastive Predictive Coding.* arXiv 2018.
- Öztürk et al. *DeepDTA: Deep Drug–Target Binding Affinity Prediction.* Bioinformatics 2018.
- Nguyen et al. *GraphDTA: Predicting Drug–Target Binding Affinity with Graph Neural Networks.* Bioinformatics 2021.
- Pahikkala et al. *Toward More Realistic Drug–Target Interaction Predictions.* Briefings in Bioinformatics 2015. (DAVIS dataset)
- Tang et al. *Making Sense of Large-Scale Kinase Inhibitor Bioactivity Data Sets.* J. Chemical Information and Modeling 2014. (KIBA dataset)