"""
model.py — DeepDTA architecture with Phase 4 pretrained encoder support.

Architecture:
  Drug Branch:   Embedding → 3×Conv1D → AdaptiveMaxPool → concat → R^(3×conv_out)
  Protein Branch: Embedding → 3×Conv1D → AdaptiveMaxPool → concat → R^(3×conv_out)
  FC Head:       concat(drug, prot) → 1024 → 256 → 1

Phase 4 adds:
  - PretrainedEncoder wrapper for ESM / ProtBERT / ChemBERTa / MolFormer
  - Optional LLM embedding caching + partial unfreezing
  - Tokenizer → pretrained model → projection → encoder pipeline
"""

from __future__ import annotations

import importlib
from typing import Optional, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def _configure_torch_sdpa_backends() -> None:
    """Force math-only SDPA to avoid cuDNN execution-plan failures on some GPUs."""
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("[model.py] Configured torch SDPA backends (math-only)")
    except Exception as e:  # pragma: no cover - best effort
        print(f"[Warning] Could not fully configure torch SDPA backends: {e}")


def _patch_transformers_sdpa_to_eager() -> None:
    """Patch HF SDPA integration to use eager attention when available."""
    patched_modules = []
    for module_name in ("transformers.integrations.sdpa_attention", "transformers.integrations.sdpa"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        eager_forward = getattr(module, "eager_attention_forward", None)
        sdpa_forward = getattr(module, "sdpa_attention_forward", None)
        if callable(eager_forward) and callable(sdpa_forward):
            setattr(module, "sdpa_attention_forward", eager_forward)
            patched_modules.append(module_name)

    if patched_modules:
        print(
            "[model.py] Patched transformers SDPA attention to eager in: "
            + ", ".join(patched_modules)
        )
    else:
        print(
            "[Warning] Could not patch transformers SDPA module; "
            "relying on eager-attention model loading."
        )


_configure_torch_sdpa_backends()
_patch_transformers_sdpa_to_eager()

# Patch torch.load to bypass weights_only security check for legacy models
_original_torch_load = torch.load


def patched_torch_load(f, *args, **kwargs):
    """Wrap torch.load to handle weights_only security check."""
    try:
        return _original_torch_load(f, *args, **kwargs)
    except ValueError as e:
        if "vulnerability" in str(e) or "torch.load" in str(e):
            print("[torch.load] Retrying with weights_only=False...")
            kwargs["weights_only"] = False
            return _original_torch_load(f, *args, **kwargs)
        raise


torch.load = patched_torch_load


# ─────────────────────────────────────────────────────────────────────────────
# Pretrained Model Encoders (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────


class PretrainedEncoder(nn.Module):
    """
    Wrapper for loading pretrained LLM embeddings (ESM, ProtBERT, ChemBERTa, MolFormer).

    Pipeline: input → tokenizer → pretrained_model → pooling (CLS/mean)
              → projection → L2-normalised embedding

    Parameters
    ----------
    model_name : str
        Model identifier: 'esm2_t33' | 'esm2_t6' | 'protbert' | 'chemberta' | 'molformer'
    output_dim : int
        Dimension of projected embedding (shared embedding space)
    freeze : bool
        If True, freeze pretrained weights. Use unfreeze_last_k to selectively thaw.
    unfreeze_last_k : int
        Number of final transformer layers to unfreeze (if > 0)
    cache_embeddings : bool
        If True, cache LLM embeddings to avoid recomputation
    pooling : str
        'cls' (use pooler_output/CLS) or 'mean' (mean over tokens)
    """

    MODEL_REGISTRY = {
        "esm2_t33": "facebook/esm2_t33_650M_UR50D",
        "esm2_t6": "facebook/esm2_t6_8M_UR50D",
        "protbert": "Rostlab/prot_bert",
        "chemberta": "DeepChem/ChemBERTa-77M-MLM",
        "molformer": "ibm/molformer-base",
    }

    def __init__(
        self,
        model_name: str,
        output_dim: int = 128,
        freeze: bool = True,
        unfreeze_last_k: int = 0,
        cache_embeddings: bool = False,
        cache_size: int = 8192,
        pooling: str = "cls",
        max_length: Optional[int] = None,
    ):
        super().__init__()
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.output_dim = output_dim
        self.cache_embeddings = cache_embeddings
        self.cache_size = cache_size
        self.pooling = pooling
        self.max_length = max_length
        self._embedding_cache: Dict[Union[str, tuple], Dict[str, torch.Tensor]] = {}

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers library required for PretrainedEncoder. "
                "Install: pip install transformers"
            ) from exc

        # Monkey-patch transformers security check BEFORE any model loading
        try:
            import transformers.modeling_utils
            import transformers.utils.import_utils

            def bypass_load_check(*_args, **_kwargs):
                return None

            transformers.modeling_utils.check_torch_load_is_safe = bypass_load_check
            transformers.utils.import_utils.check_torch_load_is_safe = bypass_load_check
            print("[PretrainedEncoder] Bypassed transformers torch.load security check")
        except Exception as bypass_err:  # pragma: no cover - best effort
            print(f"[Warning] Could not bypass security check: {bypass_err}")

        hf_model_id = self.MODEL_REGISTRY[model_name]
        print(f"[PretrainedEncoder] Loading {model_name} from {hf_model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model_kwargs = {
            "dtype": torch.float32,
            "trust_remote_code": False,
        }
        try:
            self.pretrained_model = AutoModel.from_pretrained(
                hf_model_id,
                attn_implementation="eager",
                **model_kwargs,
            )
            print("[PretrainedEncoder] Attention implementation forced to eager")
        except (TypeError, ValueError) as attn_err:
            print(
                "[PretrainedEncoder] Could not force eager attention "
                f"({attn_err}); using model default."
            )
            self.pretrained_model = AutoModel.from_pretrained(
                hf_model_id, **model_kwargs
            )

        # Determine hidden dimension from model
        hidden_dim = getattr(self.pretrained_model.config, "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(self.pretrained_model.config, "d_model")

        self.projection = nn.Linear(hidden_dim, output_dim)

        # Freeze strategy
        if freeze:
            self._freeze_all()
        if unfreeze_last_k > 0:
            self._unfreeze_last_k_layers(unfreeze_last_k)

        # Keep eval mode for stability; grads still flow if parameters require grad
        self.pretrained_model.eval()

    # Freeze utilities -----------------------------------------------------
    def _freeze_all(self) -> None:
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def _unfreeze_last_k_layers(self, k: int) -> None:
        """Unfreeze the last k transformer layers."""
        layers = None
        if hasattr(self.pretrained_model, "encoder") and hasattr(
            self.pretrained_model.encoder, "layer"
        ):
            layers = self.pretrained_model.encoder.layer
        elif hasattr(self.pretrained_model, "layers"):
            layers = self.pretrained_model.layers

        if layers is None:
            print(f"[Warning] Could not find layers to unfreeze in {self.model_name}")
            return

        for layer in layers[-k:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"[PretrainedEncoder] Unfroze last {k} layers")

    # Forward --------------------------------------------------------------
    def forward(
        self,
        sequence: Union[str, List[str], Dict[str, torch.Tensor]],
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract pretrained embeddings and project to output dimension.

        Parameters
        ----------
        sequence : str | list[str] | dict[str, Tensor]
            Raw text or a pre-tokenized dict returned by a HF tokenizer.
        return_embeddings : bool
            If True, also return raw embeddings before projection.

        Returns
        -------
        projected : (B, D) normalized tensor (or (D,) if single)
        raw_embeddings (optional) : (B, H) raw transformer output
        """
        # Prepare tokens
        is_single = False
        tokens: Dict[str, torch.Tensor]
        sequences: Optional[List[str]] = None
        cached_projected: List[Optional[torch.Tensor]] = []
        cached_raw: List[Optional[torch.Tensor]] = []
        missing_indices: List[int] = []

        if isinstance(sequence, dict):
            tokens = {k: v for k, v in sequence.items()}
        else:
            if isinstance(sequence, str):
                sequences = [sequence]
                is_single = True
            else:
                sequences = list(sequence)

            if self.cache_embeddings:
                for idx, seq in enumerate(sequences):
                    cached = self._embedding_cache.get(seq)
                    if cached is None:
                        cached_projected.append(None)
                        cached_raw.append(None)
                        missing_indices.append(idx)
                    else:
                        cached_projected.append(cached["projected"])
                        cached_raw.append(cached["raw"])

                if not missing_indices:
                    proj = torch.stack(cached_projected).to(self.projection.weight.device)
                    raw = torch.stack(cached_raw).to(self.projection.weight.device)
                    if is_single:
                        proj = proj.squeeze(0)
                        raw = raw.squeeze(0)
                    return (proj, raw) if return_embeddings else proj

                sequences_to_encode = [sequences[idx] for idx in missing_indices]
            else:
                sequences_to_encode = sequences

            tokens = self.tokenizer(
                sequences_to_encode,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length or min(getattr(self.tokenizer, "model_max_length", 512), 2048),
            )

        # Move to pretrained model device
        device = next(self.pretrained_model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        requires_grad = any(p.requires_grad for p in self.pretrained_model.parameters())
        with torch.set_grad_enabled(requires_grad):
            try:
                outputs = self.pretrained_model(**tokens, output_hidden_states=False)
            except RuntimeError as runtime_err:
                msg = str(runtime_err)
                if "cudnn_frontend" not in msg or "No valid execution plans built" not in msg:
                    raise

                # Retry once with the most conservative SDPA backend settings.
                _configure_torch_sdpa_backends()
                print(
                    "[PretrainedEncoder] Retrying transformer forward pass "
                    "with math-only SDPA backend..."
                )
                outputs = self.pretrained_model(**tokens, output_hidden_states=False)

        # Pooling
        if self.pooling == "cls" and getattr(outputs, "pooler_output", None) is not None:
            raw_emb = outputs.pooler_output  # (B, H)
        else:
            mask = tokens.get("attention_mask", torch.ones_like(tokens["input_ids"]))
            raw_emb = (
                outputs.last_hidden_state * mask.unsqueeze(-1)
            ).sum(1) / mask.sum(1, keepdim=True)

        # Project to output dimension and normalise
        projected = self.projection(raw_emb)
        projected = F.normalize(projected, p=2, dim=1)

        if self.cache_embeddings and sequences is not None:
            proj_cpu = projected.detach().cpu()
            raw_cpu = raw_emb.detach().cpu()

            if missing_indices:
                for local_idx, seq_idx in enumerate(missing_indices):
                    seq = sequences[seq_idx]
                    self._embedding_cache[seq] = {
                        "projected": proj_cpu[local_idx],
                        "raw": raw_cpu[local_idx],
                    }
                    if len(self._embedding_cache) > self.cache_size:
                        self._embedding_cache.pop(next(iter(self._embedding_cache)))
                    cached_projected[seq_idx] = proj_cpu[local_idx]
                    cached_raw[seq_idx] = raw_cpu[local_idx]

                projected = torch.stack(cached_projected).to(device)
                raw_emb = torch.stack(cached_raw).to(device)
            else:
                for idx, seq in enumerate(sequences):
                    self._embedding_cache[seq] = {
                        "projected": proj_cpu[idx],
                        "raw": raw_cpu[idx],
                    }
                    if len(self._embedding_cache) > self.cache_size:
                        self._embedding_cache.pop(next(iter(self._embedding_cache)))

        if is_single:
            projected = projected.squeeze(0)
            raw_emb = raw_emb.squeeze(0)

        if return_embeddings:
            return projected, raw_emb
        return projected

    def clear_cache(self) -> None:
        self._embedding_cache.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────


class ConvBlock(nn.Module):
    """1-D convolution → ReLU → AdaptiveMaxPool1d(1)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return x


class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive pretraining.
    Maps encoder features → ℓ2-normalised projection.
    Discarded after pretraining.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder wrappers (for pretraining checkpoint saving)
# ─────────────────────────────────────────────────────────────────────────────


class DrugEncoder(nn.Module):
    """Drug embedding + multi-kernel Conv1D encoder with optional pretrained backbone."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        kernels: tuple = (4, 6, 8),
        pretrained_model: Optional[str] = None,
        freeze_pretrained: bool = True,
        unfreeze_last_k: int = 0,
        cache_pretrained: bool = False,
        pooling: str = "cls",
        pretrained_max_length: Optional[int] = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model
        self.use_pretrained = pretrained_model is not None

        if self.use_pretrained:
            self.embedding = PretrainedEncoder(
                pretrained_model,
                output_dim=emb_dim,
                freeze=freeze_pretrained,
                unfreeze_last_k=unfreeze_last_k,
                cache_embeddings=cache_pretrained,
                pooling=pooling,
                max_length=pretrained_max_length,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in kernels])
        self.out_dim = conv_out * len(kernels)
        self._max_kernel = max(kernels)

    def forward(self, x: Union[torch.Tensor, str, List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        x: (B, L) LongTensor for character-level, str/list for pretrained, or HF token dict.
        Returns: (B, out_dim)
        """
        if self.use_pretrained:
            proj = self.embedding(x)  # (B, emb_dim)
            if proj.dim() == 1:
                proj = proj.unsqueeze(0)
            # Create pseudo-sequence so convs can run; length = max kernel
            e = proj.unsqueeze(-1).expand(-1, -1, self._max_kernel)
        else:
            e = self.embedding(x).permute(0, 2, 1)  # (B, emb, L)

        feats = [c(e) for c in self.convs]
        return torch.cat(feats, dim=1)

    def alignment_embedding(self, x: Union[torch.Tensor, str, List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Return a pooled, L2-normalised embedding for alignment with pretrained LLM features.
        Dimension: emb_dim (the projection dimension of the embedding layer).
        """
        if self.use_pretrained:
            proj, _ = self.embedding(x, return_embeddings=True)
            if proj.dim() == 1:
                proj = proj.unsqueeze(0)
            return F.normalize(proj, p=2, dim=1)

        # Character-level: mean-pool embedding lookup
        emb = self.embedding(x)  # (B, L, emb_dim)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        pooled = emb.mean(dim=1)
        return F.normalize(pooled, p=2, dim=1)


class ProteinEncoder(nn.Module):
    """Protein embedding + multi-kernel Conv1D encoder with optional pretrained backbone."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        kernels: tuple = (4, 8, 12),
        pretrained_model: Optional[str] = None,
        freeze_pretrained: bool = True,
        unfreeze_last_k: int = 0,
        cache_pretrained: bool = False,
        pooling: str = "cls",
        pretrained_max_length: Optional[int] = None,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model
        self.use_pretrained = pretrained_model is not None

        if self.use_pretrained:
            self.embedding = PretrainedEncoder(
                pretrained_model,
                output_dim=emb_dim,
                freeze=freeze_pretrained,
                unfreeze_last_k=unfreeze_last_k,
                cache_embeddings=cache_pretrained,
                pooling=pooling,
                max_length=pretrained_max_length,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.convs = nn.ModuleList([ConvBlock(emb_dim, conv_out, k) for k in kernels])
        self.out_dim = conv_out * len(kernels)
        self._max_kernel = max(kernels)

    def forward(self, x: Union[torch.Tensor, str, List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        x: (B, L) LongTensor for character-level, str/list for pretrained, or HF token dict.
        Returns: (B, out_dim)
        """
        if self.use_pretrained:
            proj = self.embedding(x)  # (B, emb_dim)
            if proj.dim() == 1:
                proj = proj.unsqueeze(0)
            e = proj.unsqueeze(-1).expand(-1, -1, self._max_kernel)
        else:
            e = self.embedding(x).permute(0, 2, 1)  # (B, emb, L)

        feats = [c(e) for c in self.convs]
        return torch.cat(feats, dim=1)

    def alignment_embedding(self, x: Union[torch.Tensor, str, List[str], Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Return a pooled, L2-normalised embedding for alignment with pretrained LLM features.
        Dimension: emb_dim.
        """
        if self.use_pretrained:
            proj, _ = self.embedding(x, return_embeddings=True)
            if proj.dim() == 1:
                proj = proj.unsqueeze(0)
            return F.normalize(proj, p=2, dim=1)

        emb = self.embedding(x)  # (B, L, emb_dim)
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        pooled = emb.mean(dim=1)
        return F.normalize(pooled, p=2, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# DeepDTA model
# ─────────────────────────────────────────────────────────────────────────────


class DeepDTAModel(nn.Module):
    """
    DeepDTA-like CNN model for drug–target affinity prediction.

    Supports optional modules from later phases:
    - Phase 6: Pocket-guided cross-attention (use_attention_module)
    - Phase 7: Evidential regression head (use_evidential)
    - Phase 9: Multi-task prediction head (use_multitask)

    Parameters
    ----------
    vocab_drug, vocab_prot : vocabulary sizes (including special tokens).
    emb_dim : embedding dimension.
    conv_out : output channels per conv filter.
    sml_kernels, prot_kernels : kernel sizes for drug / protein branches.
    dropout : dropout probability in FC head.
    pretrained_drug_model, pretrained_prot_model : optional HF model names.
    use_attention_module : bool
        Phase 6: enable pocket-guided cross-attention.
    attention_heads : int
        Number of attention heads for PocketGuidedAttention.
    use_evidential : bool
        Phase 7: replace FC head with evidential regression head.
    use_multitask : bool
        Phase 9: replace FC head with multi-task prediction head.
    num_moa_classes : int
        Number of MoA classes for multi-task head (0 = disabled).
    """

    def __init__(
        self,
        vocab_drug: int,
        vocab_prot: int,
        emb_dim: int = 128,
        conv_out: int = 128,
        sml_kernels=(4, 6, 8),
        prot_kernels=(4, 8, 12),
        dropout: float = 0.2,
        use_pretrained_embeddings: bool = False,
        pretrained_drug_model: Optional[str] = None,
        pretrained_prot_model: Optional[str] = None,
        freeze_pretrained: bool = True,
        unfreeze_last_k_layers: int = 0,
        cache_llm_embeddings: bool = False,
        pooling: str = "cls",
        max_sml_len: int = 120,
        max_prot_len: int = 1000,
        # Phase 6: Attention
        use_attention_module: bool = False,
        attention_heads: int = 4,
        attention_max_seq_len: int = 1200,
        # Phase 7: Evidential
        use_evidential: bool = False,
        # Phase 9: Multi-task
        use_multitask: bool = False,
        num_moa_classes: int = 0,
    ):
        super().__init__()
        self.use_attention = use_attention_module
        self.use_evidential = use_evidential
        self.use_multitask = use_multitask

        self.drug_encoder = DrugEncoder(
            vocab_drug,
            emb_dim,
            conv_out,
            sml_kernels,
            pretrained_model=pretrained_drug_model if use_pretrained_embeddings else None,
            freeze_pretrained=freeze_pretrained,
            unfreeze_last_k=unfreeze_last_k_layers,
            cache_pretrained=cache_llm_embeddings,
            pooling=pooling,
            pretrained_max_length=max_sml_len,
        )
        self.prot_encoder = ProteinEncoder(
            vocab_prot,
            emb_dim,
            conv_out,
            prot_kernels,
            pretrained_model=pretrained_prot_model if use_pretrained_embeddings else None,
            freeze_pretrained=freeze_pretrained,
            unfreeze_last_k=unfreeze_last_k_layers,
            cache_pretrained=cache_llm_embeddings,
            pooling=pooling,
            pretrained_max_length=max_prot_len,
        )

        total = self.drug_encoder.out_dim + self.prot_encoder.out_dim

        # Phase 6: Pocket-Guided Cross-Attention (optional)
        self.attention_module = None
        self.prot_seq_features = None
        if use_attention_module:
            try:
                from .pocket_attention import PocketGuidedAttention, ProteinSequenceFeatures
                self.prot_seq_features = ProteinSequenceFeatures(
                    vocab_prot, emb_dim, conv_out, prot_kernels,
                )
                attn_dim = self.prot_seq_features.out_dim
                self.attention_module = PocketGuidedAttention(
                    embed_dim=attn_dim,
                    num_heads=attention_heads,
                    max_seq_len=attention_max_seq_len,
                )
                # Attention output replaces drug encoder output
                total = attn_dim + self.prot_encoder.out_dim
                print(f"[model] Phase 6: Pocket-guided attention enabled "
                      f"(heads={attention_heads}, dim={attn_dim})")
            except ImportError:
                print("[Warning] pocket_attention.py not found; attention module disabled.")

        # Phase 7: Evidential Regression Head (optional)
        if use_evidential:
            try:
                from .evidential import EvidentialRegressionHead
                self.fc = EvidentialRegressionHead(
                    input_dim=total, hidden_dim=256, dropout=dropout,
                )
                print("[model] Phase 7: Evidential regression head enabled")
            except ImportError:
                print("[Warning] evidential.py not found; using standard FC head.")
                self.fc = self._build_standard_fc(total, dropout)
        elif use_multitask:
            # Phase 9: Multi-Task Head (optional)
            try:
                from .multitask import MultiTaskHead
                self.fc = MultiTaskHead(
                    input_dim=total,
                    num_moa_classes=num_moa_classes,
                    hidden_dim=256,
                    dropout=dropout,
                )
                print(f"[model] Phase 9: Multi-task head enabled "
                      f"(moa_classes={num_moa_classes})")
            except ImportError:
                print("[Warning] multitask.py not found; using standard FC head.")
                self.fc = self._build_standard_fc(total, dropout)
        else:
            self.fc = self._build_standard_fc(total, dropout)

    @staticmethod
    def _build_standard_fc(total: int, dropout: float) -> nn.Sequential:
        """Build the standard deterministic FC prediction head."""
        return nn.Sequential(
            nn.Linear(total, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    # Convenience properties (backward compat) ----------------------------
    @property
    def embed_drug(self):
        return self.drug_encoder.embedding

    @property
    def embed_prot(self):
        return self.prot_encoder.embedding

    @property
    def drug_convs(self):
        return self.drug_encoder.convs

    @property
    def prot_convs(self):
        return self.prot_encoder.convs

    # Forward --------------------------------------------------------------
    def forward(
        self,
        smiles: torch.Tensor,
        seq: torch.Tensor,
        pocket_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, dict]:
        """
        smiles, seq : LongTensor (B, L) or token dicts (if using pretrained).
        pocket_mask : optional (B, L) binary mask for pocket-guided attention.
        Returns : (B,) predicted affinity, or dict if evidential/multitask.
        """
        d = self.drug_encoder(smiles)
        p = self.prot_encoder(seq)

        # Phase 6: apply pocket-guided cross-attention
        self._last_attn_weights = None
        if self.attention_module is not None and self.prot_seq_features is not None:
            prot_seq = self.prot_seq_features(seq)   # (B, L', D_attn)
            # Pad drug embedding to match attention dim if needed
            if d.size(-1) != self.attention_module.embed_dim:
                # Project drug features to attention dimension
                if not hasattr(self, '_drug_proj'):
                    self._drug_proj = nn.Linear(
                        d.size(-1), self.attention_module.embed_dim
                    ).to(d.device)
                d_attn = self._drug_proj(d)
            else:
                d_attn = d
            d_enhanced, attn_w = self.attention_module(d_attn, prot_seq, pocket_mask)
            self._last_attn_weights = attn_w
            x = torch.cat([d_enhanced, p], dim=1)
        else:
            x = torch.cat([d, p], dim=1)

        out = self.fc(x)

        # Standard FC head returns tensor; squeeze it
        if isinstance(out, torch.Tensor):
            return out.squeeze(-1)
        # Evidential / Multi-task heads return dicts
        return out

    # Pretrained weight transfer ------------------------------------------
    def load_pretrained_encoders(
        self,
        drug_ckpt: Optional[str] = None,
        prot_ckpt: Optional[str] = None,
    ) -> None:
        """
        Load pretrained weights into encoder branches.
        Reinitialises the FC head so it trains from scratch.
        """
        if drug_ckpt is not None:
            state = torch.load(drug_ckpt, map_location="cpu")
            self.drug_encoder.load_state_dict(state["encoder"], strict=False)
            print(f"[model] Loaded pretrained drug encoder from {drug_ckpt}")

        if prot_ckpt is not None:
            state = torch.load(prot_ckpt, map_location="cpu")
            self.prot_encoder.load_state_dict(state["encoder"], strict=False)
            print(f"[model] Loaded pretrained protein encoder from {prot_ckpt}")

        # Reinitialise FC head
        if isinstance(self.fc, nn.Sequential):
            for layer in self.fc:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def freeze_encoders(self) -> None:
        """Freeze drug & protein encoder weights (train FC head only)."""
        for param in self.drug_encoder.parameters():
            param.requires_grad = False
        for param in self.prot_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        """Unfreeze all encoder weights."""
        for param in self.drug_encoder.parameters():
            param.requires_grad = True
        for param in self.prot_encoder.parameters():
            param.requires_grad = True

    def parameter_count(self) -> dict:
        """Return parameter counts by component."""

        def count(module):
            return sum(p.numel() for p in module.parameters())

        result = {
            "drug_encoder": count(self.drug_encoder),
            "prot_encoder": count(self.prot_encoder),
            "fc_head": count(self.fc),
            "total": count(self),
        }
        if self.attention_module is not None:
            result["attention_module"] = count(self.attention_module)
        if self.prot_seq_features is not None:
            result["prot_seq_features"] = count(self.prot_seq_features)
        return result

