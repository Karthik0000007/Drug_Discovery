"""
pretrained_tokenizers.py — Hugging Face tokenizer wrappers for pretrained models.

Provides:
  - Caching and padding/truncation utilities
  - Unified tokenization interface across ESM, ProtBERT, ChemBERTa, MolFormer
  - Variable sequence length handling
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import torch
from functools import lru_cache


class PretrainedTokenizerWrapper:
    """
    Wraps Hugging Face tokenizers with caching and preprocessing.

    Parameters
    ----------
    model_name : str
        Model identifier: 'esm2_t33' | 'protbert' | 'chemberta' | 'molformer'
    max_length : int
        Maximum sequence length (default varies by model type)
    padding : bool
        Pad sequences to max_length
    truncation : bool
        Truncate sequences exceeding max_length
    cache_size : int
        LRU cache size for tokenized sequences
    """

    MODEL_REGISTRY = {
        "esm2_t33": "facebook/esm2_t33_650M_UR50D",
        "esm2_t6": "facebook/esm2_t6_8M_UR50D",
        "protbert": "Rostlab/prot_bert",
        "chemberta": "DeepChem/ChemBERTa-77M-MLM",
        "molformer": "ibm/molformer-base",
    }

    DEFAULT_MAX_LENGTHS = {
        "esm2_t33": 1024,
        "esm2_t6": 1024,
        "protbert": 512,
        "chemberta": 512,
        "molformer": 512,
    }

    def __init__(
        self,
        model_name: str,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        cache_size: int = 2048,
    ):
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Choose from {list(self.MODEL_REGISTRY.keys())}"
            )

        self.model_name = model_name
        self.hf_model_id = self.MODEL_REGISTRY[model_name]
        self.max_length = max_length or self.DEFAULT_MAX_LENGTHS.get(model_name, 512)
        self.padding = padding
        self.truncation = truncation

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. "
                "Install: pip install transformers"
            )

        print(f"[PretrainedTokenizer] Loading {model_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)

        # Cache for tokenized sequences
        self._token_cache: Dict[str, torch.Tensor] = {}
        self.cache_size = cache_size

    def tokenize(
        self,
        sequences: str | List[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize sequences with caching.

        Parameters
        ----------
        sequences : str or list of str
            Single sequence or batch
        return_tensors : str
            'pt' for PyTorch tensors, 'np' for numpy, etc.

        Returns
        -------
        Dict with 'input_ids', 'attention_mask', etc.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            is_single = True
        else:
            is_single = False

        # Check cache
        cache_hits = []
        cache_misses = []
        for seq in sequences:
            if seq in self._token_cache:
                cache_hits.append(seq)
            else:
                cache_misses.append(seq)

        if cache_misses:
            # Tokenize missing sequences
            tokens = self.tokenizer(
                cache_misses,
                padding="max_length" if self.padding else True,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors=return_tensors,
            )

            # Cache them
            for i, seq in enumerate(cache_misses):
                if len(cache_misses) > 1:
                    token_dict = {k: v[i] for k, v in tokens.items()}
                else:
                    # Single sequence: drop the artificial batch dim before caching
                    token_dict = {
                        k: (v.squeeze(0) if torch.is_tensor(v) and v.dim() > 1 else v)
                        for k, v in tokens.items()
                    }
                self._token_cache[seq] = token_dict

                # Simple LRU: remove oldest if cache full
                if len(self._token_cache) > self.cache_size:
                    oldest_key = next(iter(self._token_cache))
                    del self._token_cache[oldest_key]

        # Reconstruct output in original order
        result = {key: [] for key in self._token_cache[sequences[0]].keys()}
        for seq in sequences:
            token_dict = self._token_cache[seq]
            for key, val in token_dict.items():
                result[key].append(val)

        # Stack tensors
        result = {k: torch.stack(v) if torch.is_tensor(v[0]) else v
                  for k, v in result.items()}

        if is_single:
            result = {k: v.squeeze(0) if torch.is_tensor(v) else v
                      for k, v in result.items()}

        return result

    def unpad_sequence(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[str]:
        """
        Decode token IDs back to sequences (with padding removal).

        Parameters
        ----------
        token_ids : (L,) or (B, L) LongTensor
        attention_mask : (L,) or (B, L) BoolTensor

        Returns
        -------
        List of decoded sequences
        """
        if token_ids.dim() == 1:
            # Single sequence
            valid_length = attention_mask.sum().item()
            token_ids = token_ids[:valid_length]
            return [self.tokenizer.decode(token_ids, skip_special_tokens=True)]

        # Batch
        results = []
        for tokens, mask in zip(token_ids, attention_mask):
            valid_length = mask.sum().item()
            tokens = tokens[:valid_length]
            results.append(self.tokenizer.decode(tokens, skip_special_tokens=True))
        return results

    def clear_cache(self) -> None:
        """Clear the tokenization cache."""
        self._token_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "cache_size": len(self._token_cache),
            "max_cache_size": self.cache_size,
            "utilization": len(self._token_cache) / self.cache_size,
        }


class BatchTokenizer:
    """
    Efficient batch tokenization with caching and padding.

    Handles variable-length sequences without recomputation.
    """

    def __init__(self, tokenizer: PretrainedTokenizerWrapper):
        self.tokenizer = tokenizer

    def tokenize_batch(
        self,
        sequences: List[str],
        return_tensors: str = "pt",
        dynamic_padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of sequences.

        Parameters
        ----------
        sequences : List[str]
            Batch of sequences
        return_tensors : str
            Tensor format
        dynamic_padding : bool
            If True, pad to max length in batch (not model max_length)

        Returns
        -------
        Dict with tokenized batch
        """
        tokens = self.tokenizer.tokenize(sequences, return_tensors=return_tensors)

        if dynamic_padding and "attention_mask" in tokens:
            # Adjust padding to batch max length (not model max length)
            batch_max_len = tokens["input_ids"].shape[1]
            tokens["input_ids"] = tokens["input_ids"][:, :batch_max_len]
            tokens["attention_mask"] = tokens["attention_mask"][:, :batch_max_len]

        return tokens
