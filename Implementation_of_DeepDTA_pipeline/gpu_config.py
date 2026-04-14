"""
gpu_config.py — GPU optimization utilities for CL-DTA pipeline.

Configures CUDA to utilize ~80% of GPU memory and applies
performance optimizations (cuDNN benchmarking, TF32, etc.).
"""

from __future__ import annotations

import os
import torch


def configure_gpu(
    memory_fraction: float = 0.80,
    enable_cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
    verbose: bool = True,
) -> torch.device:
    """
    Configure GPU for optimal training performance.

    Parameters
    ----------
    memory_fraction : float
        Fraction of GPU memory to allow PyTorch to use (0.0-1.0).
        Default 0.80 = 80%.
    enable_cudnn_benchmark : bool
        Enable cuDNN autotuner for fastest conv algorithms.
    enable_tf32 : bool
        Enable TF32 on Ampere+ GPUs for faster matmul/conv.
    verbose : bool
        Print GPU configuration info.

    Returns
    -------
    torch.device
        The configured device (cuda or cpu).
    """
    if not torch.cuda.is_available():
        if verbose:
            print("[gpu_config] CUDA not available — using CPU.")
        return torch.device("cpu")

    device = torch.device("cuda")

    # ── Memory fraction ──
    torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)

    # ── cuDNN benchmark: auto-selects fastest conv algorithm ──
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # ── TF32 precision (Ampere GPUs: RTX 30xx, A100, etc.) ──
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Optimize CUDA memory allocator ──
    # Use expandable segments to reduce fragmentation
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True"
    )

    if verbose:
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        usable_mem = total_mem * memory_fraction
        print(f"[gpu_config] GPU: {gpu_name}")
        print(f"[gpu_config] Total memory: {total_mem:.1f} GB")
        print(f"[gpu_config] Usable memory ({memory_fraction:.0%}): {usable_mem:.1f} GB")
        print(f"[gpu_config] cuDNN benchmark: {enable_cudnn_benchmark}")
        print(f"[gpu_config] TF32 enabled: {enable_tf32}")
        cap = torch.cuda.get_device_capability(0)
        print(f"[gpu_config] Compute capability: {cap[0]}.{cap[1]}")
        if cap[0] >= 7:
            print(f"[gpu_config] Mixed precision (FP16) supported ✓")
        if cap[0] >= 8:
            print(f"[gpu_config] TF32 + BF16 supported ✓")

    return device


def get_optimal_num_workers() -> int:
    """Return optimal number of DataLoader workers based on CPU count."""
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    # Use up to 4 workers (more can cause diminishing returns on Windows)
    return min(4, max(1, cpu_count // 2))


def get_optimal_batch_size(model_name: str = "DeepDTA", phase: str = "train") -> int:
    """
    Suggest an optimal batch size based on GPU memory and model.

    Parameters
    ----------
    model_name : str
        Model architecture name.
    phase : str
        'train' for supervised, 'pretrain' for contrastive.
    """
    if not torch.cuda.is_available():
        return 64

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # Heuristic batch sizes based on GPU memory
    if phase == "pretrain":
        if total_mem_gb >= 16:
            return 512
        elif total_mem_gb >= 8:
            return 256
        elif total_mem_gb >= 4:
            return 128
        else:
            return 64
    else:  # supervised training
        if total_mem_gb >= 16:
            return 512
        elif total_mem_gb >= 8:
            return 256
        elif total_mem_gb >= 4:
            return 128
        else:
            return 64


def try_compile_model(model: torch.nn.Module, verbose: bool = True) -> torch.nn.Module:
    """
    Attempt to compile model with torch.compile() for PyTorch 2.0+ speedups.
    Falls back gracefully if not supported.
    """
    if not hasattr(torch, "compile"):
        if verbose:
            print("[gpu_config] torch.compile not available (requires PyTorch 2.0+)")
        return model

    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        if verbose:
            print("[gpu_config] Model compiled with torch.compile (reduce-overhead) ✓")
        return compiled
    except Exception as e:
        if verbose:
            print(f"[gpu_config] torch.compile failed ({e}), using eager mode.")
        return model
