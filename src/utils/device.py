"""Device management utilities."""

import gc
import torch


def get_device() -> torch.device:
    """
    Determine the preferred torch device, preferring CUDA, then MPS, then CPU.

    Returns:
        torch.device: The selected device — 'cuda' if CUDA is available, 'mps' if MPS is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def clear_gpu_cache() -> None:
    """
    Free GPU memory resources by running garbage collection and clearing the CUDA cache when available.

    This triggers a Python garbage collection pass and empties PyTorch's CUDA cache
    if CUDA is available; it has no CUDA-specific effect on systems without CUDA.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
