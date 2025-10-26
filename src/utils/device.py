"""Device management utilities."""

import gc
import torch


def get_device() -> torch.device:
    """
    Get the available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def clear_gpu_cache() -> None:
    """Free GPU memory by clearing cache and running garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
