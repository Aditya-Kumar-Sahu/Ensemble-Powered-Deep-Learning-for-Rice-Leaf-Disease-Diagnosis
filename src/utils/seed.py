"""Seed utilities for reproducibility."""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set the global random seed to make experiments reproducible across Python, NumPy, and PyTorch.

    If CUDA is available, also seed CUDA RNGs for all devices and configure cuDNN for
    deterministic behavior with benchmarking disabled.

    Parameters:
        seed (int): Integer seed used to initialize RNGs for Python, NumPy, and PyTorch
            (CPU and, if available, CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
