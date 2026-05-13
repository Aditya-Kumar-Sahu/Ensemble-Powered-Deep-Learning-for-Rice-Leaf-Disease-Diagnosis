"""Checkpoint management utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import numpy as np


def ensure_dirs(paths: Union[List[str], List[Path]]) -> None:
    """
    Ensure each path in `paths` exists by creating missing directories, including
    any necessary parent directories.

    Parameters:
        paths (List[str] | List[Path]): Iterable of directory paths (strings or
            Path objects) to create if they do not already exist.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    filepath: Union[str, Path],
) -> None:
    """
    Save a training checkpoint containing model and optimizer state, the current epoch, and metrics to disk.

    Parameters:
        model: The PyTorch model whose state_dict will be saved.
        optimizer: The optimizer whose state_dict will be saved.
        epoch: The current epoch number to record in the checkpoint.
        metrics: Mapping of metric names to values to include in the checkpoint.
        filepath: Destination path (str or Path) where the checkpoint file will be written.

    The checkpoint is saved as a dictionary with the keys: "epoch", "model_state_dict",
    "optimizer_state_dict", and "metrics".
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a saved training checkpoint and optionally restore model and optimizer states.

    Parameters:
        filepath (str | Path): Path to the checkpoint file.
        model (torch.nn.Module, optional): If provided and the checkpoint contains a
            `model_state_dict`, load it into this model.
        optimizer (torch.optim.Optimizer, optional): If provided and the checkpoint
            contains an `optimizer_state_dict`, load it into this optimizer.
        device (torch.device, optional): Device to map loaded tensors to; defaults
            to CPU if not provided.

    Returns:
        dict: The checkpoint dictionary loaded from disk (contains keys such as `epoch`,
            `model_state_dict`, `optimizer_state_dict`, and `metrics` when present).
    """
    if device is None:
        device = torch.device("cpu")

    checkpoint = torch.load(filepath, map_location=device)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_history(history: Dict[str, Any], model_name: str, folder: str = "logs") -> str:
    """
    Save training history to a file named "<model_name>_history.npy" in the specified folder.

    Parameters:
        history (Dict[str, Any]): Training history data to persist.
        model_name (str): Model identifier used to form the history filename.
        folder (str): Destination directory; created if it does not exist.

    Returns:
        str: The path to the saved history file.
    """
    ensure_dirs([folder])
    path = os.path.join(folder, f"{model_name}_history.npy")
    np.save(path, history)
    return path


def load_history(model_name: str, folder: str = "logs") -> Dict[str, Any]:
    """
    Load a model's training history dictionary from a NumPy .npy file.

    Parameters:
        model_name (str): Base name of the model used to form the filename "<model_name>_history.npy".
        folder (str): Directory containing the history file (default "logs").

    Returns:
        dict: Training history dictionary loaded from "<folder>/<model_name>_history.npy".
    """
    path = os.path.join(folder, f"{model_name}_history.npy")
    return np.load(path, allow_pickle=True).item()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters:
        model (torch.nn.Module): Model whose trainable parameters will be counted.

    Returns:
        int: Total number of parameters with `requires_grad` set to True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
