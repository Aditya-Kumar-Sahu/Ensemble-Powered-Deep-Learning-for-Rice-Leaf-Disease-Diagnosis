"""Checkpoint management utilities."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch
import numpy as np


def ensure_dirs(paths: Union[List[str], List[Path]]) -> None:
    """
    Ensure that directories exist, creating them if necessary.
    
    Args:
        paths: List of directory paths to create
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
    Save model checkpoint with training state.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        metrics: Dictionary of metric values
        filepath: Path to save checkpoint
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
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model (optional)
        optimizer: Optimizer (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing checkpoint data
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint


def save_history(history: Dict[str, Any], model_name: str, folder: str = "logs") -> None:
    """
    Save training history to disk.
    
    Args:
        history: Dictionary containing training history
        model_name: Name of the model
        folder: Directory to save history
    """
    ensure_dirs([folder])
    np.save(os.path.join(folder, f"{model_name}_history.npy"), history)


def load_history(model_name: str, folder: str = "logs") -> Dict[str, Any]:
    """
    Load training history from disk.
    
    Args:
        model_name: Name of the model
        folder: Directory containing history
        
    Returns:
        Dictionary containing training history
    """
    path = os.path.join(folder, f"{model_name}_history.npy")
    return np.load(path, allow_pickle=True).item()


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
