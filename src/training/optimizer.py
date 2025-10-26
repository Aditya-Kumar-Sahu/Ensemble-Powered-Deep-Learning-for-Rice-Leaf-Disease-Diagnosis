"""Optimizer utilities."""

from typing import Any, Dict
import torch.optim as optim
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs: Dict[str, Any],
) -> optim.Optimizer:
    """
    Get optimizer for model training.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ("adam", "sgd", "adamw")
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer-specific arguments
        
    Returns:
        Optimizer instance
        
    Raises:
        ValueError: If optimizer_name is not supported
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    
    elif optimizer_name == "sgd":
        momentum = kwargs.pop("momentum", 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Supported optimizers are: adam, adamw, sgd"
        )
