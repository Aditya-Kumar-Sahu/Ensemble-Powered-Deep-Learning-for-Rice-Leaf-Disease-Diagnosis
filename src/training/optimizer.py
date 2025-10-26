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
    Create an optimizer configured for the provided PyTorch model.
    
    Parameters:
        model: The neural network whose parameters will be optimized.
        optimizer_name: Which optimizer to construct — "adam", "adamw", or "sgd" (case-insensitive).
        learning_rate: Learning rate for the optimizer.
        weight_decay: Weight decay (L2 regularization) factor.
        **kwargs: Additional optimizer-specific keyword arguments. For "sgd", `momentum` can be provided (default 0.9).
    
    Returns:
        An instance of torch.optim.Optimizer configured for the model.
    
    Raises:
        ValueError: If `optimizer_name` is not one of "adam", "adamw", or "sgd".
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