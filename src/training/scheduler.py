"""Learning rate scheduler utilities."""

from typing import Any, Dict, Optional
import torch.optim as optim
from torch.optim import lr_scheduler


def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = "cosine",
    num_epochs: int = 100,
    **kwargs: Dict[str, Any],
) -> Optional[lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler ("cosine", "step", "plateau", "none")
        num_epochs: Total number of training epochs
        **kwargs: Additional scheduler-specific arguments
        
    Returns:
        Scheduler instance or None
        
    Raises:
        ValueError: If scheduler_name is not supported
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "none" or scheduler_name is None:
        return None
    
    elif scheduler_name == "cosine":
        T_max = kwargs.pop("T_max", num_epochs)
        eta_min = kwargs.pop("eta_min", 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            **kwargs
        )
    
    elif scheduler_name == "step":
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )
    
    elif scheduler_name == "plateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            **kwargs
        )
    
    elif scheduler_name == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported schedulers are: cosine, step, plateau, exponential, none"
        )
