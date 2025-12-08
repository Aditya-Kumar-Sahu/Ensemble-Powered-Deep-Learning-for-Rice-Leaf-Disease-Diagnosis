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
    Create a learning-rate scheduler for the given optimizer based on the provided scheduler_name.

    Parameters:
        optimizer: Optimizer instance to attach the scheduler to.
        scheduler_name: One of "cosine", "step", "plateau", "exponential", or "none" (case-insensitive).
            - "cosine": accepts `T_max` (default num_epochs) and `eta_min` (default 0).
            - "step": accepts `step_size` (default 30) and `gamma` (default 0.1).
            - "plateau": accepts `mode` (default "min"), `factor` (default 0.1), and `patience` (default 10).
            - "exponential": accepts `gamma` (default 0.95).
        num_epochs: Total number of training epochs; used as the default `T_max` for the cosine scheduler.
        **kwargs: Additional scheduler-specific keyword arguments; defaults shown above are used when keys are absent.

    Returns:
        Scheduler instance attached to `optimizer`, or `None` if `scheduler_name` is "none".

    Raises:
        ValueError: If `scheduler_name` is not one of the supported options.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "none" or scheduler_name is None:
        return None

    elif scheduler_name == "cosine":
        T_max = kwargs.pop("T_max", num_epochs)
        eta_min = kwargs.pop("eta_min", 0)
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, **kwargs
        )

    elif scheduler_name == "step":
        step_size = kwargs.pop("step_size", 30)
        gamma = kwargs.pop("gamma", 0.1)
        return lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, **kwargs
        )

    elif scheduler_name == "plateau":
        mode = kwargs.pop("mode", "min")
        factor = kwargs.pop("factor", 0.1)
        patience = kwargs.pop("patience", 10)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, **kwargs
        )

    elif scheduler_name == "exponential":
        gamma = kwargs.pop("gamma", 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma, **kwargs)

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            f"Supported schedulers are: cosine, step, plateau, exponential, none"
        )
