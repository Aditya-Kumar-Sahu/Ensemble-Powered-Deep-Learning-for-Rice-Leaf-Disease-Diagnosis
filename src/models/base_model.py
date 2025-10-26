"""Base model factory for creating different architectures."""

import torch.nn as nn
from typing import Optional

from .mobilenet import get_mobilenet_v2
from .resnet import get_resnet50
from .efficientnet import get_efficientnet_b0


MODEL_REGISTRY = {
    "mobilenetv2": get_mobilenet_v2,
    "mobilenet_v2": get_mobilenet_v2,
    "resnet50": get_resnet50,
    "efficientnetb0": get_efficientnet_b0,
    "efficientnet_b0": get_efficientnet_b0,
}


def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory function to create different model architectures.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
        
    Returns:
        PyTorch model instance
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported models are: {supported_models}"
        )
    
    model_fn = MODEL_REGISTRY[model_name_lower]
    return model_fn(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def list_available_models() -> list:
    """
    List all available model architectures.
    
    Returns:
        List of available model names
    """
    return list(MODEL_REGISTRY.keys())
