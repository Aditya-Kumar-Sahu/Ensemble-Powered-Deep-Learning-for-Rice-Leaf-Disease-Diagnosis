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
    Create a registered model architecture by name.

    Args:
        model_name (str): Case-insensitive name of the model architecture to instantiate (e.g., "resnet50", "mobilenetv2").
        num_classes (int): Number of output classes for the model head.
        pretrained (bool): Whether to load pretrained weights.
        dropout (float): Dropout probability applied to the model's classifier head.

    Returns:
        nn.Module: Instantiated PyTorch model configured with the given parameters.

    Raises:
        ValueError: If `model_name` is not found among available models.
    """
    model_name_lower = model_name.lower()

    if model_name_lower not in MODEL_REGISTRY:
        supported_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unsupported model: {model_name}. " f"Supported models are: {supported_models}")

    model_fn = MODEL_REGISTRY[model_name_lower]
    return model_fn(num_classes=num_classes, pretrained=pretrained, dropout=dropout)


def list_available_models() -> list:
    """
    List available model architecture names.

    Returns:
        list: Available model names as strings.
    """
    return list(MODEL_REGISTRY.keys())
