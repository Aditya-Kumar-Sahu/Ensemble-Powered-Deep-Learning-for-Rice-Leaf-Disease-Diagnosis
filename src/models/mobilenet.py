"""MobileNetV2 model architecture."""

import torch.nn as nn
import torchvision.models as models


def get_mobilenet_v2(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create a MobileNetV2 model for rice leaf disease classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
        
    Returns:
        MobileNetV2 model instance
    """
    if pretrained:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    else:
        model = models.mobilenet_v2(weights=None)
    
    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    # Add dropout
    if dropout > 0:
        model.features[17].add_module("dropout", nn.Dropout(dropout))
    
    return model
