"""EfficientNet-B0 model architecture."""

import torch.nn as nn
import torchvision.models as models


def get_efficientnet_b0(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create an EfficientNet-B0 model for rice leaf disease classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
        
    Returns:
        EfficientNet-B0 model instance
    """
    if pretrained:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Update dropout if needed
    if hasattr(model.classifier[0], 'p'):
        model.classifier[0].p = dropout
    
    return model
