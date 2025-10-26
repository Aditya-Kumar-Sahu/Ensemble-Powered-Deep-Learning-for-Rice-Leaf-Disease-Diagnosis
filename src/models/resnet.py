"""ResNet50 model architecture."""

import torch.nn as nn
import torchvision.models as models


def get_resnet50(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create a ResNet50 model for rice leaf disease classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization (currently unused for ResNet)
        
    Returns:
        ResNet50 model instance
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
