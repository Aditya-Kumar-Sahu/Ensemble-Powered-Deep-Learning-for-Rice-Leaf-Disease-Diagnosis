"""ResNet50 model architecture."""

import torch.nn as nn
import torchvision.models as models


def get_resnet50(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Create a ResNet-50 model configured for classification.

    Parameters:
        num_classes (int): Number of target output classes for the final layer.
        pretrained (bool): If True, initialize weights from ImageNet; otherwise random initialization.
        dropout (float): Accepted for API compatibility; not used by this function.

    Returns:
        nn.Module: ResNet-50 model with its final fully connected layer replaced to output `num_classes`.
    """
    if pretrained:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet50(weights=None)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
