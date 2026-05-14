"""
This module defines the data augmentation pipelines for the project.
It uses the 'albumentations' library to create flexible and powerful
data augmentation strategies for both training and validation.
"""

from typing import Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Returns the augmentation pipeline for the training dataset.

    Args:
        config (Dict[str, Any]): A dictionary containing configuration
                                  parameters, including image_size.

    Returns:
        A.Compose: The training augmentation pipeline.
    """
    image_size = config["data"]["image_size"]
    return A.Compose(
        [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.CoarseDropout(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(config: Dict[str, Any]) -> A.Compose:
    """
    Returns the augmentation pipeline for the validation dataset.

    Args:
        config (Dict[str, Any]): A dictionary containing configuration
                                  parameters, including image_size.

    Returns:
        A.Compose: The validation augmentation pipeline.
    """
    image_size = config["data"]["image_size"]
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
