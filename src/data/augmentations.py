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
    Builds an augmentation pipeline for training images.
    
    Parameters:
        config (Dict[str, Any]): Configuration dictionary; must contain `config["data"]["image_size"]`
            which specifies the target square image size.
    
    Returns:
        A.Compose: An Albumentations Compose pipeline that applies, in order, random resized crop
            (scale 0.8–1.0), horizontal and vertical flips, rotation (±30°), color jitter,
            coarse dropout, ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            and converts the result to a tensor with ToTensorV2.
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
    Builds the image transformation pipeline used for validation.
    
    Parameters:
        config (Dict[str, Any]): Configuration dictionary; must contain `config["data"]["image_size"]` specifying the target height/width.
    
    Returns:
        albumentations.Compose: A composition that resizes images to (image_size, image_size), normalizes using ImageNet mean/std, and converts images to tensors.
    """
    image_size = config["data"]["image_size"]
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )
