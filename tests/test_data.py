"""Unit tests for data loading and augmentation."""

import pytest
import torch
from torchvision import transforms
import tempfile
import os
from PIL import Image
import numpy as np

from src.data.augmentations import (
    get_train_transforms,
    get_val_transforms,
    get_advanced_train_transforms,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def test_get_train_transforms(sample_image):
    """Test training transforms."""
    transform = get_train_transforms(image_size=224)
    
    assert transform is not None
    assert isinstance(transform, transforms.Compose)
    
    # Apply transform
    transformed = transform(sample_image)
    
    # Check output is a tensor
    assert isinstance(transformed, torch.Tensor)
    
    # Check shape
    assert transformed.shape == (3, 224, 224)
    
    # Check normalization (values should be roughly in [-2, 2] range)
    assert transformed.min() >= -3.0
    assert transformed.max() <= 3.0


def test_get_val_transforms(sample_image):
    """Test validation transforms."""
    transform = get_val_transforms(image_size=224)
    
    assert transform is not None
    assert isinstance(transform, transforms.Compose)
    
    # Apply transform
    transformed = transform(sample_image)
    
    # Check output is a tensor
    assert isinstance(transformed, torch.Tensor)
    
    # Check shape
    assert transformed.shape == (3, 224, 224)


def test_get_advanced_train_transforms(sample_image):
    """Test advanced training transforms."""
    transform = get_advanced_train_transforms(image_size=224)
    
    assert transform is not None
    assert isinstance(transform, transforms.Compose)
    
    # Apply transform
    transformed = transform(sample_image)
    
    # Check output is a tensor
    assert isinstance(transformed, torch.Tensor)
    
    # Check shape
    assert transformed.shape == (3, 224, 224)


def test_transform_different_sizes():
    """Test transforms with different image sizes."""
    img_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    for size in [128, 224, 256]:
        transform = get_train_transforms(image_size=size)
        transformed = transform(img)
        
        assert transformed.shape == (3, size, size)


def test_transform_reproducibility(sample_image):
    """Test that validation transforms are deterministic."""
    transform = get_val_transforms(image_size=224)
    
    # Apply transform twice
    transformed1 = transform(sample_image)
    transformed2 = transform(sample_image)
    
    # Should be identical for validation transforms
    assert torch.allclose(transformed1, transformed2)
