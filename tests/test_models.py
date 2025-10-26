"""Unit tests for model architectures."""

import pytest
import torch

from src.models import (
    get_model,
    get_mobilenet_v2,
    get_resnet50,
    get_efficientnet_b0,
)


@pytest.fixture
def num_classes():
    """Fixture for number of classes."""
    return 15


@pytest.fixture
def batch_size():
    """Fixture for batch size."""
    return 4


@pytest.fixture
def input_tensor(batch_size):
    """Fixture for input tensor."""
    return torch.randn(batch_size, 3, 224, 224)


def test_get_mobilenet_v2(num_classes, input_tensor, batch_size):
    """Test MobileNetV2 model creation."""
    model = get_mobilenet_v2(num_classes=num_classes)
    
    assert model is not None
    
    # Test forward pass
    output = model(input_tensor)
    assert output.shape == (batch_size, num_classes)


def test_get_resnet50(num_classes, input_tensor, batch_size):
    """Test ResNet50 model creation."""
    model = get_resnet50(num_classes=num_classes)
    
    assert model is not None
    
    # Test forward pass
    output = model(input_tensor)
    assert output.shape == (batch_size, num_classes)


def test_get_efficientnet_b0(num_classes, input_tensor, batch_size):
    """Test EfficientNet-B0 model creation."""
    model = get_efficientnet_b0(num_classes=num_classes)
    
    assert model is not None
    
    # Test forward pass
    output = model(input_tensor)
    assert output.shape == (batch_size, num_classes)


def test_get_model_factory(num_classes, input_tensor, batch_size):
    """Test model factory function."""
    model_names = ["resnet50", "mobilenetv2", "efficientnetb0"]
    
    for model_name in model_names:
        model = get_model(model_name, num_classes)
        assert model is not None
        
        # Test forward pass
        output = model(input_tensor)
        assert output.shape == (batch_size, num_classes)


def test_invalid_model_name(num_classes):
    """Test that invalid model name raises error."""
    with pytest.raises(ValueError):
        get_model("invalid_model", num_classes)


def test_model_with_pretrained(num_classes, input_tensor, batch_size):
    """Test model creation with pretrained weights."""
    model = get_model("resnet50", num_classes, pretrained=True)
    
    assert model is not None
    
    # Test forward pass
    output = model(input_tensor)
    assert output.shape == (batch_size, num_classes)


def test_model_with_dropout(num_classes, input_tensor, batch_size):
    """Test model creation with custom dropout."""
    model = get_model("mobilenetv2", num_classes, dropout=0.5)
    
    assert model is not None
    
    # Test forward pass
    output = model(input_tensor)
    assert output.shape == (batch_size, num_classes)
