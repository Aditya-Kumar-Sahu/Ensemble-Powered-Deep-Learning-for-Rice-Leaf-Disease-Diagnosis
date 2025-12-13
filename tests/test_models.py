"""Unit tests for model architectures."""

import pytest
import torch

from src.models import (
    get_model,
    get_mobilenet_v2,
    get_resnet50,
    get_efficientnet_b0,
    EnsembleModel,
)


@pytest.fixture
def num_classes():
    """Fixture for number of classes."""
    return 15


@pytest.fixture
def batch_size():
    """
    Provide the batch size used by the test fixtures.

    Returns:
        int: Batch size value (4).
    """
    return 4


@pytest.fixture
def input_tensor(batch_size):
    """
    Provide a random image batch tensor for model tests.

    Parameters:
        batch_size (int): Number of samples in the batch.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, 3, 224, 224) with values sampled from a standard normal distribution.
    """
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
    """
    Verify that get_model constructs each supported architecture and that a forward pass produces outputs with shape (batch_size, num_classes).

    Parameters:
        num_classes (int): Number of output classes used to construct the model.
        input_tensor (torch.Tensor): Input tensor passed to the model for the forward pass.
        batch_size (int): Expected batch size used to validate the output shape.
    """
    model_names = ["resnet50", "mobilenetv2", "efficientnetb0"]

    for model_name in model_names:
        model = get_model(model_name, num_classes)
        assert model is not None

        # Test forward pass
        output = model(input_tensor)
        assert output.shape == (batch_size, num_classes)


def test_invalid_model_name(num_classes):
    """
    Verify get_model raises ValueError when given an unknown model name.
    """
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


def test_ensemble_model(num_classes, input_tensor, batch_size):
    """Test the EnsembleModel."""
    model1 = get_model("resnet50", num_classes)
    model2 = get_model("mobilenetv2", num_classes)
    models = [model1, model2]

    # Test soft voting
    ensemble_soft = EnsembleModel(models, voting="soft")
    output_soft = ensemble_soft(input_tensor)
    assert output_soft.shape == (batch_size, num_classes)

    # Test hard voting
    ensemble_hard = EnsembleModel(models, voting="hard")
    output_hard = ensemble_hard(input_tensor)
    assert output_hard.shape == (batch_size, num_classes)
