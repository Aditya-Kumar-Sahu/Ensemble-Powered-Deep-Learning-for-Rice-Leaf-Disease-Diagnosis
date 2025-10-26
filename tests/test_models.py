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


# ============================================================================
# Additional tests for ensemble.py and base_model.py
# ============================================================================

from src.models.ensemble import EnsembleModel
from src.models.base_model import list_available_models


class TestEnsembleModel:
    """Test suite for EnsembleModel class."""
    
    def test_ensemble_soft_voting(self, num_classes, input_tensor, batch_size):
        """Test ensemble with soft voting."""
        # Create multiple models
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        ensemble = EnsembleModel([model1, model2], voting="soft")
        
        # Test forward pass
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
    
    def test_ensemble_hard_voting(self, num_classes, input_tensor, batch_size):
        """Test ensemble with hard voting."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        ensemble = EnsembleModel([model1, model2], voting="hard")
        
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
    
    def test_ensemble_weighted_voting(self, num_classes, input_tensor, batch_size):
        """Test ensemble with weighted voting."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        weights = [0.6, 0.4]
        ensemble = EnsembleModel([model1, model2], voting="weighted", weights=weights)
        
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
    
    def test_ensemble_weighted_voting_default_weights(self, num_classes, input_tensor, batch_size):
        """Test ensemble with weighted voting using default equal weights."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        ensemble = EnsembleModel([model1, model2], voting="weighted")
        
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
        
        # Weights should be equal by default
        expected_weights = torch.tensor([0.5, 0.5])
        assert torch.allclose(ensemble.weights, expected_weights)
    
    def test_ensemble_invalid_weights_length(self, num_classes):
        """Test that mismatched weights length raises error."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        with pytest.raises(ValueError, match="Number of weights must match"):
            EnsembleModel([model1, model2], voting="weighted", weights=[0.5])
    
    def test_ensemble_invalid_weights_sum(self, num_classes):
        """Test that weights not summing to 1 raises error."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        
        with pytest.raises(ValueError, match=r"Weights must sum to 1\.0"):
            EnsembleModel([model1, model2], voting="weighted", weights=[0.3, 0.5])
    
    def test_ensemble_invalid_voting_strategy(self, num_classes, input_tensor):
        """Test that invalid voting strategy raises error."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        
        ensemble = EnsembleModel([model1], voting="invalid")
        
        with pytest.raises(ValueError, match="Unknown voting strategy"):
            ensemble(input_tensor)
    
    def test_ensemble_single_model(self, num_classes, input_tensor, batch_size):
        """Test ensemble with single model."""
        model = get_mobilenet_v2(num_classes=num_classes)
        
        ensemble = EnsembleModel([model], voting="soft")
        
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
    
    def test_ensemble_three_models(self, num_classes, input_tensor, batch_size):
        """Test ensemble with three different models."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model2 = get_resnet50(num_classes=num_classes)
        model3 = get_efficientnet_b0(num_classes=num_classes)
        
        ensemble = EnsembleModel([model1, model2, model3], voting="soft")
        
        output = ensemble(input_tensor)
        assert output.shape == (batch_size, num_classes)
    
    def test_ensemble_eval_mode(self, num_classes, input_tensor):
        """Test that ensemble sets models to eval mode."""
        model1 = get_mobilenet_v2(num_classes=num_classes)
        model1.train()  # Set to train mode
        
        ensemble = EnsembleModel([model1], voting="soft")
        
        # Forward pass should set model to eval
        _ = ensemble(input_tensor)
        
        # Model should remain in eval mode after forward
        assert not model1.training


class TestBaseModel:
    """Test suite for base_model module."""
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = list_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check expected models are present
        expected_models = ["mobilenetv2", "resnet50", "efficientnetb0"]
        for model_name in expected_models:
            assert model_name in models
    
    def test_model_registry_consistency(self):
        """Test that all models in registry can be instantiated."""
        available_models = list_available_models()
        num_classes = 10
        
        for model_name in available_models:
            model = get_model(model_name, num_classes)
            assert model is not None
    
    def test_model_name_case_insensitive(self, num_classes):
        """Test that model names are case-insensitive."""
        model1 = get_model("ResNet50", num_classes)
        model2 = get_model("resnet50", num_classes)
        model3 = get_model("RESNET50", num_classes)
        
        assert model1 is not None
        assert model2 is not None
        assert model3 is not None
    
    def test_model_alias_support(self, num_classes):
        """Test that model aliases work correctly."""
        # Test MobileNet aliases
        model1 = get_model("mobilenetv2", num_classes)
        model2 = get_model("mobilenet_v2", num_classes)
        
        assert model1 is not None
        assert model2 is not None
        
        # Test EfficientNet aliases
        model3 = get_model("efficientnetb0", num_classes)
        model4 = get_model("efficientnet_b0", num_classes)
        
        assert model3 is not None
        assert model4 is not None


class TestModelArchitectures:
    """Additional tests for individual model architectures."""
    
    def test_mobilenet_dropout_injection(self, num_classes):
        """Test that dropout is properly added to MobileNetV2."""
        model = get_mobilenet_v2(num_classes=num_classes, dropout=0.3)
        assert model is not None
    
    def test_efficientnet_dropout_update(self, num_classes):
        """Test EfficientNet dropout rate update."""
        model = get_efficientnet_b0(num_classes=num_classes, dropout=0.4)
        assert model is not None
        
        # Check classifier structure
        assert hasattr(model, 'classifier')
    
    def test_resnet_without_dropout(self, num_classes):
        """Test ResNet50 which doesn't use dropout parameter."""
        model = get_resnet50(num_classes=num_classes, dropout=0.5)
        assert model is not None
        
        # Check fc layer structure
        assert hasattr(model, 'fc')
        assert model.fc.out_features == num_classes
    
    def test_model_output_gradients(self, num_classes, input_tensor):
        """Test that models produce outputs with gradients."""
        model = get_mobilenet_v2(num_classes=num_classes)
        model.train()
        
        input_tensor.requires_grad = True
        output = model(input_tensor)
        
        # Compute loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert input_tensor.grad is not None
    
    def test_different_num_classes(self, input_tensor):
        """Test models with different number of classes."""
        for num_classes in [2, 10, 100, 1000]:
            model = get_mobilenet_v2(num_classes=num_classes)
            output = model(input_tensor)
            assert output.shape[1] == num_classes
    
    def test_model_pretrained_false(self, num_classes, input_tensor, batch_size):
        """Test models without pretrained weights."""
        for model_name in ["resnet50", "mobilenetv2", "efficientnetb0"]:
            model = get_model(model_name, num_classes, pretrained=False)
            output = model(input_tensor)
            assert output.shape == (batch_size, num_classes)