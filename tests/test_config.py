"""Unit tests for configuration utilities."""

import pytest
import yaml
from pathlib import Path
from src.utils.config import load_config, _recursive_merge

# Define paths for test configurations
BASE_CONFIG_PATH = Path("configs/test_base_config.yaml")
MODEL_CONFIG_PATH = Path("configs/model_configs/test_model_config.yaml")


@pytest.fixture(scope="module", autouse=True)
def setup_test_configs():
    """Create dummy config files for testing and clean up afterwards."""
    # Ensure config directories exist
    BASE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy base config
    base_content = {
        "data": {"image_size": 224, "batch_size": 32, "num_workers": 4},
        "training": {"epochs": 10, "lr": 0.001, "optimizer": "Adam"},
        "model": {"name": "base_model", "pretrained": True, "dropout": 0.2},
        "general_setting": "default_value",
    }
    with open(BASE_CONFIG_PATH, "w") as f:
        yaml.safe_dump(base_content, f)

    # Create dummy model config
    model_content = {
        "data": {"batch_size": 64},  # Override batch_size
        "training": {"lr": 0.0001, "scheduler": "CosineAnnealing"},  # Override lr, add scheduler
        "model": {"name": "resnet50", "pretrained": False},  # Override model name and pretrained
        "new_model_setting": "specific_value",  # Add new setting
    }
    with open(MODEL_CONFIG_PATH, "w") as f:
        yaml.safe_dump(model_content, f)

    yield  # Run tests

    # Clean up
    BASE_CONFIG_PATH.unlink(missing_ok=True)
    MODEL_CONFIG_PATH.unlink(missing_ok=True)
    # Clean up the model_configs directory if empty, or parent 'configs'
    try:
        MODEL_CONFIG_PATH.parent.rmdir()
    except OSError:
        pass  # Directory might not be empty if other configs exist
    try:
        BASE_CONFIG_PATH.parent.rmdir()
    except OSError:
        pass  # Directory might not be empty if other configs exist


class TestConfigUtils:
    """Test suite for configuration utility functions."""

    def test_recursive_merge(self):
        """Test recursive merging of dictionaries."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4, "e": 5}, "f": 6}
        _recursive_merge(base, override)
        assert base == {"a": 1, "b": {"c": 4, "d": 3, "e": 5}, "f": 6}

    def test_load_base_config(self):
        """Test loading only the base configuration."""
        config = load_config(base_config_path=str(BASE_CONFIG_PATH))
        assert config["data"]["image_size"] == 224
        assert config["training"]["epochs"] == 10
        assert config["model"]["name"] == "base_model"
        assert config["general_setting"] == "default_value"
        assert "scheduler" not in config["training"]  # Should not have model-specific keys

    def test_load_merged_config(self):
        """Test loading base and merging with model-specific configuration."""
        config = load_config(
            base_config_path=str(BASE_CONFIG_PATH),
            model_config_path=str(MODEL_CONFIG_PATH),
        )

        # Check overrides
        assert config["data"]["batch_size"] == 64  # Overridden
        assert config["training"]["lr"] == 0.0001  # Overridden
        assert config["model"]["name"] == "resnet50"  # Overridden
        assert config["model"]["pretrained"] is False  # Overridden

        # Check additions
        assert config["training"]["scheduler"] == "CosineAnnealing"  # Added
        assert config["new_model_setting"] == "specific_value"  # Added

        # Check retained base values
        assert config["data"]["image_size"] == 224  # Retained
        assert config["training"]["optimizer"] == "Adam"  # Retained
        assert config["general_setting"] == "default_value"  # Retained

    def test_load_config_nonexistent_base(self):
        """Test loading with a nonexistent base config file."""
        with pytest.raises(FileNotFoundError):
            load_config(base_config_path="nonexistent_base.yaml")

    def test_load_config_nonexistent_model(self):
        """Test loading with a nonexistent model config file (base should load)."""
        # Should load base config without error, just ignore model config path
        config = load_config(
            base_config_path=str(BASE_CONFIG_PATH),
            model_config_path="nonexistent_model.yaml",
        )
        assert config["data"]["image_size"] == 224
        assert "new_model_setting" not in config
