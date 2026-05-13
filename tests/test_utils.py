"""Unit tests for utility functions."""

import pytest
import torch
import tempfile
import os
from pathlib import Path
import logging
import numpy as np

from src.utils.device import get_device, clear_gpu_cache
from src.utils.seed import set_seed
from src.utils.checkpoint import (
    ensure_dirs,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    save_history,
    load_history,
)
from src.utils.logging import setup_logger


def test_get_device():
    """Test device selection."""
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ["cuda", "cpu", "mps"]


def test_clear_gpu_cache():
    """Test GPU cache clearing."""
    # Should not raise any errors
    clear_gpu_cache()


def test_set_seed():
    """Test seed setting for reproducibility."""
    set_seed(42)

    # Generate random numbers
    rand1 = torch.rand(5)

    # Reset seed
    set_seed(42)

    # Generate again - should be the same
    rand2 = torch.rand(5)

    assert torch.allclose(rand1, rand2)


def test_ensure_dirs():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = [
            os.path.join(tmpdir, "dir1"),
            os.path.join(tmpdir, "dir2", "subdir"),
        ]

        ensure_dirs(dirs)

        for d in dirs:
            assert os.path.exists(d)
            assert os.path.isdir(d)


def test_save_and_load_checkpoint():
    """Test checkpoint saving and loading."""
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")

        # Save checkpoint
        metrics = {"loss": 0.5, "accuracy": 0.95}
        save_checkpoint(model, optimizer, epoch=10, metrics=metrics, filepath=checkpoint_path)

        assert os.path.exists(checkpoint_path)

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path)

        assert checkpoint["epoch"] == 10
        assert checkpoint["metrics"] == metrics
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint


def test_count_parameters():
    """Test parameter counting."""
    model = torch.nn.Linear(10, 5)

    # Linear layer has 10*5 + 5 = 55 parameters
    assert count_parameters(model) == 55


# ============================================================================
# Additional tests for logging.py and extended checkpoint functionality
# ============================================================================


class TestLogging:
    """Test suite for logging utilities."""

    def test_setup_logger_default(self):
        """Test logger setup with default parameters."""
        logger = setup_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "rice_leaf_disease"
        assert logger.level == logging.INFO

    def test_setup_logger_custom_name(self):
        """Test logger setup with custom name."""
        logger = setup_logger(name="test_logger")

        assert logger.name == "test_logger"

    def test_setup_logger_custom_level(self):
        """Test logger setup with custom level."""
        logger = setup_logger(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file(self):
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = setup_logger(log_file=log_file)

            # Log a message
            logger.info("Test message")

            # Check file was created
            assert os.path.exists(log_file)

            # Check content
            with open(log_file, "r") as f:
                content = f.read()
                assert "Test message" in content

            # Must close handlers or Windows denies deletion of tempdir
            logger = logging.getLogger("rice_leaf_disease")
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)

    def test_setup_logger_file_in_nested_dir(self):
        """Test logger setup with file in nested directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "logs", "nested", "test.log")
            logger = setup_logger(log_file=log_file)

            logger.info("Test message")

            # Check file and directories were created
            assert os.path.exists(log_file)

            # Must close handlers or Windows denies deletion of tempdir
            logger = logging.getLogger("rice_leaf_disease")
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)

    def test_logger_handlers_reset(self):
        """Test that logger handlers are properly reset."""
        logger1 = setup_logger(name="test_reset")
        initial_handlers = len(logger1.handlers)

        # Setup again - should reset handlers
        logger2 = setup_logger(name="test_reset")

        # Should have same number of handlers, not doubled
        assert len(logger2.handlers) == initial_handlers


class TestCheckpointHistory:
    """Test suite for checkpoint history functionality."""

    def test_save_and_load_history(self):
        """Test saving and loading training history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "train_loss": [0.5, 0.4, 0.3],
                "val_loss": [0.6, 0.5, 0.4],
                "train_acc": [80.0, 85.0, 90.0],
                "val_acc": [75.0, 80.0, 85.0],
            }

            save_history(history, "test_model", folder=tmpdir)

            # Check file exists
            history_file = os.path.join(tmpdir, "test_model_history.npy")
            assert os.path.exists(history_file)

            # Load and verify
            loaded_history = load_history("test_model", folder=tmpdir)

            assert loaded_history["train_loss"] == history["train_loss"]
            assert loaded_history["val_loss"] == history["val_loss"]
            assert loaded_history["train_acc"] == history["train_acc"]
            assert loaded_history["val_acc"] == history["val_acc"]

    def test_load_nonexistent_history(self):
        """Test loading history that doesn't exist raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_history("nonexistent_model", folder=tmpdir)

    def test_history_with_metadata(self):
        """Test saving history with additional metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "train_loss": [0.5, 0.4],
                "val_loss": [0.6, 0.5],
                "params": 1000000,
                "training_time": 120.5,
                "best_val_acc": 85.5,
            }

            save_history(history, "test_model", folder=tmpdir)
            loaded_history = load_history("test_model", folder=tmpdir)

            assert loaded_history["params"] == 1000000
            assert loaded_history["training_time"] == 120.5
            assert loaded_history["best_val_acc"] == 85.5


class TestCheckpointAdvanced:
    """Additional tests for checkpoint functionality."""

    def test_load_checkpoint_with_model_and_optimizer(self):
        """Test loading checkpoint and restoring model and optimizer state."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")

            # Save checkpoint
            metrics = {"loss": 0.5, "accuracy": 0.95}
            save_checkpoint(model, optimizer, epoch=10, metrics=metrics, filepath=checkpoint_path)

            # Create new model and optimizer
            new_model = torch.nn.Linear(10, 5)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            # Load checkpoint
            checkpoint = load_checkpoint(checkpoint_path, model=new_model, optimizer=new_optimizer)

            # Verify state was loaded
            assert checkpoint["epoch"] == 10
            assert checkpoint["metrics"]["loss"] == 0.5

    def test_load_checkpoint_with_device(self):
        """Test loading checkpoint with specific device."""
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "checkpoint.pth")

            save_checkpoint(model, optimizer, epoch=5, metrics={}, filepath=checkpoint_path)

            # Load to CPU explicitly
            checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

            assert checkpoint["epoch"] == 5

    def test_ensure_dirs_with_path_objects(self):
        """Test ensure_dirs with Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = [
                Path(tmpdir) / "dir1",
                Path(tmpdir) / "dir2" / "subdir",
            ]

            ensure_dirs(dirs)

            for d in dirs:
                assert d.exists()
                assert d.is_dir()

    def test_ensure_dirs_idempotent(self):
        """Test that ensure_dirs can be called multiple times safely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dirs = [os.path.join(tmpdir, "dir1")]

            ensure_dirs(dirs)
            assert os.path.exists(dirs[0])

            # Call again - should not raise error
            ensure_dirs(dirs)
            assert os.path.exists(dirs[0])

    def test_count_parameters_with_frozen_params(self):
        """Test parameter counting with frozen parameters."""
        model = torch.nn.Sequential(torch.nn.Linear(10, 20), torch.nn.Linear(20, 5))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        trainable_params = count_parameters(model)

        # Only second layer should be counted: 20*5 + 5 = 105
        assert trainable_params == 105


class TestSeedReproducibility:
    """Additional tests for seed setting and reproducibility."""

    def test_numpy_reproducibility(self):
        """Test numpy random number reproducibility."""
        set_seed(123)
        arr1 = np.random.rand(10)

        set_seed(123)
        arr2 = np.random.rand(10)

        assert np.allclose(arr1, arr2)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        rand1 = torch.rand(5)

        set_seed(123)
        rand2 = torch.rand(5)

        assert not torch.allclose(rand1, rand2)

    def test_python_random_reproducibility(self):
        """Test Python random module reproducibility."""
        import random

        set_seed(456)
        vals1 = [random.random() for _ in range(5)]  # noqa: S311

        set_seed(456)
        vals2 = [random.random() for _ in range(5)]  # noqa: S311

        assert vals1 == vals2
