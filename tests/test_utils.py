"""Unit tests for utility functions."""

import pytest
import torch
import tempfile
import os
from pathlib import Path

from src.utils.device import get_device, clear_gpu_cache
from src.utils.seed import set_seed
from src.utils.checkpoint import (
    ensure_dirs,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)


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
