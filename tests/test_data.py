"""Unit tests for data loading and augmentation."""

import pytest
import torch
import tempfile
import os
from PIL import Image
import numpy as np
import cv2  # For reading images in the tests as the dataset does

from src.data.augmentations import get_train_transforms, get_val_transforms
from src.data.loaders import get_dataloaders, RiceLeafDiseaseDataset
from torchvision.datasets import ImageFolder  # Used internally by get_dataloaders


@pytest.fixture
def sample_image_np():
    """
    Create and return a random 224x224 RGB NumPy array image.
    This simulates cv2.imread output.
    """
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_dataset_dir():
    """Create a dummy dataset directory for testing dataloaders."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = os.path.join(tmpdir, "Rice Leaf Disease Images")

        # Create a nested structure to mimic the actual dataset
        dataset_path = os.path.join(data_root, "class_data")
        os.makedirs(dataset_path)

        class_names = ["class_1", "class_2", "class_3"]
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(dataset_path, class_name)
            os.makedirs(class_dir)
            for i in range(5):  # 5 images per class for a small test
                img = Image.new("RGB", (224, 224), color=(class_idx * 50, class_idx * 100, class_idx * 150))
                img.save(os.path.join(class_dir, f"img_{i}.jpg"))
        yield dataset_path


class TestAugmentations:
    """Test suite for data augmentation pipelines."""

    def test_get_train_transforms(self, sample_image_np):
        """Test training transforms apply and produce correct output format."""
        config = {"data": {"image_size": 224}}
        transform = get_train_transforms(config)

        assert transform is not None

        # Apply transform
        transformed = transform(image=sample_image_np)["image"]

        # Check output is a tensor
        assert isinstance(transformed, torch.Tensor)

        # Check shape (C, H, W)
        assert transformed.shape == (3, 224, 224)

    def test_get_val_transforms(self, sample_image_np):
        """Test validation transforms apply and produce correct output format."""
        config = {"data": {"image_size": 224}}
        transform = get_val_transforms(config)

        assert transform is not None

        # Apply transform
        transformed = transform(image=sample_image_np)["image"]

        # Check output is a tensor
        assert isinstance(transformed, torch.Tensor)

        # Check shape (C, H, W)
        assert transformed.shape == (3, 224, 224)

    def test_transforms_output_range(self, sample_image_np):
        """Test that normalization places pixel values in expected range."""
        config = {"data": {"image_size": 224}}
        train_transform = get_train_transforms(config)
        val_transform = get_val_transforms(config)

        train_transformed = train_transform(image=sample_image_np)["image"]
        val_transformed = val_transform(image=sample_image_np)["image"]

        # After normalization with mean/std, values should be roughly in [-2.x, 2.x] range
        assert train_transformed.min() >= -3.0
        assert train_transformed.max() <= 3.0
        assert val_transformed.min() >= -3.0
        assert val_transformed.max() <= 3.0


class TestDataloaders:
    """Test suite for data loading functionality."""

    def test_get_dataloaders(self, dummy_dataset_dir):
        """Test the get_dataloaders function with a dummy dataset."""
        config = {
            "data": {
                "batch_size": 2,
                "num_workers": 0,  # Use 0 workers for easier debugging in tests
                "val_split": 0.2,
                "image_size": 224,
            }
        }

        train_loader, val_loader, class_names = get_dataloaders(data_dir=dummy_dataset_dir, config=config)

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)

        # Check class names
        expected_class_names = sorted(["class_1", "class_2", "class_3"])
        assert class_names == expected_class_names

        # Total images: 3 classes * 5 images/class = 15
        total_images = 15
        val_size = int(config["data"]["val_split"] * total_images)
        train_size = total_images - val_size

        assert len(train_loader.dataset) == train_size
        assert len(val_loader.dataset) == val_size

        # Test fetching a batch from train_loader
        for images, labels in train_loader:
            assert images.shape == (config["data"]["batch_size"], 3, 224, 224)
            assert labels.shape == (config["data"]["batch_size"],)
            break  # Just check one batch

        # Test fetching a batch from val_loader
        for images, labels in val_loader:
            assert images.shape == (config["data"]["batch_size"], 3, 224, 224)
            assert labels.shape == (config["data"]["batch_size"],)
            break  # Just check one batch

    def test_rice_leaf_disease_dataset_getitem(self, dummy_dataset_dir):
        """Test the __getitem__ method of RiceLeafDiseaseDataset."""
        # Create a base ImageFolder for the dataset
        base_dataset = ImageFolder(dummy_dataset_dir)

        config = {"data": {"image_size": 224}}
        transform = get_val_transforms(config)  # Use val transforms for simplicity

        # Create an instance of our custom dataset
        custom_dataset = RiceLeafDiseaseDataset(base_dataset, transform=transform)

        # Fetch an item
        image_tensor, label = custom_dataset[0]

        assert isinstance(image_tensor, torch.Tensor)
        assert image_tensor.shape == (3, 224, 224)
        assert isinstance(label, int)
        assert 0 <= label < len(custom_dataset.classes)

    def test_rice_leaf_disease_dataset_subset_getitem(self, dummy_dataset_dir):
        """Test __getitem__ of RiceLeafDiseaseDataset when initialized with a Subset."""
        base_dataset = ImageFolder(dummy_dataset_dir)
        train_size = int(0.8 * len(base_dataset))
        val_size = len(base_dataset) - train_size

        train_subset, val_subset = torch.utils.data.random_split(base_dataset, [train_size, val_size])

        config = {"data": {"image_size": 224}}
        transform = get_val_transforms(config)

        # Initialize RiceLeafDiseaseDataset with a Subset
        custom_val_dataset = RiceLeafDiseaseDataset(val_subset, transform=transform)

        # Fetch an item from the subset-based dataset
        if len(custom_val_dataset) > 0:
            image_tensor, label = custom_val_dataset[0]

            assert isinstance(image_tensor, torch.Tensor)
            assert image_tensor.shape == (3, 224, 224)
            assert isinstance(label, int)
            assert 0 <= label < len(custom_val_dataset.classes)
        else:
            pytest.skip("Validation subset is empty, cannot test __getitem__.")
