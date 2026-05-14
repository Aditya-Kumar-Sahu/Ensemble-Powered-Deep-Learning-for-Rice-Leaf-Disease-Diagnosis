"""
This module defines the dataset and dataloader functionality.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np

from .augmentations import get_train_transforms, get_val_transforms


class RiceLeafDiseaseDataset(Dataset):
    """Custom dataset for rice leaf disease classification, compatible with Subsets."""

    def __init__(self, dataset, transform=None):
        # 'dataset' here can be an ImageFolder or a Subset thereof
        """
        Initialize the dataset wrapper that applies optional image transforms and exposes class names.
        
        Parameters:
            dataset (Dataset or Subset): A torch dataset (typically `torchvision.datasets.ImageFolder`) or a `torch.utils.data.Subset` produced by `random_split`. If a `Subset` is provided, its underlying original dataset is used to obtain class names.
            transform (callable, optional): An augmentation/transform callable (e.g., an Albumentations transform) that accepts a NumPy image via the keyword `image` and returns a dict containing the transformed image under the `"image"` key.
        """
        self.dataset = dataset
        self.transform = transform

        # If it's a Subset, the classes are on the original dataset
        if isinstance(self.dataset, torch.utils.data.Subset):
            self.classes = self.dataset.dataset.classes
        else:
            self.classes = self.dataset.classes

    def __len__(self):
        """
        Get the number of samples in the wrapped dataset.
        
        Returns:
            int: Number of samples in the wrapped dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieve image and label using the dataset's (or Subset's) __getitem__
        # This correctly handles mapping from Subset index to original dataset
        """
        Return the image and label for the given index, converting the stored PIL image to a NumPy array and applying the optional transform.
        
        If `self.transform` is provided it is called as `self.transform(image=...)` and must return a dict containing the transformed image under the "image" key.
        
        Parameters:
            idx (int): Index into the wrapped dataset or subset.
        
        Returns:
            tuple: (image, label) where `image` is a NumPy ndarray (transformed if a transform is set) and `label` is the class index.
        """
        img_pil, label = self.dataset[idx]  # This returns PIL Image by default from ImageFolder

        # Convert PIL Image to numpy array for albumentations
        image = np.array(img_pil)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_dataloaders(data_dir, config):
    """
    Create training and validation PyTorch DataLoader objects and return class names.
    
    Parameters:
        data_dir (str or os.PathLike): Path to the dataset root structured for torchvision.datasets.ImageFolder (class subdirectories).
        config (dict): Configuration mapping containing at least the following keys under "data":
            - "val_split": fraction of the dataset to use for validation (float between 0 and 1).
            - "batch_size": batch size for both DataLoaders (int).
            - "num_workers": number of worker processes for both DataLoaders (int).
            Other config entries may be used by the train/validation transform builders.
    
    Returns:
        tuple: (train_loader, val_loader, class_names)
            - train_loader: DataLoader that yields batches of (image, label) from the training split with training transforms applied.
            - val_loader: DataLoader that yields batches of (image, label) from the validation split with validation transforms applied.
            - class_names: list of class folder names in the dataset, ordered by class index.
    """
    # ImageFolder with a dummy transform for initial loading, actual transforms applied later
    # The default ImageFolder transform would convert to Tensor, which albumentations doesn't expect
    base_dataset = ImageFolder(data_dir, transform=None)

    # Get class names
    class_names = base_dataset.classes

    # Get transformations
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config)

    # Split the dataset
    val_split = config["data"]["val_split"]
    val_size = int(val_split * len(base_dataset))
    train_size = len(base_dataset) - val_size

    # Ensure reproducibility for random_split if needed, though seed is set globally
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    # Create custom datasets with the appropriate transforms, passing the subsets directly
    train_dataset = RiceLeafDiseaseDataset(train_subset, transform=train_transforms)
    val_dataset = RiceLeafDiseaseDataset(val_subset, transform=val_transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"]
    )

    return train_loader, val_loader, class_names
