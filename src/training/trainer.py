"""Training loop and trainer class."""

import time
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow
import mlflow.pytorch
from ..utils.checkpoint import (
    save_history,
    ensure_dirs,
    count_parameters,
)
from ..utils.device import get_device
from ..utils.mlflow import log_params_from_config


class Trainer:
    """Trainer class for model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a Trainer that manages training and validation loops, history tracking,
        device placement, and optional learning-rate scheduling.

        Parameters:
            model: The PyTorch model to train.
            train_loader: DataLoader providing training batches.
            val_loader: DataLoader providing validation batches.
            criterion: Loss function used to compute training/validation loss.
            optimizer: Optimizer used to update model parameters.
            device: Device for computation; if None, a default device is selected automatically.
            scheduler: Optional learning-rate scheduler applied during training.
            config: The configuration dictionary.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else get_device()
        self.config = config

        self.model.to(self.device)

        self.history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """
        Performs one training epoch over the training DataLoader and updates the model parameters.

        Returns:
            epoch_loss (float): Average loss per sample over the epoch.
            epoch_acc (float): Accuracy percentage (0–100) over the epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Run one validation epoch over the validation DataLoader and compute average loss and accuracy.

        Returns:
            epoch_loss (float): Average loss per sample over the validation set.
            epoch_acc (float): Accuracy as a percentage (0.0–100.0) computed from model predictions.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({"loss": loss.item(), "acc": 100.0 * correct / total})

        if total == 0:
            return 0.0, 0.0
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        return epoch_loss, epoch_acc

    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        model_name: str = "model",
    ) -> Dict:
        """
        Run training for a specified number of epochs, track metrics, save the best model
        checkpoint, and persist training history.

        Parameters:
            num_epochs (int): Number of epochs to run.
            save_dir (str): Directory where the best model checkpoint will be saved (default "models").
            model_name (str): Base filename to use when saving the model and history (default "model").

        Returns:
            history (Dict): Dictionary with per-epoch lists ('train_loss', 'val_loss',
                'train_acc', 'val_acc', 'learning_rates') and metadata fields
                ('training_time', 'params', 'best_val_acc').
        """
        ensure_dirs([save_dir, "logs"])
        best_val_acc = -1.0
        start_time = time.time()

        with mlflow.start_run():
            log_params_from_config(self.config)

            for epoch in range(num_epochs):
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 50)

                # Train and validate
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate_epoch()

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Store history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_acc"].append(val_acc)
                self.history["learning_rates"].append(current_lr)

                # Log metrics to MLflow
                metrics = {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                }
                mlflow.log_metrics(metrics, step=epoch)

                # Print epoch summary
                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"Learning Rate: {current_lr:.6f}")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_path = f"{save_dir}/{model_name}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"[OK] Saved best model to {save_path}")

            # Log the best model as an artifact
            mlflow.pytorch.log_model(self.model, "model", registered_model_name=model_name)

            # Calculate total training time
            total_time = time.time() - start_time

            # Add metadata to history
            self.history["training_time"] = total_time
            self.history["params"] = count_parameters(self.model)
            self.history["best_val_acc"] = best_val_acc

            # Save history
            history_path = save_history(self.history, model_name)
            mlflow.log_artifact(history_path)

        print(f"\n{'='*50}")
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"{'='*50}")

        return self.history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    num_epochs: int = 10,
    lr: float = 5e-5,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run training with common defaults and return the training history.

    Parameters:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        model_name (str): Base name used when saving the best model and logs.
        num_epochs (int): Number of epochs to train.
        lr (float): Initial learning rate for the Adam optimizer.
        device (torch.device, optional): Device to run training on; if None, a default device is chosen.

    Returns:
        dict: Training history containing per-epoch losses, accuracies, learning rates, and metadata.
    """
    if device is None:
        device = get_device()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    return trainer.train(num_epochs=num_epochs, model_name=model_name)
