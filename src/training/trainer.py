"""Training loop and trainer class."""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..utils.checkpoint import save_checkpoint, save_history, ensure_dirs, count_parameters
from ..utils.device import get_device


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
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else get_device()
        
        self.model.to(self.device)
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
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
            pbar.set_postfix({
                "loss": loss.item(),
                "acc": 100.0 * correct / total
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
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
                pbar.set_postfix({
                    "loss": loss.item(),
                    "acc": 100.0 * correct / total
                })
        
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
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            model_name: Name for saving the model
            
        Returns:
            Training history dictionary
        """
        ensure_dirs([save_dir, "logs"])
        best_val_acc = 0.0
        start_time = time.time()
        
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
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f"{save_dir}/{model_name}.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Saved best model to {save_path}")
        
        # Calculate total training time
        total_time = time.time() - start_time
        
        # Add metadata to history
        self.history["training_time"] = total_time
        self.history["params"] = count_parameters(self.model)
        self.history["best_val_acc"] = best_val_acc
        
        # Save history
        save_history(self.history, model_name)
        
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
    Simplified training function compatible with original notebook code.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        model_name: Name for saving the model
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Training history dictionary
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
