"""Ensemble model for combining multiple models."""

from typing import List, Dict, Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .base_model import get_model


class EnsembleModel(nn.Module):
    """
    Ensemble model that combines predictions from multiple base models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        voting: Literal["soft", "hard", "weighted"] = "soft",
        weights: List[float] = None,
    ):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of base models
            voting: Voting strategy ("soft", "hard", or "weighted")
            weights: Optional weights for weighted voting (must sum to 1.0)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.voting = voting
        
        if voting == "weighted":
            if weights is None:
                # Default to equal weights
                weights = [1.0 / len(models)] * len(models)
            elif len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            elif abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
            self.weights = torch.tensor(weights, dtype=torch.float32)
        else:
            self.weights = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble predictions
        """
        outputs = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)
        
        outputs = torch.stack(outputs)  # [num_models, batch_size, num_classes]
        
        if self.voting == "soft":
            # Average probabilities
            probs = torch.softmax(outputs, dim=2)
            ensemble_probs = probs.mean(dim=0)
            return torch.log(ensemble_probs + 1e-10)  # Convert back to logits
        
        elif self.voting == "weighted":
            # Weighted average of probabilities
            probs = torch.softmax(outputs, dim=2)
            weights = self.weights.view(-1, 1, 1).to(probs.device)
            ensemble_probs = (probs * weights).sum(dim=0)
            return torch.log(ensemble_probs + 1e-10)
        
        elif self.voting == "hard":
            # Majority voting
            preds = torch.argmax(outputs, dim=2)
            ensemble_pred = torch.mode(preds, dim=0).values
            num_classes = outputs.shape[2]
            # Convert to one-hot and then to logits
            one_hot = F.one_hot(ensemble_pred, num_classes=num_classes).float()
            return torch.log(one_hot + 1e-10)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting}")


def load_ensemble_models(
    model_names: List[str],
    num_classes: int,
    checkpoint_dir: str,
    device: torch.device,
) -> List[nn.Module]:
    """
    Load multiple models from checkpoints.
    
    Args:
        model_names: List of model names
        num_classes: Number of output classes
        checkpoint_dir: Directory containing model checkpoints
        device: Device to load models to
        
    Returns:
        List of loaded models
    """
    models = []
    
    for model_name in model_names:
        model = get_model(model_name, num_classes)
        checkpoint_path = f"{checkpoint_dir}/{model_name}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
    
    return models


def predict_ensemble(
    val_loader: DataLoader,
    model_names: List[str],
    num_classes: int,
    checkpoint_dir: str = "models",
    device: torch.device = None,
    voting: str = "soft",
    weights: List[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using an ensemble of models.
    
    Args:
        val_loader: Validation data loader
        model_names: List of model names
        num_classes: Number of output classes
        checkpoint_dir: Directory containing model checkpoints
        device: Device to run inference on
        voting: Voting strategy
        weights: Optional weights for weighted voting
        
    Returns:
        Tuple of (true_labels, predictions)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    models = load_ensemble_models(model_names, num_classes, checkpoint_dir, device)
    
    # Create ensemble
    ensemble = EnsembleModel(models, voting=voting, weights=weights)
    ensemble.to(device)
    ensemble.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Ensemble Prediction"):
            inputs = inputs.to(device)
            
            if voting == "soft" or voting == "weighted":
                # Collect probabilities from all models
                probs = []
                for model in models:
                    output = model(inputs)
                    prob = F.softmax(output, dim=1)
                    probs.append(prob)
                
                probs = torch.stack(probs)  # [num_models, batch_size, num_classes]
                
                if voting == "soft":
                    avg_prob = probs.mean(dim=0)
                else:  # weighted
                    weights_tensor = torch.tensor(weights, device=device).view(-1, 1, 1)
                    avg_prob = (probs * weights_tensor).sum(dim=0)
                
                preds = torch.argmax(avg_prob, dim=1)
            
            else:  # hard voting
                votes = []
                for model in models:
                    output = model(inputs)
                    pred = torch.argmax(output, dim=1)
                    votes.append(pred)
                
                votes = torch.stack(votes)  # [num_models, batch_size]
                preds = torch.mode(votes, dim=0).values
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_labels), np.array(all_preds)
