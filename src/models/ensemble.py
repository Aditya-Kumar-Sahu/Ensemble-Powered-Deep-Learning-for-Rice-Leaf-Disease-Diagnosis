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
        Create an ensemble wrapper that combines multiple base models using a specified voting strategy.

        Parameters:
            models (List[nn.Module]): Base models to include in the ensemble; stored as an nn.ModuleList.
            voting (Literal["soft", "hard", "weighted"]): Voting strategy to aggregate model outputs. Supported values:
                "soft" — average per-model probability distributions;
                "hard" — majority vote on per-model class predictions;
                "weighted" — weighted average of per-model probability distributions.
            weights (List[float], optional): Per-model weights for "weighted" voting. If omitted when
                voting is "weighted", equal weights are used. When provided, the number of weights must
                equal the number of models and their sum must equal 1.0 within a tolerance of 1e-6.
                Validated weights are stored as a torch.float32 tensor. For non-"weighted" voting,
                the weights attribute is set to None.
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
        Compute ensemble logits from the registered base models using the configured voting strategy.

        Parameters:
            x (torch.Tensor): Input batch passed to each base model (batch dimension first). Each base model must accept x and produce logits over classes.

        Returns:
            torch.Tensor: Logits with shape [batch_size, num_classes]. For "soft" and "weighted" voting this is the log of the averaged (or weighted-averaged) class probabilities; for "hard" voting this is the log of a one-hot encoding of the majority-vote class.

        Raises:
            ValueError: If self.voting is not "soft", "weighted", or "hard".
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
    Load and return models instantiated for the given names by restoring their checkpoints.

    Parameters:
        model_names (List[str]): Names of models to instantiate and load.
        num_classes (int): Number of output classes for each model constructor.
        checkpoint_dir (str): Directory containing model checkpoint files named "<model_name>.pth".
        device (torch.device): Device to place each loaded model on.

    Returns:
        List[nn.Module]: List of models with their state restored, moved to `device`, and set to evaluation mode.
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
    Generate ensemble predictions for a validation dataset using soft, hard, or weighted voting.

    Parameters:
        val_loader (DataLoader): Validation data loader yielding (inputs, labels) batches.
        model_names (List[str]): Names of base models; each name is used to load a corresponding checkpoint file from checkpoint_dir.
        num_classes (int): Number of output classes for each model.
        checkpoint_dir (str): Directory containing model checkpoint files (default: "models").
        device (torch.device | None): Device to run inference on; if None, CUDA is used if available, otherwise CPU.
        voting (str): Voting strategy to combine model outputs. One of "soft", "hard", or "weighted".
            - "soft": average per-model softmax probabilities.
            - "hard": majority vote on per-model argmax predictions.
            - "weighted": weighted sum of per-model softmax probabilities using `weights`.
        weights (List[float] | None): Per-model weights for "weighted" voting. Must have length equal to the number of models and sum to 1.0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple (true_labels, predictions) where both are 1-D NumPy arrays of shape (N,) containing ground-truth labels and ensemble-predicted class indices for all samples in val_loader.
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
