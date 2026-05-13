"""Metrics calculation utilities."""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def predict_single(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference with a single PyTorch model over a validation DataLoader
    and collect labels, predictions, and predicted probabilities.

    Returns:
        Tuple containing:
        - y_true (np.ndarray): Ground-truth labels with shape (N,).
        - y_pred (np.ndarray): Predicted class indices with shape (N,).
        - y_probs (np.ndarray): Predicted class probabilities with shape (N, num_classes).
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Predicting"):
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)

    return y_true, y_pred, y_probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute common classification metrics and optionally ROC-AUC (one-vs-rest).

    Parameters:
        y_true (np.ndarray): Ground-truth integer class labels.
        y_pred (np.ndarray): Predicted integer class labels.
        y_probs (Optional[np.ndarray]): Class probability estimates with shape (n_samples, n_classes).
            If provided, ROC-AUC (OvR) will be computed.
        num_classes (Optional[int]): Number of classes. If omitted and `y_probs` is provided, the
            number of classes is inferred from `y_true`.

    Returns:
        Dict[str, float]: Dictionary containing:
            - "accuracy": overall accuracy.
            - "precision_macro", "recall_macro", "f1_macro": macro-averaged precision, recall, and F1.
            - "precision_weighted", "recall_weighted", "f1_weighted": weighted averages for
              precision, recall, and F1.
            - "roc_auc_ovr" (optional): macro-averaged ROC-AUC computed in a one-vs-rest fashion
              when `y_probs` is supplied.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Calculate ROC-AUC if probabilities are provided
    if y_probs is not None:
        try:
            if num_classes is None:
                num_classes = len(np.unique(y_true))
            y_true_bin = np.eye(num_classes)[y_true]
            metrics["roc_auc_ovr"] = roc_auc_score(y_true_bin, y_probs, average="macro", multi_class="ovr")
        except Exception as e:
            print(f"Warning: Could not calculate ROC-AUC: {e}")

    return metrics


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Print a formatted evaluation report for classification predictions and return computed metrics.

    Parameters:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_names (list): Names of classes in the order corresponding to label indices.
        y_probs (Optional[np.ndarray]): Prediction probabilities or scores for each class;
            when provided, ROC-AUC (OvR) will be computed if possible.

    Returns:
        Dict[str, float]: Mapping of metric names to their values (includes accuracy, macro/weighted
            precision, recall, and F1). May include the key 'roc_auc_ovr' when `y_probs` is supplied
            and ROC-AUC computation succeeds.
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    # Calculate overall metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs, len(class_names))

    print("\nOverall Metrics:")
    print(f"  Accuracy     : {metrics['accuracy']:.4f}")
    print(f"  Precision    : {metrics['precision_macro']:.4f} (macro)")
    print(f"  Recall       : {metrics['recall_macro']:.4f} (macro)")
    print(f"  F1-score     : {metrics['f1_macro']:.4f} (macro)")

    if "roc_auc_ovr" in metrics:
        print(f"  ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")

    # Class-wise F1 scores
    print("\nClass-wise F1 Scores:")
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for cls, score in zip(class_names, class_f1):
        print(f"  {cls:<30}: {score:.4f}")

    print("=" * 60 + "\n")

    return metrics
