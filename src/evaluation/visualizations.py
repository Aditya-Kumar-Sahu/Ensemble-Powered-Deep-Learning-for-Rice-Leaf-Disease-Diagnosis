"""Visualization utilities for evaluation."""

from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from ..utils.checkpoint import load_history


COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_training_history(
    history: dict,
    metrics: List[str] = ["loss", "acc"],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history and val_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            
            ax.plot(epochs, history[train_key], "b-", label="Train", linewidth=2)
            ax.plot(epochs, history[val_key], "r--", label="Validation", linewidth=2)
            
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f"{metric.capitalize()} over Epochs", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training history to {save_path}")
    
    plt.show()


def compare_models(
    model_names: List[str],
    metric: str = "loss",
    log_folder: str = "logs",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """
    Compare training logs of multiple models.
    
    Args:
        model_names: List of model names
        metric: Metric to compare ("loss" or "acc")
        log_folder: Directory containing training logs
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for i, name in enumerate(model_names):
        try:
            hist = load_history(name, folder=log_folder)
            color = COLORS[i % len(COLORS)]
            
            train_key = f"train_{metric}"
            val_key = f"val_{metric}"
            
            if train_key in hist:
                epochs = range(1, len(hist[train_key]) + 1)
                plt.plot(
                    epochs,
                    hist[train_key],
                    label=f"{name} - train",
                    color=color,
                    linewidth=2,
                )
            
            if val_key in hist:
                epochs = range(1, len(hist[val_key]) + 1)
                plt.plot(
                    epochs,
                    hist[val_key],
                    linestyle="--",
                    label=f"{name} - val",
                    color=color,
                    linewidth=2,
                )
        except FileNotFoundError:
            print(f"Warning: History not found for {name}")
    
    plt.title(f"Model {metric.capitalize()} Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    ylabel = "Loss" if metric == "loss" else "Accuracy (%)"
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model comparison to {save_path}")
    
    plt.show()
