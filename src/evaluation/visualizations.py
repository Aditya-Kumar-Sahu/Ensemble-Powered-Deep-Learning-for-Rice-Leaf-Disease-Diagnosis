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
    Create and display a confusion matrix heatmap for the provided true and predicted labels.

    Parameters:
        y_true (np.ndarray): True class labels.
        y_pred (np.ndarray): Predicted class labels.
        class_names (List[str]): Ordered list of class names used as tick labels for both axes.
        save_path (Optional[str]): If provided, save the figure to this path; the plot is displayed regardless.
        figsize (tuple): Figure size passed to matplotlib.
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
    Plot one or more training metrics (train vs validation) over epochs.

    For each metric in `metrics`, looks for `train_<metric>` and `val_<metric>` keys in `history`
    and, if both are present, plots their values across epochs with labels, a title, legend,
    and grid. If `save_path` is provided, saves the figure to that path. The figure is
    displayed after plotting.

    Parameters:
        history (dict): Mapping containing training histories, expected keys like
            "train_loss", "val_loss", etc.
        metrics (List[str]): Metrics to plot; for each metric the function looks for
            `train_<metric>` and `val_<metric>` in `history`.
        save_path (Optional[str]): Path to save the generated figure; when omitted the
            figure is not saved.
        figsize (tuple): Matplotlib figure size.
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
    Plot a comparison of a specified training metric across multiple model histories.

    Parameters:
        model_names (List[str]): Identifiers of models whose histories will be compared.
        metric (str): Metric to plot; expected values are "loss" or "acc".
        log_folder (str): Directory where each model's history is stored and loaded from.
        save_path (Optional[str]): If provided, path to save the resulting figure.
        figsize (tuple): Figure size (width, height) in inches.
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
