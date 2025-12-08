"""Report generation utilities."""

from typing import List, Dict
import numpy as np
from sklearn.metrics import classification_report

from ..utils.checkpoint import load_history


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Create a formatted classification report string containing per-class precision, recall, f1-score, and support.

    Parameters:
        y_true: Array of true class labels.
        y_pred: Array of predicted class labels.
        class_names: Ordered list of class names corresponding to label indices; used as target names in the report.

    Returns:
        A string with the classification report including per-class metrics and aggregated averages (accuracy, macro avg, weighted avg), formatted with four decimal digits and treating zero-division cases as 0.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    return report


def print_model_summary(
    model_names: List[str],
    log_folder: str = "logs",
) -> None:
    """
    Print a formatted summary table of trained models' parameters, training time, and best validation accuracy.

    For each model in `model_names`, attempts to load its training history from `log_folder` and prints a row containing the model name, parameter count, total training time in seconds, and best validation accuracy as a percentage. If a model's history file is missing, prints "N/A" for that model's fields.

    Parameters:
        model_names (List[str]): Names of the models to include in the summary.
        log_folder (str): Directory containing training history files (default "logs").
    """
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Parameters':>15} {'Training Time':>15} {'Best Val Acc':>15}"
    )
    print("-" * 80)

    for name in model_names:
        try:
            hist = load_history(name, folder=log_folder)
            params = hist.get("params", 0)
            training_time = hist.get("training_time", 0)
            best_val_acc = hist.get("best_val_acc", max(hist.get("val_acc", [0])))

            print(
                f"{name:<20} {params:>15,} {training_time:>14.2f}s "
                f"{best_val_acc:>14.2f}%"
            )
        except FileNotFoundError:
            print(f"{name:<20} {'N/A':>15} {'N/A':>15} {'N/A':>15}")

    print("=" * 80 + "\n")


def print_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
) -> None:
    """
    Print a formatted table of metrics for multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
    """
    if not metrics_dict:
        print("No metrics to display.")
        return

    # Get all unique metric keys
    all_keys = set()
    for metrics in metrics_dict.values():
        all_keys.update(metrics.keys())
    metric_keys = sorted(all_keys)

    # Print header
    print("\n" + "=" * 100)
    print("METRICS COMPARISON")
    print("=" * 100)

    header = f"{'Model':<20}"
    for key in metric_keys:
        header += f"{key:>15}"
    print(header)
    print("-" * 100)

    # Print rows
    for model_name, metrics in metrics_dict.items():
        row = f"{model_name:<20}"
        for key in metric_keys:
            value = metrics.get(key, 0.0)
            row += f"{value:>15.4f}"
        print(row)

    print("=" * 100 + "\n")
