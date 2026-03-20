"""MLflow utility functions."""

import mlflow
from typing import Dict, Any


def log_params_from_config(config: Dict[str, Any]) -> None:
    """
    Logs a nested dictionary of parameters to MLflow by flattening it.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
    """

    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    if config is not None:
        mlflow.log_params(flatten_dict(config))
