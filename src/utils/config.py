"""Configuration loading and management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(base_config_path: str = "configs/base_config.yaml", model_config_path: str = None) -> Dict[str, Any]:
    """
    Loads a base YAML configuration and merges a model-specific configuration on top.

    Args:
        base_config_path (str): Path to the base configuration file.
        model_config_path (str, optional): Path to the model-specific
                                            configuration file. Defaults to None.

    Returns:
        Dict[str, Any]: The merged configuration dictionary.
    """
    with open(base_config_path, "r") as f:
        config = yaml.safe_load(f)

    if model_config_path:
        model_path = Path(model_config_path)
        if model_path.exists():
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
            _recursive_merge(config, model_config)
        else:
            print(f"Warning: Model config {model_config_path} not found. Proceeding with base config only.")

    return config


def _recursive_merge(base_dict: Dict, new_dict: Dict) -> None:
    """
    Recursively merges the new_dict into the base_dict.

    Args:
        base_dict (Dict): The base dictionary to merge into.
        new_dict (Dict): The new dictionary with values to merge.
    """
    for key, value in new_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _recursive_merge(base_dict[key], value)
        else:
            base_dict[key] = value
