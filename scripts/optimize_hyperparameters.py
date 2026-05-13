"""
Hyperparameter optimization script using Optuna.
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, Any
import optuna
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders  # noqa: E402
from src.models import get_model  # noqa: E402
from src.training import Trainer, get_optimizer, get_scheduler  # noqa: E402
from src.utils import set_seed, get_device, load_config  # noqa: E402

# Global variables for data and number of classes
DATA_DIR = ""
NUM_CLASSES = 0
CLASS_NAMES: list[str] = []
BASE_CONFIG: Dict[str, Any] = {}


def objective(trial: optuna.Trial) -> float:
    """
    The objective function for Optuna to optimize.
    A single trial consists of a full training run with a set of hyperparameters.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    # Create a trial-specific config by overriding the base config
    trial_config = copy.deepcopy(BASE_CONFIG)
    trial_config["training"]["learning_rate"] = lr
    trial_config["model"]["dropout"] = dropout
    trial_config["training"]["optimizer"] = optimizer_name
    trial_config["training"]["weight_decay"] = weight_decay

    # Set seed for reproducibility within a trial
    set_seed(trial_config["seed"])

    # Get device
    device = get_device()

    # Load data
    train_loader, val_loader, _ = get_dataloaders(data_dir=DATA_DIR, config=trial_config)

    # Create model
    model = get_model(
        model_name=trial_config["model"]["name"],
        num_classes=NUM_CLASSES,
        pretrained=trial_config["model"]["pretrained"],
        dropout=trial_config["model"]["dropout"],
    )

    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=trial_config["training"]["optimizer"],
        learning_rate=trial_config["training"]["learning_rate"],
        weight_decay=trial_config["training"]["weight_decay"],
    )

    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=trial_config["training"]["scheduler"],
        num_epochs=trial_config["training"]["num_epochs"],
    )

    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=trial_config,
    )

    # Train model
    history = trainer.train(
        num_epochs=trial_config["training"]["num_epochs"],
        save_dir=str(Path(trial_config["output"]["models_dir"]) / "optimization"),
        model_name=f"{trial_config['model']['name']}_trial_{trial.number}",
    )

    return history["best_val_acc"]


def main():
    global DATA_DIR, NUM_CLASSES, CLASS_NAMES, BASE_CONFIG

    parser = argparse.ArgumentParser(description="Hyperparameter optimization for models.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument(
        "--model", type=str, required=True, choices=["resnet50", "mobilenetv2", "efficientnetb0"], help="Model to optimize."
    )
    parser.add_argument("--n-trials", type=int, default=20, help="Number of optimization trials.")

    args = parser.parse_args()

    DATA_DIR = args.data_dir

    # Load base and model-specific configs
    model_config_path = f"configs/model_configs/{args.model}.yaml"
    BASE_CONFIG = load_config(model_config_path=model_config_path)

    # Get number of classes from a preliminary dataloader
    # This is a bit inefficient but necessary to configure the model correctly
    _, _, CLASS_NAMES = get_dataloaders(data_dir=DATA_DIR, config=BASE_CONFIG)
    NUM_CLASSES = len(CLASS_NAMES)

    # Create study
    study = optuna.create_study(
        study_name=f"{args.model}_optimization",
        direction="maximize",
        storage=f"sqlite:///optuna_{args.model}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
