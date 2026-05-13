#!/usr/bin/env python3
"""
Training script for rice leaf disease classification models.
"""

import argparse
import sys
from pathlib import Path
import torch
import mlflow

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders  # noqa: E402
from src.models import get_model  # noqa: E402
from src.training import Trainer, get_optimizer, get_scheduler  # noqa: E402
from src.utils import set_seed, get_device, setup_logger, ensure_dirs, load_config  # noqa: E402


def main():
    """
    Orchestrate end-to-end training of a rice leaf disease classification model using
    CLI arguments and a YAML configuration.

    Loads configuration, applies command-line overrides for epochs, batch size, and learning rate,
    initializes randomness and logging, selects the compute device, ensures output directories,
    prepares data loaders and model, constructs optimizer, scheduler, and loss, runs the training
    loop via Trainer, and logs final metrics and model save location.
    """
    parser = argparse.ArgumentParser(description="Train rice leaf disease classification models")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenetv2", "efficientnetb0"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Path to save trained models",
    )

    args = parser.parse_args()

    # Load configuration
    model_config_path = f"configs/model_configs/{args.model}.yaml"
    config = load_config(model_config_path=model_config_path)

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Setup logger
    log_path = Path(config["output"]["logs_dir"]) / "training.log"
    logger = setup_logger(log_file=log_path)
    logger.info("Starting training script")
    logger.info(f"Configuration: {config}")

    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Set MLFlow experiment
    if "mlflow" in config and "experiment_name" in config["mlflow"]:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Create output directories
    output_dir = Path(config["output"]["models_dir"])
    ensure_dirs([output_dir, log_path.parent])

    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        config=config,
    )
    num_classes = len(class_names)
    logger.info(
        f"Dataset loaded: {len(train_loader.dataset)} training samples, " f"{len(val_loader.dataset)} validation samples"
    )
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")

    # Create model
    logger.info(f"Creating model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create optimizer
    optimizer = get_optimizer(
        model=model,
        optimizer_name=config["training"]["optimizer"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=config["training"]["scheduler"],
        num_epochs=config["training"]["num_epochs"],
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
        config=config,
    )

    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_dir=args.output_dir,
        model_name=args.model,
    )

    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {history['best_val_acc']:.2f}%")
    logger.info(f"Model saved to {args.output_dir}/{args.model}.pth")


if __name__ == "__main__":
    main()
