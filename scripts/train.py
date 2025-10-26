#!/usr/bin/env python3
"""
Training script for rice leaf disease classification models.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import get_model
from src.training import Trainer, get_optimizer, get_scheduler
from src.utils import set_seed, get_device, setup_logger, ensure_dirs


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train rice leaf disease classification models"
    )
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
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup logger
    logger = setup_logger(log_file="logs/training.log")
    logger.info("Starting training script")
    logger.info(f"Configuration: {config}")
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directories
    ensure_dirs([args.output_dir, "logs"])
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        image_size=config["data"]["image_size"],
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        num_workers=config["data"]["num_workers"],
        seed=args.seed,
    )
    num_classes = len(class_names)
    logger.info(f"Dataset loaded: {len(train_loader.dataset)} training samples, "
                f"{len(val_loader.dataset)} validation samples")
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
