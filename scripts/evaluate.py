#!/usr/bin/env python3
"""
Evaluation script for rice leaf disease classification models.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models import get_model, predict_ensemble
from src.evaluation import (
    evaluate_model,
    predict_single,
    plot_confusion_matrix,
    generate_classification_report,
    print_model_summary,
)
from src.utils import set_seed, get_device, setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rice leaf disease classification models"
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
        default=None,
        help="Single model to evaluate (resnet50, mobilenetv2, efficientnetb0)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Evaluate ensemble of all models",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluation results",
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
        help="Device to use for evaluation",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.ensemble:
        parser.error("Either --model or --ensemble must be specified")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting evaluation script")
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
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
    logger.info(f"Dataset loaded: {len(val_loader.dataset)} validation samples")
    logger.info(f"Number of classes: {num_classes}")
    
    # Evaluate ensemble
    if args.ensemble:
        logger.info("Evaluating ensemble model...")
        model_names = ["mobilenetv2", "resnet50", "efficientnetb0"]
        
        y_true, y_pred = predict_ensemble(
            val_loader=val_loader,
            model_names=model_names,
            num_classes=num_classes,
            checkpoint_dir=args.models_dir,
            device=device,
            voting=config["ensemble"]["voting"],
        )
        
        # Evaluate
        metrics = evaluate_model(y_true, y_pred, class_names)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            save_path=f"{args.output_dir}/confusion_matrix_ensemble.png",
        )
        
        # Generate classification report
        report = generate_classification_report(y_true, y_pred, class_names)
        print("\nClassification Report:")
        print(report)
        
        # Save report
        with open(f"{args.output_dir}/classification_report_ensemble.txt", "w") as f:
            f.write(report)
        
        # Print model summary
        print_model_summary(model_names)
    
    # Evaluate single model
    elif args.model:
        logger.info(f"Evaluating model: {args.model}")
        
        # Load model
        model = get_model(args.model, num_classes)
        checkpoint_path = f"{args.models_dir}/{args.model}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Make predictions
        y_true, y_pred, y_probs = predict_single(model, val_loader, device)
        
        # Evaluate
        metrics = evaluate_model(y_true, y_pred, class_names, y_probs)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            y_true,
            y_pred,
            class_names,
            save_path=f"{args.output_dir}/confusion_matrix_{args.model}.png",
        )
        
        # Generate classification report
        report = generate_classification_report(y_true, y_pred, class_names)
        print("\nClassification Report:")
        print(report)
        
        # Save report
        with open(f"{args.output_dir}/classification_report_{args.model}.txt", "w") as f:
            f.write(report)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
