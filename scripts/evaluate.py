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
from src.utils import set_seed, get_device, setup_logger, load_config


def main():
    """
    Run the evaluation workflow for rice leaf disease classification models using command-line arguments.

    Parses CLI options, loads configuration and dataset, evaluates either a single specified model or an ensemble of models, computes metrics, saves a confusion matrix image and a classification report to the output directory, and prints the classification report and model summary to stdout. Requires either the --model or --ensemble flag to be provided; uses --models-dir for checkpoints and --config for the YAML configuration.
    """
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

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.ensemble:
        parser.error("Either --model or --ensemble must be specified")

    # Load configuration
    model_config_path = (
        f"configs/model_configs/{args.model}.yaml" if args.model else None
    )
    config = load_config(model_config_path=model_config_path)

    # Set seed
    set_seed(config["seed"])

    # Setup logger
    logger = setup_logger()
    logger.info("Starting evaluation script")

    # Get device
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading dataset...")
    _, val_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        config=config,
    )
    num_classes = len(class_names)
    logger.info(f"Dataset loaded: {len(val_loader.dataset)} validation samples")
    logger.info(f"Number of classes: {num_classes}")

    # Create output directory
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(exist_ok=True)

    # Evaluate ensemble
    if args.ensemble:
        logger.info("Evaluating ensemble model...")
        model_names = ["mobilenetv2", "resnet50", "efficientnetb0"]

        y_true, y_pred = predict_ensemble(
            val_loader=val_loader,
            model_names=model_names,
            num_classes=num_classes,
            checkpoint_dir=config["output"]["models_dir"],
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
            save_path=output_dir / "confusion_matrix_ensemble.png",
        )

        # Generate classification report
        report = generate_classification_report(y_true, y_pred, class_names)
        print("\nClassification Report:")
        print(report)

        # Save report
        with open(output_dir / "classification_report_ensemble.txt", "w") as f:
            f.write(report)

        # Print model summary
        print_model_summary(model_names, log_folder=config["output"]["logs_dir"])

    # Evaluate single model
    elif args.model:
        logger.info(f"Evaluating model: {args.model}")

        # Load model
        model = get_model(args.model, num_classes, dropout=config["model"]["dropout"])
        checkpoint_path = Path(config["output"]["models_dir"]) / f"{args.model}.pth"
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
            save_path=output_dir / f"confusion_matrix_{args.model}.png",
        )

        # Generate classification report
        report = generate_classification_report(y_true, y_pred, class_names)
        print("\nClassification Report:")
        print(report)

        # Save report
        with open(output_dir / f"classification_report_{args.model}.txt", "w") as f:
            f.write(report)


if __name__ == "__main__":
    main()
