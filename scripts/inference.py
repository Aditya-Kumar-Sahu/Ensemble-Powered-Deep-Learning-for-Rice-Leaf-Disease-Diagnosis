#!/usr/bin/env python3
"""
Inference script for rice leaf disease classification.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model
from src.data.augmentations import get_val_transforms
from src.utils import get_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def predict_image(
    image_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    class_names: list,
) -> dict:
    """
    Predict disease class for a single image.
    
    Args:
        image_path: Path to image file
        model: Trained model
        transform: Image transform
        device: Device to run inference on
        class_names: List of class names
        
    Returns:
        Dictionary with prediction results
    """
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
    
    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))
    top5_predictions = [
        {
            "class": class_names[idx],
            "confidence": prob.item(),
        }
        for idx, prob in zip(top5_indices, top5_probs)
    ]
    
    return {
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "top5_predictions": top5_predictions,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Perform inference on rice leaf disease images"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenetv2", "efficientnetb0"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: models/{model}.pth)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class names",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set checkpoint path
    if args.checkpoint is None:
        args.checkpoint = f"models/{args.model}.pth"
    
    # Load model
    print(f"Loading model: {args.model}")
    num_classes = len(args.classes)
    model = get_model(args.model, num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()
    
    # Get transform
    transform = get_val_transforms(config["data"]["image_size"])
    
    # Make prediction
    print(f"Predicting image: {args.image}")
    result = predict_image(
        args.image,
        model,
        transform,
        device,
        args.classes,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nPredicted Class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
    
    print("\nTop 5 Predictions:")
    for i, pred in enumerate(result["top5_predictions"], 1):
        print(f"  {i}. {pred['class']:<30} {pred['confidence']:.4f} ({pred['confidence']*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
