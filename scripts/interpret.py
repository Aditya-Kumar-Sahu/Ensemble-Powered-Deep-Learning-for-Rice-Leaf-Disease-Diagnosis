"""
Generate Grad-CAM visualizations for a given model and image.
"""

import argparse
import sys
from pathlib import Path
import torch
import cv2
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model  # noqa: E402
from src.evaluation import generate_gradcam_overlay  # noqa: E402
from src.utils import load_config  # noqa: E402
from src.data.augmentations import get_val_transforms  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for trained models.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "mobilenetv2", "efficientnetb0"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--output_path", type=str, default="results/grad_cam.jpg", help="Path to save the Grad-CAM overlay image."
    )

    args = parser.parse_args()

    # Load configuration
    model_config_path = f"configs/model_configs/{args.model}.yaml"
    config = load_config(model_config_path=model_config_path)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    val_transforms = get_val_transforms(config)
    img = np.array(Image.open(args.image_path).convert("RGB"))
    input_tensor = val_transforms(image=img)["image"].unsqueeze(0)

    # Load model
    num_classes = 15  # Assuming 15 classes for this project
    model = get_model(args.model, num_classes=num_classes)
    checkpoint_path = Path(config["output"]["models_dir"]) / f"{args.model}.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Define the target layer
    if args.model == "resnet50":
        target_layer = model.layer4[-1]
    elif args.model == "mobilenetv2":
        target_layer = model.features[-1]
    elif args.model == "efficientnetb0":
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Target layer not defined for model {args.model}")

    # Generate Grad-CAM overlay
    visualization = generate_gradcam_overlay(
        model=model, target_layer=target_layer, input_tensor=input_tensor, use_cuda=device.type == "cuda"
    )

    # Save the visualization
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), visualization)

    print(f"Grad-CAM visualization saved to: {args.output_path}")


if __name__ == "__main__":
    main()
