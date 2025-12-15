"""
Gradio application for interactive Rice Leaf Disease Classification.
"""

import sys
from pathlib import Path
import torch
import gradio as gr
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model
from src.utils import load_config
from src.data.augmentations import get_val_transforms

# --- Model and Config Loading ---
# This section runs once at startup

MODEL_NAME = "resnet50"
config = load_config(model_config_path=f"configs/model_configs/{MODEL_NAME}.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = Path(config["output"]["models_dir"]) / f"{MODEL_NAME}.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")

# This should be dynamically loaded or stored in a config
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]  # Example, please update

NUM_CLASSES = len(CLASS_NAMES)
model = get_model(MODEL_NAME, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

val_transforms = get_val_transforms(config)

# --- Prediction Function ---


def predict(image: np.ndarray) -> dict:
    """
    Takes a NumPy image, preprocesses it, and returns a dictionary of class probabilities.

    Args:
        image (np.ndarray): The input image from the Gradio interface.

    Returns:
        dict: A dictionary mapping class names to their confidence scores.
    """
    if image is None:
        raise gr.Error("No image uploaded. Please upload an image to get a prediction.")

    try:
        # Preprocess the image
        input_tensor = val_transforms(image=image)["image"].unsqueeze(0)
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Create a dictionary of class names and their probabilities
        confidence_scores = {CLASS_NAMES[i]: prob.item() for i, prob in enumerate(probabilities)}

        return confidence_scores
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise gr.Error("Failed to process the image. Please try another one or ensure it is a valid format (JPEG, PNG).")


# --- Gradio Interface ---

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a Rice Leaf Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Rice Leaf Disease Diagnosis",
    description="An interactive web app to diagnose rice leaf diseases. Upload an image to see the model's prediction.",
    examples=[
        # Add paths to example images if available
        # ["path/to/example1.jpg"],
        # ["path/to/example2.jpg"]
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()
