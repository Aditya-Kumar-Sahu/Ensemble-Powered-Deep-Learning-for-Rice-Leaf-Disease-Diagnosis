"""
FastAPI application for serving the Rice Leaf Disease Classification model.
"""

import io
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model
from src.utils import load_config
from src.data.augmentations import get_val_transforms

# --- Application Setup ---
app = FastAPI(
    title="Rice Leaf Disease Diagnosis API",
    description="An API to predict rice leaf diseases from images.",
    version="1.0.0",
)

# --- Model and Config Loading ---
# This section runs once at startup

# Load a default model configuration (e.g., resnet50)
MODEL_NAME = "resnet50"
config = load_config(model_config_path=f"configs/model_configs/{MODEL_NAME}.yaml")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
# Note: Ensure you have a trained model checkpoint at the specified path
model_path = Path(config["output"]["models_dir"]) / f"{MODEL_NAME}.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Please train the model first.")

# Assuming 15 classes for the Rice Leaf Disease dataset
NUM_CLASSES = 15
model = get_model(MODEL_NAME, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Get the validation transforms
val_transforms = get_val_transforms(config)

# --- API Endpoints ---


@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the Rice Leaf Disease Diagnosis API!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the disease from an uploaded rice leaf image.

    Args:
        file (UploadFile): The image file to be classified.

    Returns:
        JSONResponse: A JSON response containing the predicted class and confidence score.
    """
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Preprocess the image
    input_tensor = val_transforms(image=image_np)["image"].unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class_idx = torch.max(probabilities, 0)

    # Assuming class names can be retrieved or are known
    # For simplicity, we'll just return the class index.
    # A more robust solution would map this index to a class name.
    predicted_class = predicted_class_idx.item()
    confidence_score = confidence.item()

    return JSONResponse(content={"predicted_class_index": predicted_class, "confidence": f"{confidence_score:.4f}"})


# To run this app:
# uvicorn apps.fastapi_app:app --reload
