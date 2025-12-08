#!/usr/bin/env python3
"""
FastAPI REST API for rice leaf disease classification.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import yaml
import torch
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model
from src.data.augmentations import get_val_transforms
from src.utils import get_device


# Response models
class PredictionResult(BaseModel):
    """Prediction result model."""

    predicted_class: str
    confidence: float
    top_predictions: Dict[str, float]
    model_used: str


class HealthCheck(BaseModel):
    """Health check response model."""

    status: str
    models_loaded: List[str]
    device: str


class DiseaseInfo(BaseModel):
    """Disease information model."""

    name: str
    description: str
    treatment: str
    prevention: str


# Disease information database
DISEASE_DATABASE = {
    "BacterialLeafBlight": {
        "description": "Bacterial disease causing water-soaked lesions on leaves that turn yellow to white.",
        "treatment": "Apply copper-based bactericides. Remove and destroy infected plants. Maintain proper field drainage.",
        "prevention": "Use resistant varieties, proper water management, avoid excess nitrogen fertilization.",
    },
    "BrownSpot": {
        "description": "Fungal disease causing small, circular to oval brown spots with yellow halos on leaves.",
        "treatment": "Apply fungicides like mancozeb, tricyclazole, or carbendazim. Ensure balanced fertilization.",
        "prevention": "Use disease-free seeds, maintain balanced nutrition, avoid water stress.",
    },
    "LeafSmut": {
        "description": "Fungal disease producing black powdery spore masses on leaves and stems.",
        "treatment": "Apply systemic fungicides. Remove and destroy infected plant parts.",
        "prevention": "Use certified disease-free seeds, practice crop rotation, maintain field hygiene.",
    },
    "Healthy": {
        "description": "Leaf appears healthy with no visible signs of disease.",
        "treatment": "No treatment needed. Continue regular crop maintenance practices.",
        "prevention": "Maintain good agricultural practices, monitor regularly for early disease detection.",
    },
}


# Initialize FastAPI app
app = FastAPI(
    title="Rice Leaf Disease Classification API",
    description="REST API for classifying rice leaf diseases using deep learning",
    version="1.0.0",
)


# Global variables for model management
classifier = None


class RiceDiseaseAPI:
    """Rice disease classification API."""

    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Create a RiceDiseaseAPI instance configured from the given YAML file.

        Parameters:
            config_path (str): Path to a YAML configuration file. If the file is missing or unreadable, a minimal default configuration is used.

        Description:
            Initializes runtime state including the compute device, loaded configuration, model registry, validation image transform, and class name list.
        """
        self.device = get_device()
        self.config = self.load_config(config_path)
        self.models = {}
        self.transform = get_val_transforms(self.config["data"]["image_size"])
        self.class_names = self.get_class_names()

    def load_config(self, config_path: str) -> dict:
        """
        Load configuration from a YAML file, falling back to a minimal default if the file is missing.

        Parameters:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration dictionary parsed from the YAML file, or a default config with image size 224 if the file is not found.
        """
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                "data": {"image_size": 224},
            }

    def get_class_names(self) -> list:
        """
        Return the list of disease class names used by the classifier.

        This is a static, ordered list mapping model output indices to human-readable class labels.

        Returns:
            List[str]: Class names in the order corresponding to model output indices.
        """
        # Placeholder - should be loaded from config or dataset
        return [
            "BacterialLeafBlight",
            "BrownSpot",
            "LeafSmut",
            "Healthy",
        ]

    def load_model(self, model_name: str, checkpoint_path: str):
        """
        Load and cache a model for the given name from a checkpoint file.

        If the model is already cached in self.models, the cached instance is returned; otherwise the model is constructed for the configured number of classes, its state dict is loaded from checkpoint_path, moved to the configured device, set to evaluation mode, cached, and returned.

        Parameters:
            model_name (str): Identifier of the model architecture/key used to construct and cache the model.
            checkpoint_path (str): Filesystem path to the checkpoint containing the model's state dict.

        Returns:
            torch.nn.Module: The loaded PyTorch model instance moved to the configured device and set to evaluation mode.
        """
        if model_name not in self.models:
            num_classes = len(self.class_names)
            model = get_model(model_name, num_classes)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[model_name] = model
        return self.models[model_name]

    def predict(
        self,
        image: Image.Image,
        model_name: str = "resnet50",
        top_k: int = 5,
    ) -> dict:
        """
        Predict the disease class and associated probabilities for a given image using a specified model.

        Parameters:
            image (PIL.Image.Image): Input image to classify; expected in RGB.
            model_name (str): Name of the model to use (selects which pretrained model is loaded).
            top_k (int): Number of top class predictions to include in the `top_predictions` map; capped at the number of available classes.

        Returns:
            dict: Mapping with keys:
                - "predicted_class" (str): Class with highest probability.
                - "confidence" (float): Probability of the predicted class (0.0–1.0).
                - "top_predictions" (Dict[str, float]): Top `top_k` class names mapped to their probabilities.
                - "model_used" (str): The model name that produced the prediction.
        """
        # Load model
        checkpoint_path = f"models/{model_name}.pth"
        model = self.load_model(model_name, checkpoint_path)

        # Transform image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()

        # Get top K predictions
        top_k = min(top_k, len(self.class_names))
        topk_probs, topk_indices = torch.topk(probs, top_k)

        predicted_class = self.class_names[pred_idx]

        # Build results
        results = {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "top_predictions": {
                self.class_names[idx]: float(prob)
                for idx, prob in zip(topk_indices, topk_probs)
            },
            "model_used": model_name,
        }

        return results


@app.on_event("startup")
async def startup_event():
    """
    Create and assign the application classifier used for predictions.

    Initializes a RiceDiseaseAPI instance and stores it in the module-level variable `classifier` for use by request handlers.
    """
    global classifier
    classifier = RiceDiseaseAPI()


@app.get("/", response_model=dict)
async def root():
    """
    Return basic API metadata and available endpoints.

    Returns:
        info (dict): Metadata containing:
            - message (str): Brief API title.
            - version (str): API version.
            - endpoints (dict): Mapping of endpoint names to their paths, including "health", "predict", "disease_info", and "models".
    """
    return {
        "message": "Rice Leaf Disease Classification API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "disease_info": "/disease-info/{disease_name}",
            "models": "/models",
        },
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Return current service health status and runtime metadata.

    Returns:
        dict: A mapping with keys:
            - "status": `"healthy"` when the service is operational.
            - "models_loaded": list of loaded model names.
            - "device": string representation of the device used for inference.
    """
    return {
        "status": "healthy",
        "models_loaded": list(classifier.models.keys()),
        "device": str(classifier.device),
    }


@app.get("/models", response_model=List[str])
async def list_models():
    """
    List available model names.

    Returns:
        List[str]: Available model names.
    """
    return ["resnet50", "mobilenetv2", "efficientnetb0"]


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    model: str = Query("resnet50", description="Model to use for prediction"),
    top_k: int = Query(5, ge=1, le=10, description="Number of top predictions"),
):
    """
    Handle an uploaded image and return rice leaf disease predictions from the selected model.

    Parameters:
        file: Uploaded image file; must be a valid image readable by Pillow (will be converted to RGB).
        model: Model name to use for prediction; must be one of "resnet50", "mobilenetv2", or "efficientnetb0".
        top_k: Number of top predictions to include (1–10).

    Returns:
        dict: Prediction result with keys `predicted_class` (str), `confidence` (float), `top_predictions` (Dict[str, float]), and `model_used` (str).

    Raises:
        HTTPException: 400 if the model name is invalid or the uploaded file is not a valid image.
        HTTPException: 404 if the model checkpoint is not found.
        HTTPException: 500 for other prediction errors.
    """
    # Validate model name
    if model not in ["resnet50", "mobilenetv2", "efficientnetb0"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Choose from resnet50, mobilenetv2, efficientnetb0",
        )

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}",
        )

    # Make prediction
    try:
        results = classifier.predict(image, model, top_k)
        return results
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint not found for {model}. Please train the model first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


@app.get("/disease-info/{disease_name}", response_model=DiseaseInfo)
async def get_disease_info(disease_name: str):
    """
    Return detailed information for a disease from the in-memory disease database.

    Parameters:
        disease_name (str): Disease key to look up in DISEASE_DATABASE (must match exactly, case-sensitive).

    Returns:
        dict: A mapping with keys "name", "description", "treatment", and "prevention".

    Raises:
        HTTPException: 404 if the disease_name is not found in DISEASE_DATABASE.
    """
    if disease_name not in DISEASE_DATABASE:
        raise HTTPException(
            status_code=404,
            detail=f"Disease information not found for: {disease_name}",
        )

    info = DISEASE_DATABASE[disease_name]
    return {
        "name": disease_name,
        "description": info["description"],
        "treatment": info["treatment"],
        "prevention": info["prevention"],
    }


@app.get("/classes", response_model=List[str])
async def list_classes():
    """
    Retrieve the list of disease class names.

    Returns:
        List[str]: Class name strings used by the classifier.
    """
    return classifier.class_names


def main():
    """
    Start the Uvicorn server for the FastAPI application.

    Runs the app on host 0.0.0.0 and port 8000 with log level set to "info".
    """
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
