#!/usr/bin/env python3
"""
Gradio web application for rice leaf disease classification.
"""

import sys
from pathlib import Path
import yaml
import torch
import gradio as gr
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model
from src.data.augmentations import get_val_transforms
from src.utils import get_device


# Disease information database
DISEASE_INFO = {
    "Healthy": {
        "description": "The leaf appears healthy with no visible signs of disease.",
        "treatment": "No treatment needed. Continue regular crop maintenance.",
        "prevention": "Maintain good field hygiene and monitor regularly.",
    },
    "BacterialLeafBlight": {
        "description": "Bacterial disease causing water-soaked lesions on leaves.",
        "treatment": "Apply copper-based bactericides. Remove infected plants.",
        "prevention": "Use resistant varieties, proper water management, avoid excess nitrogen.",
    },
    "BrownSpot": {
        "description": "Fungal disease causing brown spots with yellow halos.",
        "treatment": "Apply fungicides like mancozeb or tricyclazole.",
        "prevention": "Use disease-free seeds, balanced fertilization, avoid water stress.",
    },
    "LeafSmut": {
        "description": "Fungal disease causing black powdery spores on leaves.",
        "treatment": "Apply systemic fungicides. Remove infected parts.",
        "prevention": "Use certified seeds, crop rotation, maintain field hygiene.",
    },
    # Add more diseases as needed
}


class RiceDiseaseClassifier:
    """Rice disease classification application."""

    def __init__(self, config_path: str = "configs/base_config.yaml"):
        """
        Create a RiceDiseaseClassifier instance and initialize runtime resources.

        Initializes the computation device, loads configuration from `config_path`, prepares the image validation transform, initializes an empty model cache, and retrieves class names.

        Parameters:
                config_path (str): Path to the YAML configuration file used to load settings (e.g., image size and other data/model options).
        """
        self.device = get_device()
        self.config = self.load_config(config_path)
        self.models = {}
        self.transform = get_val_transforms(self.config["data"]["image_size"])
        self.class_names = self.get_class_names()

    def load_config(self, config_path: str) -> dict:
        """
        Load a YAML configuration file from disk and parse it into a dictionary.

        Parameters:
            config_path (str): Path to the YAML configuration file.

        Returns:
            dict: Parsed configuration mapping from the YAML file.
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def get_class_names(self) -> list:
        """
        List of class names used by the classifier.

        These names correspond to model output indices and should match the dataset's label order.

        Returns:
            class_names (list[str]): Ordered list of class label strings.
        """
        # Placeholder - replace with actual class names from your dataset
        return [
            "BacterialLeafBlight",
            "BrownSpot",
            "LeafSmut",
            "Healthy",
            # Add other classes
        ]

    def load_model(self, model_name: str, checkpoint_path: str):
        """
        Load and cache a model architecture, restore its weights from a checkpoint, move it to the configured device, and set it to evaluation mode.

        Parameters:
            model_name (str): Identifier of the model architecture to instantiate (used by get_model).
            checkpoint_path (str): Filesystem path to a PyTorch state dict to load into the model.

        Returns:
            torch.nn.Module: The loaded model instance placed on the classifier's device and set to eval mode.
        """
        if model_name not in self.models:
            num_classes = len(self.class_names)
            model = get_model(model_name, num_classes)
            model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[model_name] = model
        return self.models[model_name]

    def predict(self, image: Image.Image, model_name: str = "resnet50") -> dict:
        """
        Predict the rice leaf disease from a PIL Image using a specified model.

        Parameters:
            image (PIL.Image.Image): Input image of a rice leaf.
            model_name (str): Model identifier to use for inference (e.g., "resnet50").

        Returns:
            dict: A dictionary with prediction results containing:
                - "prediction" (str): Predicted class name.
                - "confidence" (float): Probability of the predicted class (0.0 to 1.0).
                - "top5" (dict): Mapping of up to five class names to their probability scores.
        """
        # Load model if not already loaded
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

        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, min(5, len(self.class_names)))

        predicted_class = self.class_names[pred_idx]

        # Build results
        results = {
            "prediction": predicted_class,
            "confidence": confidence,
            "top5": {self.class_names[idx]: prob.item() for idx, prob in zip(top5_indices, top5_probs)},
        }

        return results

    def format_output(self, results: dict) -> tuple:
        """
        Builds a Markdown-formatted prediction summary and returns it alongside the top-5 predictions.

        Parameters:
            results (dict): Result dictionary produced by predict containing:
                - "prediction" (str): predicted class name.
                - "confidence" (float): confidence for the top prediction (0.0–1.0).
                - "top5" (dict): mapping of class names to their probabilities.

        Returns:
            tuple: (prediction_text, top5_dict)
                prediction_text (str): Markdown string with the predicted disease, confidence percentage,
                    and, if available in DISEASE_INFO, Description, Treatment, and Prevention sections.
                top5_dict (dict): the "top5" mapping from the input results (class name -> probability).
        """
        prediction = results["prediction"]
        confidence = results["confidence"]

        # Main prediction text
        prediction_text = f"## Predicted Disease: {prediction}\n"
        prediction_text += f"**Confidence:** {confidence:.2%}\n\n"

        # Disease information
        if prediction in DISEASE_INFO:
            info = DISEASE_INFO[prediction]
            prediction_text += f"### Description\n{info['description']}\n\n"
            prediction_text += f"### Treatment\n{info['treatment']}\n\n"
            prediction_text += f"### Prevention\n{info['prevention']}\n"

        # Top 5 predictions
        top5_dict = results["top5"]

        return prediction_text, top5_dict


def create_app():
    """
    Create and configure the Gradio web application for rice leaf disease classification.

    Sets up the UI (image input, model choice, analyze button, prediction display, and top-5 label),
    binds a prediction wrapper that uses RiceDiseaseClassifier, and includes informational text.

    Returns:
        app (gr.Blocks): A configured Gradio Blocks application ready to be launched.
    """
    classifier = RiceDiseaseClassifier()

    def predict_wrapper(image, model_choice):
        """
        Handle an uploaded image and model selection, perform classification, and return formatted UI-ready results.

        Parameters:
            image (PIL.Image.Image or None): Uploaded image to classify; if None, no prediction is performed.
            model_choice (str): Name of the model to use (e.g., "resnet50", "mobilenetv2", "efficientnetb0").

        Returns:
            tuple: A pair (prediction_text, top5_dict). `prediction_text` is a user-facing message or Markdown containing the main prediction and details; `top5_dict` maps top-5 class names to their probabilities. If no image is provided or an error occurs, `prediction_text` contains a guidance or error message and `top5_dict` is an empty dict.
        """
        if image is None:
            return "Please upload an image.", {}

        try:
            results = classifier.predict(image, model_choice)
            return classifier.format_output(results)
        except Exception as e:
            return f"Error: {str(e)}", {}

    # Create Gradio interface
    with gr.Blocks(title="Rice Leaf Disease Classifier") as app:
        gr.Markdown(
            """
            # 🌾 Rice Leaf Disease Classification System
            
            Upload an image of a rice leaf to diagnose potential diseases using deep learning.
            
            **Features:**
            - Multiple model architectures (ResNet50, MobileNetV2, EfficientNet-B0)
            - Ensemble prediction for improved accuracy
            - Disease information and treatment recommendations
            """
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload Rice Leaf Image",
                )

                model_choice = gr.Radio(
                    choices=["resnet50", "mobilenetv2", "efficientnetb0"],
                    value="resnet50",
                    label="Select Model",
                )

                predict_btn = gr.Button("🔍 Analyze", variant="primary")

                gr.Examples(
                    examples=[],  # Add example images here
                    inputs=image_input,
                    label="Example Images",
                )

            with gr.Column():
                prediction_output = gr.Markdown(label="Prediction Results")

                confidence_output = gr.Label(
                    label="Top 5 Predictions",
                    num_top_classes=5,
                )

        predict_btn.click(
            fn=predict_wrapper,
            inputs=[image_input, model_choice],
            outputs=[prediction_output, confidence_output],
        )

        gr.Markdown(
            """
            ---
            ### About
            
            This application uses ensemble deep learning to classify rice leaf diseases.
            The models are trained on a comprehensive dataset of 15 disease categories.
            
            **Disclaimer:** This tool is for educational and research purposes. 
            Always consult agricultural experts for critical decisions.
            
            **Models:** ResNet50, MobileNetV2, EfficientNet-B0
            """
        )

    return app


def main():
    """
    Start and launch the Gradio web application.

    Creates the app via create_app() and starts the Gradio server listening on 0.0.0.0:7860 without public sharing.
    """
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
