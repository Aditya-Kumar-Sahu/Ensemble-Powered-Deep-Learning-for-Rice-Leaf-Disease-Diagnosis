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
        """Initialize the classifier."""
        self.device = get_device()
        self.config = self.load_config(config_path)
        self.models = {}
        self.transform = get_val_transforms(self.config["data"]["image_size"])
        self.class_names = self.get_class_names()
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def get_class_names(self) -> list:
        """Get class names. This should be loaded from dataset or config."""
        # Placeholder - replace with actual class names from your dataset
        return [
            "BacterialLeafBlight",
            "BrownSpot", 
            "LeafSmut",
            "Healthy",
            # Add other classes
        ]
    
    def load_model(self, model_name: str, checkpoint_path: str):
        """Load a trained model."""
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
        Predict disease from image.
        
        Args:
            image: PIL Image
            model_name: Name of model to use
            
        Returns:
            Dictionary with prediction results
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
            "top5": {
                self.class_names[idx]: prob.item()
                for idx, prob in zip(top5_indices, top5_probs)
            },
        }
        
        return results
    
    def format_output(self, results: dict) -> tuple:
        """Format output for Gradio."""
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
    """Create Gradio application."""
    classifier = RiceDiseaseClassifier()
    
    def predict_wrapper(image, model_choice):
        """Wrapper for prediction."""
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
    """Main function."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
