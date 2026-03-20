"""
Utility for accessing the Disease Knowledge Base.
"""

import json
from pathlib import Path
from typing import Dict, Any


def get_disease_info(class_name: str, data_path: str = "data/disease_info.json") -> Dict[str, Any]:
    """
    Retrieves information for a specific disease class from the knowledge base.

    Args:
        class_name (str): The name of the disease class (e.g., 'Bacterialblight').
        data_path (str): Path to the JSON file containing disease info.

    Returns:
        Dict[str, Any]: A dictionary containing details about the disease (symptoms, treatment, etc.).
                        Returns a default "Unknown" dictionary if the class or file is not found.
    """
    path = Path(data_path)

    if not path.exists():
        return {
            "name": class_name,
            "description": "No information available.",
            "symptoms": "N/A",
            "cause": "N/A",
            "treatment": "N/A",
            "prevention": "N/A",
        }

    try:
        with open(path, "r") as f:
            data = json.load(f)

        info = data.get(class_name)
        if info:
            return info
        else:
            return {
                "name": class_name,
                "description": "Disease details not found in knowledge base.",
                "symptoms": "Unknown",
                "cause": "Unknown",
                "treatment": "Consult a local agronomist.",
                "prevention": "Unknown",
            }

    except Exception as e:
        print(f"Error loading disease info: {e}")
        return {
            "name": class_name,
            "description": "Error retrieving information.",
            "symptoms": "Error",
            "cause": "Error",
            "treatment": "Error",
            "prevention": "Error",
        }
