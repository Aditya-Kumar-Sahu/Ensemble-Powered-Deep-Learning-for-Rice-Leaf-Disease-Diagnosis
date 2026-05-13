# Ensemble-Powered Deep Learning for Rice Leaf Disease Diagnosis

A production-ready, full-stack machine learning system for classifying rice leaf diseases. This project has been refactored from a research notebook into a modular, scalable, and deployable application.

## 🚀 Features

*   **Modular Architecture**: Clean separation of data, modeling, training, and evaluation logic.
*   **Multi-Model Support**: Supports **ResNet50**, **MobileNetV2**, and **EfficientNet-B0**.
*   **Ensemble Learning**: Combines predictions from multiple models for improved accuracy.
*   **Advanced Data Pipeline**: Uses `albumentations` for robust image augmentation and custom datasets compatible with PyTorch.
*   **Experiment Tracking**: Integrated with **MLflow** to track hyperparameters, metrics, and model artifacts.
*   **Hyperparameter Optimization**: Automated tuning using **Optuna**.
*   **Model Interpretability**: **Grad-CAM** integration to visualize model focus areas.
*   **Deployment**:
    *   **FastAPI**: High-performance REST API for model serving.
    *   **Gradio**: Interactive web UI for easy testing and demonstration.
*   **Domain Knowledge**: Integrated disease knowledge base providing symptoms, treatments, and prevention tips.
*   **Robustness**: Comprehensive unit and integration tests, plus CI/CD with GitHub Actions.

## 📂 Directory Structure

```
.
├── apps/                         # Deployment applications
│   ├── fastapi_app.py            # REST API
│   └── gradio_app.py             # Web UI
├── configs/                      # Configuration files
│   ├── base_config.yaml          # Default settings
│   └── model_configs/            # Model-specific overrides
├── data/                         # Dataset and knowledge base
│   └── disease_info.json         # Disease details
├── models/                       # Saved model checkpoints
├── logs/                         # Training logs
├── results/                      # Evaluation outputs (plots, reports)
├── scripts/                      # Executable scripts
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── interpret.py              # Grad-CAM visualization
│   └── optimize_hyperparameters.py # Optuna optimization
├── src/                          # Source code
│   ├── data/                     # Data loading & augmentation
│   ├── evaluation/               # Metrics & visualization
│   ├── models/                   # Model architectures
│   ├── training/                 # Trainer & optimizer logic
│   └── utils/                    # Utilities (config, logging, etc.)
└── tests/                        # Unit and integration tests
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### 1. Data Preparation
Ensure your dataset is located in the `data/` directory. The project expects a structure compatible with `torchvision.datasets.ImageFolder` (subdirectories for each class).

### 2. Training
Train a model using the `train.py` script. You can specify the model architecture (`resnet50`, `mobilenetv2`, `efficientnetb0`) and data directory.

```bash
python scripts/train.py --data-dir "data/Rice Leaf Disease Images" --model resnet50
```

Configuration is handled via YAML files in `configs/`. You can modify `configs/base_config.yaml` or model-specific files in `configs/model_configs/`.

### 3. Evaluation
Evaluate a trained model or an ensemble of models.

**Single Model:**
```bash
python scripts/evaluate.py --data-dir "data/Rice Leaf Disease Images" --model resnet50
```

**Ensemble:**
```bash
python scripts/evaluate.py --data-dir "data/Rice Leaf Disease Images" --ensemble
```

### 4. Hyperparameter Optimization
Use Optuna to find the best hyperparameters.

```bash
python scripts/optimize_hyperparameters.py --data-dir "data/Rice Leaf Disease Images" --model resnet50 --n-trials 20
```

### 5. Model Interpretability (Grad-CAM)
Visualize what the model is looking at.

```bash
python scripts/interpret.py --image_path "data/sample_image.jpg" --model resnet50
```

## 🚀 Deployment

### REST API (FastAPI)
Start the API server:
```bash
uvicorn apps.fastapi_app:app --reload
```
Access the API docs at `http://127.0.0.1:8000/docs`.

### Web Application (Gradio)
Launch the interactive UI:
```bash
python apps/gradio_app.py
```
Open your browser to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

## 🧪 Testing & CI/CD

Run the test suite:
```bash
pytest
```

This project uses **GitHub Actions** for CI/CD. On every pull request, the workflow:
*   Lints code with `flake8`.
*   Checks formatting with `black`.
*   Runs type checks with `mypy`.
*   Executes the full test suite with `pytest`.

## 📈 MLOps

Experiments are tracked using **MLflow**. To view the UI:
```bash
mlflow ui
```
This will allow you to compare training runs, view metrics charts, and access logged artifacts.
