# Developer Guide & Architecture

This document provides an overview of the project's architecture and development workflows suitable for contributors.

## Architecture Overview

The project follows a modular design pattern to ensure scalability and maintainability.

*   **`src/`**: The core logic library.
    *   **`data/`**: Handles data ingestion (`loaders.py`) and augmentation (`augmentations.py`). It uses `albumentations` for flexible image transformations.
    *   **`models/`**: Contains PyTorch model definitions. The `base_model.py` acts as a factory, while `ensemble.py` implements voting strategies.
    *   **`training/`**: The `Trainer` class manages the training loop, validation, and MLflow logging. It decouples the training logic from the script.
    *   **`evaluation/`**: Utilities for calculating metrics, generating reports, and visualizations (confusion matrices, Grad-CAM).
    *   **`utils/`**: General-purpose utilities including configuration management, logging, seeding, and MLflow helpers.

*   **`scripts/`**: Command-line entry points. These are thin wrappers around the `src/` modules, handling argument parsing and high-level orchestration.

*   **`apps/`**: Deployment applications.
    *   `fastapi_app.py`: Asynchronous REST API using FastAPI.
    *   `gradio_app.py`: Interactive web UI using Gradio.

*   **`configs/`**: Hierarchical configuration. `base_config.yaml` contains defaults, which are overridden by model-specific YAML files.

## Development Workflows

### Configuration Management
Do not hardcode hyperparameters. Add them to `configs/base_config.yaml` or a model-specific config. The `load_config` utility in `src/utils/config.py` handles recursive merging.

### Experiment Tracking
Use **MLflow** for all experiments. The `Trainer` class automatically logs:
*   Hyperparameters (from config).
*   Metrics (loss, accuracy, learning rate).
*   Artifacts (best model checkpoint, history file).

To view results locally:
```bash
mlflow ui
```

### Testing
Write unit tests for all new functions and classes.
*   **Unit Tests**: Place in `tests/`.
*   **Integration Tests**: `tests/test_integration.py` validates the end-to-end training pipeline.

Run tests before committing:
```bash
pytest
```

### Code Style
Follow PEP 8. The CI pipeline enforces style using `flake8` and `black`.
```bash
black src/ tests/ scripts/ apps/
flake8 src/
```

### Adding New Models
1.  Define the model architecture in a new file in `src/models/` (or add to existing).
2.  Register it in `src/models/base_model.py`.
3.  Create a corresponding config file in `configs/model_configs/`.
