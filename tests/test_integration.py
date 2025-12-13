"""Integration tests for the training pipeline."""

import pytest
import subprocess
import os
import shutil
import tempfile
from pathlib import Path
import mlflow
from PIL import Image  # Added import for Image


# Fixture to create a dummy dataset (similar to the one in test_data.py)
@pytest.fixture
def dummy_integration_dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_root = Path(tmpdir) / "dummy_dataset"
        data_root.mkdir()

        class_names = ["disease_a", "disease_b"]
        for class_name in class_names:
            class_dir = data_root / class_name
            class_dir.mkdir()
            for i in range(2):  # 2 images per class
                img = Image.new("RGB", (224, 224), color=(i * 100, i * 50, i * 20))
                img.save(class_dir / f"img_{i}.jpg")
        yield str(data_root)


# Fixture to create a dummy config directory and files for the integration test
@pytest.fixture
def dummy_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / "configs"
        config_dir.mkdir()
        model_configs_dir = config_dir / "model_configs"
        model_configs_dir.mkdir()

        # Create dummy base_config.yaml
        base_config_content = """
data:
  image_size: 224
  batch_size: 2
  num_workers: 0 # Use 0 workers for tests to avoid multiprocessing issues
  val_split: 0.2
training:
  num_epochs: 1
  learning_rate: 0.0001
  weight_decay: 0.0
  optimizer: \"adam\"
  scheduler: \"cosine\"
model:
  dropout: 0.0
  pretrained: false
ensemble:
  voting: \"soft\"
  weights: null
output:
  models_dir: \"models_test\"
  logs_dir: \"logs_test\"
  results_dir: \"results_test\"
seed: 42
device: \"cpu\" # Force CPU for tests
mlflow:
  experiment_name: \"test_experiment\"
  run_name: \"test_run\"
"""
        (config_dir / "base_config.yaml").write_text(base_config_content)

        # Create dummy resnet50.yaml
        resnet50_config_content = """
model:
  name: \"resnet50\"
  architecture: \"resnet\"
  pretrained: false
  dropout: 0.0
"""
        (model_configs_dir / "resnet50.yaml").write_text(resnet50_config_content)

        # Ensure that src/utils/config.py uses these test configs
        # This setup needs to ensure load_config picks up the right paths.
        # For simplicity, during testing, we'll explicitly pass the base config path
        # from this fixture to the train script, and the train script's logic will
        # construct the model_config_path correctly relative to the base 'configs' dir.

        yield str(config_dir)


def test_training_pipeline_integration(dummy_integration_dataset_dir, dummy_config_dir):
    """
    Test the full training pipeline integration, including MLflow logging.
    Ensures the script runs without errors and MLflow logs are created.
    """
    # Set MLflow tracking URI to a temporary directory
    with tempfile.TemporaryDirectory() as mlflow_tmp_dir:
        mlflow.set_tracking_uri(Path(mlflow_tmp_dir).as_uri())

        # Define paths for outputs relative to the current working directory
        test_models_dir = Path("models_test")
        test_logs_dir = Path("logs_test")

        # Clean up any previous test runs
        if test_models_dir.exists():
            shutil.rmtree(test_models_dir)
        if test_logs_dir.exists():
            shutil.rmtree(test_logs_dir)

        command = [
            "python",
            "scripts/train.py",
            "--data-dir",
            dummy_integration_dataset_dir,
            "--model",
            "resnet50",
            # We don't specify --config directly, as scripts/train.py uses a default
            # based on project structure. The dummy_config_dir provides the files.
        ]

        # Ensure the subprocess uses the correct Python interpreter if in a venv
        # For testing, we assume 'python' is already the correct interpreter in path

        # Run the training script
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # Changed check=True to check=False

        # Print stdout and stderr for debugging in case of failure
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

        assert result.returncode == 0, f"Training script failed with error: {result.stderr}"

        # Check if MLflow run was created and logs exist
        runs = mlflow.search_runs(experiment_names=["test_experiment"])
        assert len(runs) > 0, "MLflow run was not created."

        run_id = runs.iloc[0].run_id
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]

        # Check if model artifact was logged
        assert (
            "model/pytorch_model.bin" in artifact_paths or "model" in artifact_paths
        ), "Trained model artifact not logged to MLflow."

        # Check if history artifact was logged
        assert "resnet50_history.npy" in artifact_paths, "History artifact not logged to MLflow."

        # Check if local model file was saved
        assert (test_models_dir / "resnet50.pth").exists(), "Local model checkpoint not saved."

        # Check if log file was created
        assert (test_logs_dir / "training.log").exists(), "Local log file not created."

        # Clean up local test outputs
        if test_models_dir.exists():
            shutil.rmtree(test_models_dir)
        if test_logs_dir.exists():
            shutil.rmtree(test_logs_dir)
