"""Unit tests for evaluation modules."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from sklearn.metrics import accuracy_score

from src.evaluation.metrics import (
    calculate_metrics,
    predict_single,
    evaluate_model,
)
from src.evaluation.reports import (
    generate_classification_report,
    print_model_summary,
    print_metrics_table,
)
from src.evaluation.visualizations import (
    plot_confusion_matrix,
    plot_training_history,
    compare_models,
)


@pytest.fixture
def sample_predictions():
    """Fixture for sample predictions."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    return y_true, y_pred


@pytest.fixture
def sample_probabilities():
    """Fixture for sample probability predictions."""
    # 9 samples, 3 classes
    y_probs = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.05, 0.05, 0.9],
            [0.85, 0.1, 0.05],
            [0.2, 0.3, 0.5],
            [0.1, 0.1, 0.8],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.3, 0.5, 0.2],
        ]
    )
    return y_probs


@pytest.fixture
def class_names():
    """Fixture for class names."""
    return ["Class A", "Class B", "Class C"]


@pytest.fixture
def mock_model():
    """Fixture for mock model."""
    model = Mock()
    model.eval = Mock()
    model.to = Mock(return_value=model)
    return model


@pytest.fixture
def mock_dataloader():
    """Fixture for mock dataloader."""
    # Create mock data
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])

    dataloader = [(images[:1], labels[:1]), (images[1:], labels[1:])]
    return dataloader


class TestCalculateMetrics:
    """Test suite for calculate_metrics function."""

    def test_calculate_metrics_basic(self, sample_predictions):
        """Test basic metrics calculation."""
        y_true, y_pred = sample_predictions

        metrics = calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert "precision_weighted" in metrics
        assert "recall_weighted" in metrics
        assert "f1_weighted" in metrics

        # Check values are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1

    def test_calculate_metrics_with_probabilities(self, sample_predictions, sample_probabilities):
        """Test metrics calculation with probabilities."""
        y_true, y_pred = sample_predictions

        metrics = calculate_metrics(y_true, y_pred, y_probs=sample_probabilities, num_classes=3)

        assert "roc_auc_ovr" in metrics
        assert 0 <= metrics["roc_auc_ovr"] <= 1

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["f1_macro"] == 1.0

    def test_calculate_metrics_all_wrong(self):
        """Test metrics with all wrong predictions."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_calculate_metrics_binary_classification(self):
        """Test metrics with binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        metrics = calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert isinstance(metrics["accuracy"], float)

    def test_calculate_metrics_multiclass(self):
        """Test metrics with multiclass classification."""
        y_true = np.array([0, 1, 2, 3, 4] * 3)
        y_pred = np.array([0, 1, 2, 3, 3] * 3)

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.8

    def test_calculate_metrics_with_zero_division(self):
        """Test metrics calculation handles zero division."""
        # All predictions same class
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0, 0, 0])

        # Should not raise error due to zero_division=0
        metrics = calculate_metrics(y_true, y_pred)

        assert "precision_macro" in metrics


class TestPredictSingle:
    """Test suite for predict_single function."""

    def test_predict_single_basic(self, mock_model, mock_dataloader):
        """Test basic prediction functionality."""
        device = torch.device("cpu")

        # Mock model output
        def mock_forward(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 3)  # 3 classes

        mock_model.side_effect = mock_forward

        with patch("torch.no_grad"):
            y_true, y_pred, y_probs = predict_single(mock_model, mock_dataloader, device)

        assert len(y_true) == 2
        assert len(y_pred) == 2
        assert len(y_probs) == 2

    def test_predict_single_output_shapes(self):
        """Test output shapes of predict_single."""
        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 224 * 224, 5))

        # Create simple dataloader
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        dataloader = [(images, labels)]

        device = torch.device("cpu")

        y_true, y_pred, y_probs = predict_single(model, dataloader, device)

        assert y_true.shape == (4,)
        assert y_pred.shape == (4,)
        assert y_probs.shape == (4, 5)


class TestEvaluateModel:
    """Test suite for evaluate_model function."""

    def test_evaluate_model_output(self, sample_predictions, class_names, capsys):
        """Test evaluate_model prints and returns metrics."""
        y_true, y_pred = sample_predictions

        metrics = evaluate_model(y_true, y_pred, class_names)

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

        # Check that something was printed
        captured = capsys.readouterr()
        assert "EVALUATION REPORT" in captured.out
        assert "Accuracy" in captured.out


class TestGenerateClassificationReport:
    """Test suite for generate_classification_report function."""

    def test_generate_classification_report(self, sample_predictions, class_names):
        """Test classification report generation."""
        y_true, y_pred = sample_predictions

        report = generate_classification_report(y_true, y_pred, class_names)

        assert isinstance(report, str)
        assert "precision" in report
        assert "recall" in report
        assert "f1-score" in report

        # Check class names are in report
        for name in class_names:
            assert name in report

    def test_generate_classification_report_binary(self):
        """Test classification report for binary classification."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        class_names = ["Negative", "Positive"]

        report = generate_classification_report(y_true, y_pred, class_names)

        assert "Negative" in report
        assert "Positive" in report


class TestPrintModelSummary:
    """Test suite for print_model_summary function."""

    def test_print_model_summary_no_files(self, capsys):
        """Test print_model_summary with no history files."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            print_model_summary(["model1", "model2"], log_folder=tmpdir)

            captured = capsys.readouterr()
            assert "MODEL SUMMARY" in captured.out
            assert "N/A" in captured.out

    def test_print_model_summary_with_history(self, capsys):
        """Test print_model_summary with existing history."""
        import tempfile
        from src.utils.checkpoint import save_history

        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "train_loss": [0.5, 0.4],
                "val_loss": [0.6, 0.5],
                "val_acc": [80.0, 85.0],
                "params": 1000000,
                "training_time": 120.5,
                "best_val_acc": 85.0,
            }

            save_history(history, "test_model", folder=tmpdir)
            print_model_summary(["test_model"], log_folder=tmpdir)

            captured = capsys.readouterr()
            assert "test_model" in captured.out
            assert "1,000,000" in captured.out or "1000000" in captured.out


class TestPrintMetricsTable:
    """Test suite for print_metrics_table function."""

    def test_print_metrics_table(self, capsys):
        """Test metrics table printing."""
        metrics_dict = {
            "model1": {"accuracy": 0.85, "f1_macro": 0.83},
            "model2": {"accuracy": 0.87, "f1_macro": 0.85},
        }

        print_metrics_table(metrics_dict)

        captured = capsys.readouterr()
        assert "METRICS COMPARISON" in captured.out
        assert "model1" in captured.out
        assert "model2" in captured.out
        assert "accuracy" in captured.out

    def test_print_metrics_table_empty(self, capsys):
        """Test metrics table with empty dict."""
        print_metrics_table({})

        captured = capsys.readouterr()
        assert "No metrics to display" in captured.out


class TestVisualizationFunctions:
    """Test suite for visualization functions."""

    def test_plot_confusion_matrix_basic(self, sample_predictions, class_names):
        """Test confusion matrix plotting."""
        y_true, y_pred = sample_predictions

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.pyplot.figure"):
                with patch("seaborn.heatmap"):
                    # Should not raise error
                    plot_confusion_matrix(y_true, y_pred, class_names)

    def test_plot_confusion_matrix_with_save(self, sample_predictions, class_names):
        """Test confusion matrix saving."""
        import tempfile

        y_true, y_pred = sample_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = f"{tmpdir}/confusion_matrix.png"

            with patch("matplotlib.pyplot.show"):
                plot_confusion_matrix(y_true, y_pred, class_names, save_path=save_path)

    def test_plot_training_history(self):
        """Test training history plotting."""
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
            "train_acc": [80, 85, 90],
            "val_acc": [75, 80, 85],
        }

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                fig = Mock()
                axes = [Mock(), Mock()]
                mock_subplots.return_value = (fig, axes)

                plot_training_history(history, metrics=["loss", "acc"])

    def test_plot_training_history_single_metric(self):
        """Test training history plotting with single metric."""
        history = {
            "train_loss": [0.5, 0.4, 0.3],
            "val_loss": [0.6, 0.5, 0.4],
        }

        with patch("matplotlib.pyplot.show"):
            with patch("matplotlib.pyplot.subplots") as mock_subplots:
                fig = Mock()
                ax = Mock()
                mock_subplots.return_value = (fig, ax)

                plot_training_history(history, metrics=["loss"])

    def test_compare_models(self):
        """Test model comparison plotting."""
        import tempfile
        from src.utils.checkpoint import save_history

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock histories
            for model_name in ["model1", "model2"]:
                history = {
                    "train_loss": [0.5, 0.4, 0.3],
                    "val_loss": [0.6, 0.5, 0.4],
                }
                save_history(history, model_name, folder=tmpdir)

            with patch("matplotlib.pyplot.show"):
                with patch("matplotlib.pyplot.figure"):
                    compare_models(["model1", "model2"], metric="loss", log_folder=tmpdir)

    def test_compare_models_missing_history(self, capsys):
        """Test model comparison with missing history."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("matplotlib.pyplot.show"):
                with patch("matplotlib.pyplot.figure"):
                    compare_models(["nonexistent_model"], metric="loss", log_folder=tmpdir)

            captured = capsys.readouterr()
            assert "Warning" in captured.out or "not found" in captured.out
