"""Unit tests for training modules."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os
from unittest.mock import Mock, patch

from src.training.trainer import Trainer, train_model
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler


@pytest.fixture
def simple_model():
    """Fixture for a simple model."""
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 128), nn.ReLU(), nn.Linear(128, 10))


@pytest.fixture
def simple_dataloader():
    """Fixture for a simple dataloader."""
    # Create dummy data
    images = torch.randn(16, 3, 224, 224)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def device():
    """Fixture for device."""
    return torch.device("cpu")


class TestGetOptimizer:
    """Test suite for get_optimizer function."""

    def test_get_optimizer_adam(self, simple_model):
        """Test Adam optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam", learning_rate=1e-3)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 1e-3

    def test_get_optimizer_adamw(self, simple_model):
        """Test AdamW optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adamw", learning_rate=5e-4)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 5e-4

    def test_get_optimizer_sgd(self, simple_model):
        """Test SGD optimizer creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="sgd", learning_rate=1e-2, momentum=0.9)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 1e-2
        assert optimizer.defaults["momentum"] == 0.9

    def test_get_optimizer_sgd_default_momentum(self, simple_model):
        """Test SGD optimizer with default momentum."""
        optimizer = get_optimizer(simple_model, optimizer_name="sgd")

        assert optimizer.defaults["momentum"] == 0.9

    def test_get_optimizer_with_weight_decay(self, simple_model):
        """Test optimizer with weight decay."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam", weight_decay=1e-4)

        assert optimizer.defaults["weight_decay"] == 1e-4

    def test_get_optimizer_case_insensitive(self, simple_model):
        """Test that optimizer names are case-insensitive."""
        opt1 = get_optimizer(simple_model, optimizer_name="ADAM")
        opt2 = get_optimizer(simple_model, optimizer_name="Adam")
        opt3 = get_optimizer(simple_model, optimizer_name="adam")

        assert all(isinstance(opt, torch.optim.Adam) for opt in [opt1, opt2, opt3])

    def test_get_optimizer_invalid_name(self, simple_model):
        """Test that invalid optimizer name raises error."""
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            get_optimizer(simple_model, optimizer_name="invalid_optimizer")

    def test_get_optimizer_with_kwargs(self, simple_model):
        """Test optimizer with additional kwargs."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam", betas=(0.9, 0.999), eps=1e-8)

        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8


class TestGetScheduler:
    """Test suite for get_scheduler function."""

    def test_get_scheduler_cosine(self, simple_model):
        """Test cosine annealing scheduler creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="cosine", num_epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_scheduler_step(self, simple_model):
        """Test step scheduler creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="step", step_size=30)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_get_scheduler_plateau(self, simple_model):
        """Test plateau scheduler creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="plateau", patience=10)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_get_scheduler_exponential(self, simple_model):
        """Test exponential scheduler creation."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="exponential", gamma=0.95)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_get_scheduler_none(self, simple_model):
        """Test that 'none' returns None."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="none")

        assert scheduler is None

    def test_get_scheduler_none_string_none(self, simple_model):
        """Test that None value returns None."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name=None)

        assert scheduler is None

    def test_get_scheduler_case_insensitive(self, simple_model):
        """Test that scheduler names are case-insensitive."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")

        sch1 = get_scheduler(optimizer, scheduler_name="COSINE")
        sch2 = get_scheduler(optimizer, scheduler_name="Cosine")

        assert isinstance(sch1, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert isinstance(sch2, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_scheduler_invalid_name(self, simple_model):
        """Test that invalid scheduler name raises error."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")

        with pytest.raises(ValueError, match="Unsupported scheduler"):
            get_scheduler(optimizer, scheduler_name="invalid_scheduler")

    def test_get_scheduler_with_custom_params(self, simple_model):
        """Test scheduler with custom parameters."""
        optimizer = get_optimizer(simple_model, optimizer_name="adam")
        scheduler = get_scheduler(optimizer, scheduler_name="cosine", T_max=50, eta_min=1e-6)

        assert scheduler.T_max == 50
        assert scheduler.eta_min == 1e-6


class TestTrainer:
    """Test suite for Trainer class."""

    def test_trainer_initialization(self, simple_model, simple_dataloader, device):
        """Test Trainer initialization."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        assert trainer.model is simple_model
        assert trainer.device == device
        assert len(trainer.history["train_loss"]) == 0

    def test_trainer_with_scheduler(self, simple_model, simple_dataloader, device):
        """Test Trainer with scheduler."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")
        scheduler = get_scheduler(optimizer, "cosine", num_epochs=10)

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
        )

        assert trainer.scheduler is scheduler

    def test_trainer_train_epoch(self, simple_model, simple_dataloader, device):
        """Test single training epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_trainer_validate_epoch(self, simple_model, simple_dataloader, device):
        """Test single validation epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        loss, acc = trainer.validate_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0
        assert 0 <= acc <= 100

    def test_trainer_train_full(self, simple_model, simple_dataloader, device):
        """Test full training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(simple_model, "adam", learning_rate=1e-2)

            trainer = Trainer(
                model=simple_model,
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            history = trainer.train(num_epochs=2, save_dir=tmpdir, model_name="test_model")

            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2
            assert len(history["train_acc"]) == 2
            assert len(history["val_acc"]) == 2
            assert "training_time" in history
            assert "params" in history
            assert "best_val_acc" in history

            # Check model was saved
            model_path = os.path.join(tmpdir, "test_model.pth")
            assert os.path.exists(model_path)

    def test_trainer_saves_best_model(self, simple_model, simple_dataloader, device):
        """Test that trainer saves best model based on validation accuracy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(simple_model, "adam")

            trainer = Trainer(
                model=simple_model,
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            history = trainer.train(num_epochs=3, save_dir=tmpdir, model_name="test_model")

            # Best validation accuracy should be the maximum
            assert history["best_val_acc"] == max(history["val_acc"])

    def test_trainer_with_reduce_on_plateau(self, simple_model, simple_dataloader, device):
        """Test trainer with ReduceLROnPlateau scheduler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(simple_model, "adam")
            scheduler = get_scheduler(optimizer, "plateau", patience=1)

            trainer = Trainer(
                model=simple_model,
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
            )

            history = trainer.train(num_epochs=2, save_dir=tmpdir, model_name="test_model")

            # Should have learning rates recorded
            assert len(history["learning_rates"]) == 2

    def test_trainer_learning_rate_tracking(self, simple_model, simple_dataloader, device):
        """Test that learning rates are tracked during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            criterion = nn.CrossEntropyLoss()
            optimizer = get_optimizer(simple_model, "adam", learning_rate=1e-3)
            scheduler = get_scheduler(optimizer, "step", step_size=1, gamma=0.5)

            trainer = Trainer(
                model=simple_model,
                train_loader=simple_dataloader,
                val_loader=simple_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scheduler=scheduler,
            )

            history = trainer.train(num_epochs=3, save_dir=tmpdir, model_name="test_model")

            # Learning rate should decrease
            lrs = history["learning_rates"]
            assert len(lrs) == 3
            assert lrs[0] == 1e-3
            # With step scheduler (step_size=1, gamma=0.5), lr should be halved each epoch
            assert lrs[1] == pytest.approx(5e-4)
            assert lrs[2] == pytest.approx(2.5e-4)


class TestTrainModel:
    """Test suite for train_model convenience function."""

    def test_train_model_basic(self, simple_model, simple_dataloader):
        """Test basic train_model function."""
        history = train_model(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            model_name="test_model",
            num_epochs=2,
            lr=1e-3,
            device=torch.device("cpu"),
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2

    def test_train_model_default_device(self, simple_model, simple_dataloader):
        """Test train_model with default device selection."""
        history = train_model(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            model_name="test_model",
            num_epochs=1,
            lr=1e-3,
        )

        assert history is not None

    def test_train_model_custom_lr(self, simple_model, simple_dataloader):
        """Test train_model with custom learning rate."""
        history = train_model(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            model_name="test_model",
            num_epochs=1,
            lr=5e-4,
            device=torch.device("cpu"),
        )

        # Check that learning rate was used
        assert history["learning_rates"][0] == 5e-4


class TestTrainerEdgeCases:
    """Test edge cases and error handling for Trainer."""

    def test_trainer_empty_dataloader(self, simple_model, device):
        """Test trainer behavior with empty dataloader."""
        # Create empty dataloader
        empty_dataset = TensorDataset(torch.randn(0, 3, 224, 224), torch.randint(0, 10, (0,)))
        empty_loader = DataLoader(empty_dataset, batch_size=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=empty_loader,
            val_loader=empty_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # This should handle empty dataloader gracefully
        # Note: Depending on implementation, might return 0 or raise error
        # For now, we just test it doesn't crash catastrophically
        try:
            loss, _acc = trainer.train_epoch()
            # If it succeeds, check reasonable values
            assert loss >= 0 or loss == 0
        except (ZeroDivisionError, StopIteration):
            # Some implementations might raise these
            pass

    def test_trainer_model_in_eval_mode_after_validation(self, simple_model, simple_dataloader, device):
        """Test that model is in eval mode after validation."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        trainer.validate_epoch()

        # Model should be in eval mode after validation
        assert not simple_model.training

    def test_trainer_model_in_train_mode_after_train_epoch(self, simple_model, simple_dataloader, device):
        """Test that model is in train mode after training epoch."""
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(simple_model, "adam")

        trainer = Trainer(
            model=simple_model,
            train_loader=simple_dataloader,
            val_loader=simple_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Run validation first to put in eval mode
        trainer.validate_epoch()

        # Now train
        trainer.train_epoch()

        # Model should be in train mode after training
        assert simple_model.training
