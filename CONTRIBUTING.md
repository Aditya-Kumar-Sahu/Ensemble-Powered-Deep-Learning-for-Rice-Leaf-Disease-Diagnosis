# Contributing to Rice Leaf Disease Classification

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Testing](#testing)
6. [Code Style](#code-style)
7. [Pull Request Process](#pull-request-process)
8. [Areas for Contribution](#areas-for-contribution)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help maintain a positive community
- Report unacceptable behavior to project maintainers

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic knowledge of PyTorch and deep learning
- Familiarity with the agricultural domain (helpful but not required)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis.git
cd Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis
```

3. Add upstream remote:
```bash
git remote add upstream https://github.com/Aditya-Kumar-Sahu/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis.git
```

## Development Setup

### Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies including development tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy isort
```

### Project Structure

```
src/
├── data/          # Data loading and augmentation
├── models/        # Model architectures
├── training/      # Training utilities
├── evaluation/    # Metrics and visualization
└── utils/         # Helper functions

scripts/           # CLI tools
apps/              # Web applications and APIs
tests/             # Unit and integration tests
configs/           # Configuration files
```

## Making Changes

### Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use prefixes:
- `feature/` - New features
- `bugfix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications

### Implement Your Changes

1. Write clean, readable code
2. Add docstrings to functions and classes
3. Include type hints
4. Add unit tests for new functionality
5. Update documentation as needed

### Example: Adding a New Model

```python
"""New model architecture."""

import torch.nn as nn
import torchvision.models as models


def get_new_model(
    num_classes: int,
    pretrained: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """
    Create a new model for classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate for regularization
        
    Returns:
        Model instance
    """
    # Implementation here
    pass
```

## Testing

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

### Write New Tests

Create test files in `tests/` directory:

```python
"""Tests for new functionality."""

import pytest
from src.models import get_new_model


def test_new_model_creation():
    """Test that new model can be created."""
    model = get_new_model(num_classes=15)
    assert model is not None


def test_new_model_forward_pass():
    """Test forward pass through new model."""
    model = get_new_model(num_classes=15)
    input_tensor = torch.randn(4, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (4, 15)
```

### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Include integration tests for complex workflows

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters (instead of 79)
- Use double quotes for strings
- Use meaningful variable names

### Formatting

```bash
# Format code with Black
black src/ tests/ scripts/

# Sort imports with isort
isort src/ tests/ scripts/

# Check style with flake8
flake8 src/ tests/ scripts/
```

### Type Hints

Always add type hints to function signatures:

```python
def process_image(
    image_path: str,
    target_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Process image for model input."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
) -> dict:
    """
    Train a model on the dataset.
    
    Args:
        model: PyTorch model to train
        train_loader: Data loader for training data
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training history
        
    Raises:
        ValueError: If epochs is less than 1
        
    Examples:
        >>> model = get_model("resnet50", num_classes=15)
        >>> history = train_model(model, train_loader, epochs=5)
    """
    pass
```

## Pull Request Process

### Before Submitting

1. **Update your branch**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests**:
```bash
pytest tests/ -v
```

3. **Check code style**:
```bash
black --check src/ tests/ scripts/
flake8 src/
```

4. **Update documentation**:
- Update README.md if needed
- Update CHANGELOG.md
- Add docstrings to new code

### Submit Pull Request

1. Push your branch:
```bash
git push origin feature/your-feature-name
```

2. Create PR on GitHub with:
   - Clear title describing the change
   - Description of what was changed and why
   - Link to related issues
   - Screenshots for UI changes
   - Test results

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Areas for Contribution

### High Priority

1. **Model Improvements**
   - Add new architectures (Vision Transformers, ConvNeXt)
   - Implement advanced augmentation (Albumentations)
   - Add model interpretability (Grad-CAM, SHAP)

2. **Deployment**
   - Mobile optimization (TFLite, CoreML)
   - Model quantization and pruning
   - Serverless deployment examples

3. **MLOps**
   - Experiment tracking (MLflow, W&B)
   - Model registry
   - A/B testing framework

### Medium Priority

4. **Data Pipeline**
   - Data validation and quality checks
   - Active learning implementation
   - Semi-supervised learning

5. **Testing**
   - Integration tests
   - Performance benchmarks
   - Adversarial robustness tests

6. **Documentation**
   - Video tutorials
   - Architecture diagrams
   - API usage examples

### Low Priority

7. **Features**
   - Multi-language support
   - Disease progression tracking
   - Batch inference optimization

8. **Research**
   - Transfer learning experiments
   - Few-shot learning
   - Domain adaptation

## Development Workflow

### Typical Workflow

1. Pick an issue or create one
2. Discuss approach in issue comments
3. Create feature branch
4. Implement changes with tests
5. Ensure all tests pass
6. Submit pull request
7. Address review feedback
8. Merge after approval

### Communication

- Use GitHub Issues for bugs and feature requests
- Use Pull Requests for code reviews
- Tag maintainers with @mention for urgent issues

## Review Process

- PRs require at least one approval
- CI/CD must pass
- Code coverage should not decrease
- Documentation must be updated
- Breaking changes require major version bump

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in publications using the code

## Questions?

- Open a GitHub Discussion
- Ask in PR comments
- Contact maintainers directly

Thank you for contributing! 🌾
