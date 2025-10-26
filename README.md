# Ensemble-Powered Deep Learning for Rice Leaf Disease Diagnosis

A production-ready deep learning system for classifying rice leaf diseases into 15 distinct categories. This project leverages PyTorch and ensemble learning techniques to provide accurate disease detection for agronomists and farmers.

## 🌟 Features

### Core Capabilities
- **Multi-Model Ensemble**: Combines predictions from MobileNetV2, ResNet50, and EfficientNet-B0
- **15 Disease Classes**: Comprehensive classification covering major rice leaf diseases
- **Production-Ready Code**: Modular architecture with proper separation of concerns
- **CLI Interface**: Easy-to-use command-line tools for training, evaluation, and inference
- **Configuration Management**: YAML-based configuration for reproducible experiments
- **Advanced Data Augmentation**: Domain-specific augmentations for leaf images

### Model Architectures
- **MobileNetV2**: Lightweight model optimized for efficiency
- **ResNet50**: Deep residual network for high accuracy
- **EfficientNet-B0**: Efficient scaling for balanced performance

### Ensemble Strategies
- **Soft Voting**: Average class probabilities (default)
- **Hard Voting**: Majority vote on class predictions
- **Weighted Voting**: Weighted average based on model performance

## 📁 Project Structure

```
.
├── src/                          # Source code
│   ├── data/                     # Data loading and augmentation
│   │   ├── dataset.py
│   │   ├── loaders.py
│   │   └── augmentations.py
│   ├── models/                   # Model architectures
│   │   ├── base_model.py
│   │   ├── resnet.py
│   │   ├── mobilenet.py
│   │   ├── efficientnet.py
│   │   └── ensemble.py
│   ├── training/                 # Training utilities
│   │   ├── trainer.py
│   │   ├── optimizer.py
│   │   └── scheduler.py
│   ├── evaluation/               # Evaluation and metrics
│   │   ├── metrics.py
│   │   ├── visualizations.py
│   │   └── reports.py
│   └── utils/                    # Utility functions
│       ├── device.py
│       ├── seed.py
│       ├── checkpoint.py
│       └── logging.py
├── configs/                      # Configuration files
│   ├── base_config.yaml
│   ├── training_config.yaml
│   └── model_configs/
├── scripts/                      # CLI scripts
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/                        # Unit tests
├── models/                       # Trained model checkpoints
├── logs/                         # Training logs
├── results/                      # Evaluation results
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Aditya-Kumar-Sahu/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis.git
   cd Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API credentials**:
   - Place your `kaggle.json` in `~/.kaggle/`
   - Download the dataset using the notebook or manually

### Training Models

Train a single model:
```bash
python scripts/train.py \
    --data-dir /path/to/dataset \
    --model resnet50 \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.00005
```

Train all models for ensemble:
```bash
# Train ResNet50
python scripts/train.py --data-dir /path/to/dataset --model resnet50

# Train MobileNetV2
python scripts/train.py --data-dir /path/to/dataset --model mobilenetv2

# Train EfficientNet-B0
python scripts/train.py --data-dir /path/to/dataset --model efficientnetb0
```

### Evaluation

Evaluate a single model:
```bash
python scripts/evaluate.py \
    --data-dir /path/to/dataset \
    --model resnet50 \
    --output-dir results
```

Evaluate ensemble:
```bash
python scripts/evaluate.py \
    --data-dir /path/to/dataset \
    --ensemble \
    --output-dir results
```

### Inference

Predict disease for a single image:
```bash
python scripts/inference.py \
    --image /path/to/image.jpg \
    --model resnet50 \
    --classes "BacterialLeafBlight" "BrownSpot" "LeafSmut" ...
```

## 📊 Performance

The ensemble model achieves superior performance compared to individual models:

| Model | Accuracy | F1-Score | Parameters |
|-------|----------|----------|------------|
| MobileNetV2 | ~92% | ~0.91 | 2.2M |
| ResNet50 | ~94% | ~0.93 | 23.5M |
| EfficientNet-B0 | ~93% | ~0.92 | 4.0M |
| **Ensemble (Soft Voting)** | **~95%** | **~0.94** | - |

*Note: Actual performance depends on dataset and training configuration*

## 🔧 Configuration

Edit `configs/base_config.yaml` to customize training:

```yaml
data:
  image_size: 224
  batch_size: 32
  val_split: 0.1

training:
  num_epochs: 15
  learning_rate: 0.00005
  optimizer: "adam"
  scheduler: "cosine"

ensemble:
  voting: "soft"  # Options: soft, hard, weighted
```

## 📈 Results & Visualizations

The system automatically generates:
- **Confusion matrices** for each model and ensemble
- **Training curves** (loss and accuracy over epochs)
- **Classification reports** with per-class metrics
- **Model comparison plots**

Results are saved to the `results/` directory.

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Support (Coming Soon)

```bash
# Build Docker image
docker build -t rice-disease-classifier .

# Run training
docker run -v /path/to/data:/data rice-disease-classifier \
    python scripts/train.py --data-dir /data
```

## 🌐 Deployment Options (Coming Soon)

### REST API (FastAPI)
```bash
python api/app.py
```

### Web Application (Gradio)
```bash
python apps/gradio_app.py
```

### Streamlit Dashboard
```bash
streamlit run apps/streamlit_app.py
```

## 📚 Dataset

The project uses the [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/maimunulkjisan/rice-leaf-dataset-from-mendeley-data) from Kaggle, which includes:
- **15 disease classes**
- **Thousands of labeled images**
- **High-resolution leaf images**

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add type hints to functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Kaggle Rice Leaf Disease Dataset
- **Frameworks**: PyTorch, torchvision
- **Inspiration**: Agricultural AI for crop disease management

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

## 🗺️ Roadmap

### Phase 1: Core Features (Completed)
- [x] Modular code architecture
- [x] Multiple model architectures
- [x] Ensemble implementation
- [x] CLI interface
- [x] Configuration management

### Phase 2: Advanced Features (In Progress)
- [ ] Advanced augmentation (Albumentations, AutoAugment)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model interpretability (Grad-CAM)
- [ ] Test-time augmentation

### Phase 3: Deployment (Planned)
- [ ] REST API (FastAPI)
- [ ] Web application (Gradio/Streamlit)
- [ ] Docker support
- [ ] Mobile optimization (TFLite/CoreML)

### Phase 4: MLOps (Planned)
- [ ] Experiment tracking (MLflow/W&B)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model registry
- [ ] Automated testing

### Phase 5: Domain Features (Planned)
- [ ] Disease knowledge base
- [ ] Treatment recommendations
- [ ] Geographic tracking
- [ ] Multi-language support

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{rice_leaf_disease_classifier,
  title = {Ensemble-Powered Deep Learning for Rice Leaf Disease Diagnosis},
  author = {Aditya Kumar Sahu},
  year = {2024},
  url = {https://github.com/Aditya-Kumar-Sahu/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis}
}
```

---

**Made with ❤️ for sustainable agriculture**
