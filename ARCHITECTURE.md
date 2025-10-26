# System Architecture

## Overview

The Rice Leaf Disease Classification system is a production-ready deep learning application designed following modern software engineering and MLOps best practices.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   CLI Tools  │  │  Web Browser │  │ API Clients  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Scripts    │  │    Gradio    │  │   FastAPI    │     │
│  │  (train.py,  │  │   Web App    │  │  REST API    │     │
│  │ evaluate.py) │  │              │  │              │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Library (src/)                     │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    Data     │  │    Models    │  │   Training   │      │
│  │             │  │              │  │              │      │
│  │ • Loaders   │  │ • ResNet50   │  │ • Trainer    │      │
│  │ • Transforms│  │ • MobileNet  │  │ • Optimizer  │      │
│  │ • Dataset   │  │ • EfficientNet│  │ • Scheduler  │      │
│  │             │  │ • Ensemble   │  │              │      │
│  └─────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐                         │
│  │ Evaluation  │  │   Utilities  │                         │
│  │             │  │              │                         │
│  │ • Metrics   │  │ • Checkpoint │                         │
│  │ • Viz       │  │ • Logging    │                         │
│  │ • Reports   │  │ • Device     │                         │
│  └─────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   PyTorch    │  │  torchvision │  │   Storage    │     │
│  │              │  │              │  │              │     │
│  │ • Tensors    │  │ • Transforms │  │ • Models/    │     │
│  │ • Autograd   │  │ • Pretrained │  │ • Logs/      │     │
│  │ • CUDA       │  │  Models      │  │ • Results/   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Layer (`src/data/`)

**Responsibilities:**
- Loading and preprocessing images
- Data augmentation
- Creating data loaders

**Key Components:**
- `dataset.py`: Custom dataset classes
- `loaders.py`: DataLoader creation and management
- `augmentations.py`: Transformation pipelines

**Data Flow:**
```
Raw Images → Transforms → Tensor → DataLoader → Model
```

### 2. Model Layer (`src/models/`)

**Responsibilities:**
- Model architecture definitions
- Model factory pattern
- Ensemble logic

**Key Components:**
- `base_model.py`: Model factory and registry
- `resnet.py`, `mobilenet.py`, `efficientnet.py`: Individual architectures
- `ensemble.py`: Ensemble model with multiple voting strategies

**Model Selection:**
```python
model = get_model(
    model_name="resnet50",
    num_classes=15,
    pretrained=True,
    dropout=0.2
)
```

### 3. Training Layer (`src/training/`)

**Responsibilities:**
- Training loop management
- Optimization and scheduling
- Checkpoint management

**Key Components:**
- `trainer.py`: Main training orchestration
- `optimizer.py`: Optimizer factory
- `scheduler.py`: Learning rate scheduler factory

**Training Pipeline:**
```
Model → Trainer → Train Loop → Validation → Checkpoint
```

### 4. Evaluation Layer (`src/evaluation/`)

**Responsibilities:**
- Model evaluation
- Metrics calculation
- Visualization generation
- Report creation

**Key Components:**
- `metrics.py`: Accuracy, F1, confusion matrix, etc.
- `visualizations.py`: Plots and charts
- `reports.py`: Classification reports and summaries

### 5. Utility Layer (`src/utils/`)

**Responsibilities:**
- Cross-cutting concerns
- Helper functions
- Common utilities

**Key Components:**
- `device.py`: Device management (CPU/GPU)
- `seed.py`: Reproducibility
- `checkpoint.py`: Model saving/loading
- `logging.py`: Structured logging

## Application Interfaces

### CLI Applications (`scripts/`)

**Purpose:** Command-line tools for ML workflows

**Components:**
1. `train.py`: Model training
2. `evaluate.py`: Model evaluation
3. `inference.py`: Single image prediction

**Example Usage:**
```bash
python scripts/train.py --data-dir /data --model resnet50
python scripts/evaluate.py --data-dir /data --ensemble
python scripts/inference.py --image leaf.jpg --model resnet50
```

### Web Application (`apps/gradio_app.py`)

**Purpose:** User-friendly web interface for farmers and agronomists

**Features:**
- Image upload (drag-and-drop)
- Real-time prediction
- Top-5 predictions display
- Disease information
- Treatment recommendations

**Technology:** Gradio

### REST API (`apps/fastapi_app.py`)

**Purpose:** Programmatic access for integration with other systems

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Make prediction
- `GET /disease-info/{name}` - Disease information
- `GET /models` - List available models
- `GET /classes` - List disease classes

**Technology:** FastAPI + Uvicorn

## Data Flow

### Training Flow

```
1. Load Dataset
   └── ImageFolder(dataset_path)
       └── Apply Transforms

2. Create DataLoaders
   └── Train/Val Split
       └── Batch Creation

3. Initialize Model
   └── Load Pretrained Weights (optional)
       └── Modify Final Layer

4. Training Loop
   ├── Forward Pass
   ├── Loss Calculation
   ├── Backward Pass
   └── Optimizer Step

5. Validation
   ├── Compute Metrics
   └── Save Best Model

6. Save History
   └── Training curves
   └── Model checkpoints
```

### Inference Flow

```
1. Load Image
   └── PIL.Image.open()

2. Preprocess
   └── Apply Transforms
       └── Normalize

3. Model Inference
   ├── Load Checkpoint
   ├── Forward Pass
   └── Softmax

4. Post-process
   ├── Get Top-K Predictions
   ├── Map to Class Names
   └── Format Results

5. Return Predictions
   └── JSON Response
```

### Ensemble Prediction Flow

```
1. Load Multiple Models
   ├── ResNet50
   ├── MobileNetV2
   └── EfficientNet-B0

2. Individual Predictions
   ├── Model 1 → Probabilities
   ├── Model 2 → Probabilities
   └── Model 3 → Probabilities

3. Voting Strategy
   ├── Soft: Average probabilities
   ├── Hard: Majority vote
   └── Weighted: Weighted average

4. Final Prediction
   └── Argmax of combined probabilities
```

## Configuration Management

### Configuration Hierarchy

```
configs/
├── base_config.yaml          # Global settings
├── training_config.yaml      # Training parameters
└── model_configs/
    ├── resnet50.yaml         # Model-specific config
    ├── mobilenetv2.yaml
    └── efficientnetb0.yaml
```

### Configuration Loading

```python
# Load base config
config = load_config("configs/base_config.yaml")

# Override with CLI arguments
config["training"]["num_epochs"] = args.epochs

# Model-specific config
model_config = load_config(f"configs/model_configs/{model_name}.yaml")
```

## Deployment Architecture

### Local Development

```
Developer Machine
├── Python Environment
├── Dataset (local)
└── Trained Models
```

### Docker Deployment

```
Docker Host
├── API Container (port 8000)
│   └── FastAPI Server
├── Web Container (port 7860)
│   └── Gradio App
└── Shared Volumes
    ├── /models (trained models)
    └── /data (dataset)
```

### Cloud Deployment (AWS Example)

```
AWS Cloud
├── EC2 Instance / ECS
│   ├── Docker Containers
│   └── Load Balancer
├── S3
│   ├── Model Storage
│   └── Dataset Storage
├── CloudWatch
│   └── Logs and Metrics
└── API Gateway (optional)
    └── Rate Limiting
```

## Security Architecture

### API Security

```
Request Flow:
1. Client Request
2. Rate Limiting Check
3. Authentication (if enabled)
4. Input Validation
5. Model Inference
6. Response Sanitization
7. Logging
```

### Security Layers

1. **Input Validation**
   - Image format verification
   - File size limits
   - Content type checking

2. **Authentication** (optional)
   - API key validation
   - JWT tokens
   - OAuth 2.0

3. **Rate Limiting**
   - Per-IP limits
   - Per-user limits
   - DDoS protection

4. **Data Privacy**
   - No persistent storage of images
   - Anonymized logging
   - GDPR compliance

## Scalability Patterns

### Horizontal Scaling

```
Load Balancer
├── API Instance 1
├── API Instance 2
└── API Instance N
    └── Shared Model Cache (Redis)
```

### Batch Processing

```
Request Queue (RabbitMQ/SQS)
├── Worker 1
├── Worker 2
└── Worker N
    └── Batch Inference
```

### Model Optimization

1. **Quantization**: INT8 precision
2. **Pruning**: Remove unnecessary weights
3. **Knowledge Distillation**: Train smaller model
4. **ONNX Export**: Cross-platform deployment

## Monitoring and Observability

### Metrics to Track

1. **System Metrics**
   - CPU/GPU usage
   - Memory consumption
   - Request latency
   - Throughput (requests/sec)

2. **Model Metrics**
   - Prediction accuracy
   - Confidence scores
   - Class distribution
   - Model drift

3. **Business Metrics**
   - User engagement
   - Popular disease classes
   - Geographic distribution
   - Peak usage times

### Logging Architecture

```
Application Logs
├── Structured JSON logs
├── Log Aggregation (ELK/CloudWatch)
└── Alert System
    ├── Error rate threshold
    ├── Latency threshold
    └── Model performance threshold
```

## Testing Strategy

### Test Pyramid

```
        E2E Tests (Few)
           /\
          /  \
         /    \
        / Inte \
       / gration\
      /  Tests   \
     /   (Some)   \
    /______________\
    Unit Tests (Many)
```

### Test Types

1. **Unit Tests** (`tests/`)
   - Model creation
   - Data transformations
   - Utility functions

2. **Integration Tests** (planned)
   - End-to-end training
   - API endpoints
   - Data pipelines

3. **Performance Tests** (planned)
   - Inference latency
   - Throughput
   - Memory usage

## CI/CD Pipeline

```
Code Push
    ↓
GitHub Actions
    ├── Linting (flake8, black)
    ├── Type Checking (mypy)
    ├── Unit Tests (pytest)
    ├── Coverage Report
    ├── Docker Build
    └── Security Scan (Trivy)
    ↓
Deploy to Staging
    ↓
Manual Approval
    ↓
Deploy to Production
```

## Future Architecture Enhancements

1. **Model Registry**
   - MLflow for model versioning
   - A/B testing framework
   - Model performance tracking

2. **Feature Store**
   - Cached predictions
   - User feedback
   - Historical data

3. **Real-time Pipeline**
   - Streaming predictions
   - WebSocket support
   - Event-driven architecture

4. **Edge Deployment**
   - TensorFlow Lite models
   - Mobile applications
   - Offline inference

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
