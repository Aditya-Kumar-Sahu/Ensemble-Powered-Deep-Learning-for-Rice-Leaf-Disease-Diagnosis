# Deployment Guide

This guide covers various deployment options for the Rice Leaf Disease Classification system.

## Table of Contents

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [FastAPI REST API](#fastapi-rest-api)
4. [Gradio Web Application](#gradio-web-application)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Checklist](#production-checklist)

## Local Development

### Prerequisites

- Python 3.9+
- pip
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/Aditya-Kumar-Sahu/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis.git
cd Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train a single model
python scripts/train.py \
    --data-dir /path/to/dataset \
    --model resnet50 \
    --epochs 15 \
    --batch-size 32

# Train all models for ensemble
for model in resnet50 mobilenetv2 efficientnetb0; do
    python scripts/train.py --data-dir /path/to/dataset --model $model
done
```

### Evaluation

```bash
# Evaluate single model
python scripts/evaluate.py \
    --data-dir /path/to/dataset \
    --model resnet50 \
    --output-dir results

# Evaluate ensemble
python scripts/evaluate.py \
    --data-dir /path/to/dataset \
    --ensemble \
    --output-dir results
```

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t rice-disease-classifier:latest .

# Or use docker-compose
docker-compose build
```

### Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Individual Services

```bash
# Run API only
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    rice-disease-classifier:latest \
    python apps/fastapi_app.py

# Run Gradio app only
docker run -p 7860:7860 \
    -v $(pwd)/models:/app/models \
    rice-disease-classifier:latest \
    python apps/gradio_app.py
```

## FastAPI REST API

### Start the API Server

```bash
python apps/fastapi_app.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### List Available Models
```bash
curl http://localhost:8000/models
```

#### Predict Disease
```bash
curl -X POST http://localhost:8000/predict \
    -F "file=@path/to/image.jpg" \
    -F "model=resnet50" \
    -F "top_k=5"
```

#### Get Disease Information
```bash
curl http://localhost:8000/disease-info/BrownSpot
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Python Client

```python
import requests

# Upload and predict
with open("leaf_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"model": "resnet50", "top_k": 5}
    )

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Gradio Web Application

### Start the Web App

```bash
python apps/gradio_app.py
```

The web interface will be available at `http://localhost:7860`

### Features

- Drag-and-drop image upload
- Real-time predictions
- Top-5 class probabilities
- Disease information and treatment recommendations
- Model selection (ResNet50, MobileNetV2, EfficientNet-B0)

### Public Sharing

To create a public URL for testing:

```python
# In gradio_app.py, modify the launch call:
app.launch(share=True)
```

## Cloud Deployment

### AWS EC2

1. Launch EC2 instance (Ubuntu 20.04+)
2. Install Docker:
```bash
sudo apt update
sudo apt install docker.io docker-compose -y
sudo systemctl start docker
sudo systemctl enable docker
```

3. Clone repository and deploy:
```bash
git clone <repo-url>
cd Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis
docker-compose up -d
```

4. Configure security group to allow ports 8000 and 7860

### Google Cloud Run

1. Build container:
```bash
gcloud builds submit --tag gcr.io/PROJECT-ID/rice-disease-api
```

2. Deploy:
```bash
gcloud run deploy rice-disease-api \
    --image gcr.io/PROJECT-ID/rice-disease-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

### Heroku

1. Create `Procfile`:
```
web: python apps/fastapi_app.py
```

2. Deploy:
```bash
heroku create rice-disease-classifier
git push heroku main
```

### Azure Container Instances

```bash
az container create \
    --resource-group myResourceGroup \
    --name rice-disease-api \
    --image rice-disease-classifier:latest \
    --dns-name-label rice-disease-unique \
    --ports 8000
```

## Production Checklist

### Security
- [ ] Enable HTTPS/TLS
- [ ] Implement authentication (API keys, JWT)
- [ ] Add rate limiting
- [ ] Validate and sanitize inputs
- [ ] Set up CORS policies
- [ ] Regular security updates

### Performance
- [ ] Model optimization (quantization, pruning)
- [ ] Add caching layer (Redis)
- [ ] Implement request batching
- [ ] Use CDN for static assets
- [ ] Load balancing for multiple replicas

### Monitoring
- [ ] Set up logging (ELK stack, CloudWatch)
- [ ] Add metrics collection (Prometheus)
- [ ] Configure alerts
- [ ] Monitor model performance drift
- [ ] Track API response times

### Reliability
- [ ] Health checks and liveness probes
- [ ] Graceful shutdown handling
- [ ] Automatic restarts
- [ ] Backup models and data
- [ ] Disaster recovery plan

### Scalability
- [ ] Kubernetes deployment
- [ ] Auto-scaling policies
- [ ] Database for request logs
- [ ] Message queue for async processing
- [ ] Model versioning system

## Troubleshooting

### Common Issues

**ModuleNotFoundError**
```bash
# Ensure you're in the project root and using the correct Python environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Model not found**
```bash
# Ensure models are trained and in the correct directory
ls -la models/
# Should see resnet50.pth, mobilenetv2.pth, efficientnetb0.pth
```

**Out of memory**
```bash
# Reduce batch size or use CPU
python scripts/train.py --data-dir /path/to/dataset --batch-size 16 --device cpu
```

**Port already in use**
```bash
# Kill process using the port
sudo lsof -ti:8000 | xargs kill -9
```

## Performance Optimization

### Model Optimization

```python
# Quantize model for faster inference
import torch.quantization as quantization

model_fp32 = get_model("resnet50", num_classes=15)
model_int8 = quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
```

### Caching

Add Redis for caching predictions:

```python
import redis
import hashlib

redis_client = redis.Redis(host='localhost', port=6379)

def get_cached_prediction(image_hash, model_name):
    key = f"{model_name}:{image_hash}"
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    return None
```

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/Aditya-Kumar-Sahu/Ensemble-Powered-Deep-Learning-for-Rice-Leaf-Disease-Diagnosis/issues)
- Documentation: See README.md
- Email: Contact repository maintainers
