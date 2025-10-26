# Dockerfile for Rice Leaf Disease Classification

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY apps/ ./apps/

# Create necessary directories
RUN mkdir -p models logs results

# Expose ports
EXPOSE 8000 7860

# Default command (can be overridden)
CMD ["python", "apps/fastapi_app.py"]
