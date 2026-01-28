# Docker Deployment Guide

Deploy Angel Intelligence using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit (for GPU)
- 16GB RAM (minimum)
- 50GB disk space

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Angel-FulFilment-Services/angel-intelligence.git
cd angel-intelligence

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## Docker Compose Configuration

### Production Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
    environment:
      - ANGEL_ENV=production
    env_file:
      - .env
    depends_on:
      - mysql
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build: .
    command: python -m src.worker.worker
    env_file:
      - .env
    depends_on:
      - mysql
      - api
    restart: unless-stopped
    deploy:
      replicas: 4
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models:/models:ro
      - ./temp:/tmp/angel

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: ${AI_DB_PASSWORD}
      MYSQL_DATABASE: ${AI_DB_DATABASE}
    volumes:
      - mysql_data:/var/lib/mysql
      - ./documentation/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    restart: unless-stopped

volumes:
  mysql_data:
```

### High-Throughput Setup (Shared Services)

For production with multiple workers sharing GPU models:

```yaml
# docker-compose.yml (excerpt)
services:
  # Shared transcription service (WhisperX on GPU)
  transcription:
    build: .
    ports:
      - "8001:8001"
    environment:
      - WORKER_MODE=api
      - WHISPER_MODEL=medium
      - USE_GPU=true
    command: uvicorn src.api:app --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Workers call shared services instead of loading models
  worker:
    build: .
    environment:
      - TRANSCRIPTION_SERVICE_URL=http://transcription:8001
      - LLM_API_URL=http://vllm:8000/v1  # Text LLM (transcript analysis)
      - AUDIO_ANALYSIS_API_URL=http://audio-vllm:8000/v1  # Audio LLM (if ANALYSIS_MODE=audio)
    command: python -m src.worker.worker
    depends_on:
      - transcription
```

See `docker-compose.yml` for the complete configuration.

---

### Development Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    command: uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    ports:
      - "8000:8000"
    environment:
      - ANGEL_ENV=development
      - USE_MOCK_MODELS=true
    env_file:
      - .env
    volumes:
      - ./src:/app/src:ro
      - ./tests:/app/tests:ro
    depends_on:
      - mysql

  worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: python -m src.worker.worker
    environment:
      - USE_MOCK_MODELS=true
      - POLL_INTERVAL_SECONDS=5
    env_file:
      - .env
    volumes:
      - ./src:/app/src:ro
    depends_on:
      - mysql
      - api

  mysql:
    image: mysql:8.0
    ports:
      - "3306:3306"
    environment:
      MYSQL_ROOT_PASSWORD: dev
      MYSQL_DATABASE: ai
    volumes:
      - mysql_dev_data:/var/lib/mysql

volumes:
  mysql_dev_data:
```

---

## Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sox \
    libsox-fmt-all \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install WhisperX
RUN pip install git+https://github.com/m-bain/whisperx.git

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Copy application
COPY . .

# Create temp directory
RUN mkdir -p /tmp/angel

# Default command
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## GPU Configuration

### NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Building Images

### Build for Local Architecture

```bash
docker build -t angel-intelligence:latest .
```

### Build for ARM64 (Jetson)

```bash
docker buildx build --platform linux/arm64 -t angel-intelligence:arm64 .
```

### Push to Registry

```bash
# Tag
docker tag angel-intelligence:latest your-registry/angel-intelligence:v1.0.0

# Push
docker push your-registry/angel-intelligence:v1.0.0
```

---

## Running Containers

### Start All Services

```bash
docker-compose up -d
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f worker

# Last 100 lines
docker-compose logs --tail=100 worker
```

### Stop Services

```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Restart Service

```bash
docker-compose restart worker
```

---

## Scaling Workers

```bash
# Scale to 4 workers
docker-compose up -d --scale worker=4

# Check running containers
docker-compose ps
```

---

## Health Checks

```bash
# Check API health
curl http://localhost:8000/health

# Check container health
docker inspect --format='{{.State.Health.Status}}' angel-intelligence-api-1
```

---

## Debugging

### Enter Container

```bash
docker-compose exec api bash
docker-compose exec worker bash
```

### Run One-off Commands

```bash
# Run tests
docker-compose exec api pytest tests/ -v

# Check GPU
docker-compose exec worker python -c "import torch; print(torch.cuda.is_available())"

# Database query
docker-compose exec mysql mysql -uroot -p$AI_DB_PASSWORD ai -e "SELECT COUNT(*) FROM ai_call_recordings"
```

---

## Volume Management

### List Volumes

```bash
docker volume ls
```

### Backup Database

```bash
docker-compose exec mysql mysqldump -uroot -p$AI_DB_PASSWORD ai > backup.sql
```

### Restore Database

```bash
docker-compose exec -T mysql mysql -uroot -p$AI_DB_PASSWORD ai < backup.sql
```

---

## Updating

### Pull Latest Changes

```bash
git pull origin master
```

### Rebuild and Restart

```bash
docker-compose build --no-cache
docker-compose up -d
```

### Rolling Update (Zero Downtime)

```bash
# Update one worker at a time
docker-compose up -d --no-deps --build worker
```

---

## Environment Variables

Pass via `.env` file or environment:

```env
# .env
ANGEL_ENV=production
API_AUTH_TOKEN=your-64-character-token

AI_DB_HOST=mysql
AI_DB_DATABASE=ai
AI_DB_USERNAME=root
AI_DB_PASSWORD=secure-password

USE_GPU=true
ANALYSIS_MODE=audio
WHISPER_MODEL=medium
```

---

## Networking

### Expose API Externally

```yaml
services:
  api:
    ports:
      - "0.0.0.0:8000:8000"
```

### Internal Only

```yaml
services:
  api:
    ports:
      - "127.0.0.1:8000:8000"
```

### Custom Network

```yaml
networks:
  angel-network:
    driver: bridge

services:
  api:
    networks:
      - angel-network
```

---

## Resource Limits

```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs api

# Check events
docker-compose events
```

### GPU Not Available

```bash
# Verify NVIDIA runtime
docker info | grep -i nvidia

# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Database Connection Failed

```bash
# Check MySQL is running
docker-compose ps mysql

# Check MySQL logs
docker-compose logs mysql

# Test connection
docker-compose exec api python -c "from src.database import get_db_connection; get_db_connection()"
```

### Out of Disk Space

```bash
# Clean up unused resources
docker system prune -a

# Remove old images
docker image prune -a
```
