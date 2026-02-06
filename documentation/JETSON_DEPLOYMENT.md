# Jetson Deployment Guide

Complete guide for deploying Angel Intelligence on NVIDIA Jetson devices (Orin Nano, AGX Orin, Jetson Thor).

For Thor-only deployments with shared services architecture, see [THOR_DEPLOYMENT.md](THOR_DEPLOYMENT.md).

## Quick Start

```bash
# On your Jetson device
cd ~/Desktop/Angel-Intelligence

# Configure environment
cp .env.example .env
nano .env
# Set your database credentials, API token, etc.

# Build and run
sudo docker-compose -f docker-compose.jetson.yml up -d

# Check logs
sudo docker-compose -f docker-compose.jetson.yml logs -f

# Check status
curl http://localhost:8080/health
```

## Modular Docker Images

For production Kubernetes deployments, use pod-specific Dockerfiles:

| Dockerfile | Pod Type | Size | Description |
|------------|----------|------|-------------|
| `Dockerfile.worker-jetson` | Worker | ~400MB | HTTP orchestrator (uses shared vLLM/Transcription) |
| `Dockerfile.transcription-jetson` | Transcription | ~4GB | WhisperX + pyannote |

Build commands:
```bash
# Worker image (lightweight - calls shared services)
docker build -f Dockerfile.worker-jetson -t angel-intelligence:worker-arm64 .

# Transcription image (GPU required)
docker build -f Dockerfile.transcription-jetson -t angel-intelligence:transcription-arm64 .
```

See [requirements/README.md](../requirements/README.md) for details on modular requirements.

## Hardware Requirements

| Device | GPU Memory | Recommended Pods | Notes |
|--------|-----------|------------------|-------|
| Orin Nano | 8GB | 0 | Insufficient memory |
| Orin Nano Super | 16GB | 1 (interactive only) | Chat/summaries only |
| AGX Orin 32GB | 32GB | 2 (1 batch + 1 interactive) | Production ready |
| AGX Orin 64GB | 64GB | 4 (2 batch + 2 interactive) | **Recommended** |
| Jetson Thor | 120GB | 8+ | High-volume production |

## Configuration

### Essential .env Settings

```env
# Environment
ANGEL_ENV=production
WORKER_ID=jetson-1

# Models - Use standard Instruct model (Omni requires special setup)
ANALYSIS_MODEL=Qwen/Qwen2.5-7B-Instruct
CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
ANALYSIS_MODE=transcript

# Model preloading (disable if you get OpenMP errors)
PRELOAD_CHAT_MODEL=true

# Database
AI_DB_HOST=your-database-host
AI_DB_PORT=3306
AI_DB_DATABASE=ai
AI_DB_USERNAME=your-username
AI_DB_PASSWORD=your-password

# API Token (generate secure token)
API_AUTH_TOKEN=your-64-char-token-here
```

## Known Issues & Fixes

### Issue 1: OpenMP/scikit-learn Error

**Symptom:** `cannot allocate memory in static TLS block` error on startup

**Fix Option A** - Add environment variable to docker-compose:

```yaml
# docker-compose.jetson.yml
services:
  angel-intelligence:
    environment:
      - LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

**Fix Option B** - Disable model preloading:

```env
# .env
PRELOAD_CHAT_MODEL=false
```

### Issue 2: Chat Model Fails to Preload

**Symptom:** `Failed to preload chat model` error but API still starts

**Impact:** Model loads on first request instead (10-15 second delay)

**Fix:** Use Fix Option B above or ignore (system works fine)

### Issue 3: Voice Fingerprinting Warnings

**Symptom:** `resemblyzer not available - voice fingerprinting disabled`

**Impact:** None - this feature is optional

**Fix:** Not needed (or install build tools and resemblyzer in container)

### Issue 4: Qwen2.5-Omni Warnings

**Symptom:** `Qwen2.5-Omni not available` warnings

**Impact:** None if using `ANALYSIS_MODE=transcript`

**Fix:** Use standard Instruct model (recommended) or install special transformers version

## Advanced: Full Feature Setup

To enable ALL features (audio analysis, voice fingerprinting):

```bash
# Access running container
sudo docker exec -it angel-intelligence bash

# Inside container:
# 1. Install Qwen2.5-Omni
pip3 install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip3 install qwen-omni-utils

# 2. Install voice fingerprinting (requires build tools on host first)
exit
sudo apt install build-essential python3-dev
sudo docker exec -it angel-intelligence bash
pip3 install webrtcvad resemblyzer

# 3. Exit and update .env
exit
nano .env
```

Update .env:
```env
ANALYSIS_MODE=audio
ANALYSIS_MODEL=Qwen/Qwen2.5-Omni-7B
```

Restart:
```bash
sudo docker-compose -f docker-compose.jetson.yml restart
```

## Performance Tuning

### Enable Quantization (More Pods per Device)

```env
# .env
ANALYSIS_MODEL_QUANTIZATION=int4
CHAT_MODEL_QUANTIZATION=int4
```

**Impact:** 
- Memory usage: ~14GB → ~7GB per model
- Speed: Slightly faster inference
- Quality: Minimal impact (<5% difference)

### Smaller Whisper Model

```env
WHISPER_MODEL=small  # Instead of medium
```

**Savings:** ~3GB per worker

## Monitoring

```bash
# View logs
sudo docker-compose -f docker-compose.jetson.yml logs -f

# Check GPU usage
sudo tegrastats

# Check container stats
sudo docker stats angel-intelligence

# Health check
curl http://localhost:8080/health
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
sudo docker-compose -f docker-compose.jetson.yml logs

# Rebuild from scratch
sudo docker-compose -f docker-compose.jetson.yml down
sudo docker system prune -a
sudo docker-compose -f docker-compose.jetson.yml up --build
```

### Out of Memory

1. Enable quantization (see above)
2. Use smaller Whisper model
3. Reduce concurrent jobs: `MAX_CONCURRENT_JOBS=1`

### Models Won't Download

```bash
# Check HuggingFace cache
ls -lh ~/.cache/huggingface

# Clear cache and retry
rm -rf ~/.cache/huggingface
sudo docker-compose -f docker-compose.jetson.yml restart
```

## Production Deployment

### Multiple Jetson Cluster

For high availability, deploy multiple Jetson devices:

```yaml
# On each Jetson, set unique WORKER_ID
WORKER_ID=jetson-1  # jetson-2, jetson-3, etc.

# Use load balancer (nginx) to distribute traffic
```

### Kubernetes on Jetson (Coming Soon)

See `k8s/` directory for Kubernetes manifests.

## Maintenance

### Updating

```bash
# Pull latest code
git pull origin master

# Rebuild
sudo docker-compose -f docker-compose.jetson.yml down
sudo docker-compose -f docker-compose.jetson.yml up --build -d
```

### Backup

```bash
# Backup models cache (optional - can re-download)
tar -czf models-backup.tar.gz ~/.cache/huggingface

# Backup database (on database server)
mysqldump ai > ai-backup.sql
```

## Support

For issues specific to Jetson deployment, check:
- NVIDIA Jetson Forums: https://forums.developer.nvidia.com/
- Jetson compatibility matrix for PyTorch/CUDA versions
- Angel Intelligence GitHub Issues
