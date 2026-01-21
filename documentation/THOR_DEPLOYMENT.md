# AGX Thor 128GB Deployment Guide

This guide covers deploying Angel Intelligence on NVIDIA AGX Thor nodes with 128GB unified memory.

## Hardware Overview

| Spec | Value |
|------|-------|
| GPU Memory | 128GB unified |
| Compute | ~2000 TOPS INT8 |
| Architecture | Blackwell-based |

## Recommended Model Configuration

| Component | Model | VRAM | Notes |
|-----------|-------|------|-------|
| Transcription | Whisper large-v3 | ~3 GB | Best accuracy |
| Diarization | Pyannote 3.1 | ~0.5 GB | Speaker identification |
| Analysis | Qwen2.5-32B INT4 | ~18 GB | Excellent JSON output |
| Chat | Qwen2.5-14B INT4 | ~8 GB | Interactive queries |

## Performance Estimates

### Per 5-minute Call

| Stage | Time |
|-------|------|
| Transcription (large-v3) | 20-30 sec |
| Diarization | 5-10 sec |
| Analysis (32B INT4) | 45-90 sec |
| **Total** | **~1.5-2.5 min** |

### Throughput

| Configuration | Calls/Hour | Calls/Day |
|---------------|------------|-----------|
| 5 concurrent workers | 150-200 | 3,600-4,800 |
| Shared model server | 300-400 | 7,200-9,600 |

## Deployment Options

### Option 1: Local Development (.env.thor)

```bash
# Copy Thor configuration
cp .env.thor .env

# Edit with your credentials
nano .env

# Download models
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
"

# Run worker
python -m src.worker.worker
```

### Option 2: Kubernetes (thor-deployment.yaml)

```bash
# Create secrets first
cp k8s/secret.yaml.example k8s/secret.yaml
# Edit with your credentials
kubectl apply -f k8s/secret.yaml

# Deploy Thor configuration
kubectl apply -f k8s/thor-deployment.yaml

# Check status
kubectl get pods -l tier=thor
kubectl logs -f deployment/angel-intelligence-thor-batch
```

## Model Pre-download

Before first run, download models to shared storage:

```bash
# On Thor node or with NFS mounted
export MODELS_PATH=/models

# Whisper large-v3
python -c "
import whisperx
whisperx.load_model('large-v3', 'cuda', compute_type='float16')
"

# Analysis model (32B)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-32B-Instruct',
    cache_dir='$MODELS_PATH/analysis'
)
"

# Chat model (14B)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-14B-Instruct',
    cache_dir='$MODELS_PATH/chat'
)
"
```

## Node Labelling (Kubernetes)

Label your Thor nodes for scheduling:

```bash
# Option 1: Use GPU product name (automatic with NVIDIA device plugin)
# Pods will schedule on nodes with nvidia.com/gpu.product=AGX-Thor

# Option 2: Custom label
kubectl label node thor-node-01 angel.ai/tier=thor
kubectl label node thor-node-02 angel.ai/tier=thor
```

Update nodeSelector in deployment if using custom labels:

```yaml
nodeSelector:
  angel.ai/tier: "thor"
```

## Memory Management

The 128GB allows flexible deployment:

| Strategy | Workers | Memory/Worker | Notes |
|----------|---------|---------------|-------|
| **Isolated** | 5 | ~22 GB | Each worker loads own models |
| **Shared Whisper** | 8 | ~15 GB | Whisper loaded once |
| **Full vLLM** | 12+ | ~8 GB | All LLM via inference server |

### Isolated (Default)

Each worker independently loads/unloads models:
- Simplest to deploy
- No shared state
- Models unloaded between stages

### Shared Model Server (vLLM - Recommended)

For higher throughput, deploy a shared vLLM server. This allows multiple workers 
to share a single model, significantly improving resource utilisation.

**Benefits:**
- ~10-12 workers instead of 5 (isolated)
- Single 32B model for both analysis and chat
- Automatic request batching
- Lower memory per worker (~8GB instead of ~22GB)

**Deploy vLLM:**

```bash
# Deploy vLLM server (must complete before workers)
kubectl apply -f k8s/vllm-deployment.yaml

# Wait for vLLM to be ready (model loading takes 2-5 minutes)
kubectl wait --for=condition=ready pod -l app=vllm-server --timeout=600s

# Enable vLLM in thor-deployment.yaml:
# 1. Uncomment the llm-api-url line in the ConfigMap
# 2. Uncomment the LLM_API_URL env vars in both deployments

# Deploy workers
kubectl apply -f k8s/thor-deployment.yaml
```

**Test vLLM:**

```bash
# Port-forward to test locally
kubectl port-forward svc/vllm-server 8000:8000

# Test the API
curl http://localhost:8000/v1/models
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

**Environment Variables for vLLM mode:**

```bash
# .env or k8s configmap
LLM_API_URL=http://vllm-server.voice-ai.svc.cluster.local:8000/v1
LLM_API_KEY=  # Optional, only if vLLM requires auth
```

## Monitoring

### GPU Memory

```bash
# On Thor node
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 5

# Or via kubectl
kubectl exec -it <pod> -- nvidia-smi
```

### Processing Metrics

```sql
-- Calls processed per hour
SELECT 
    DATE_FORMAT(processing_completed_at, '%Y-%m-%d %H:00') as hour,
    COUNT(*) as calls_processed,
    AVG(TIMESTAMPDIFF(SECOND, processing_started_at, processing_completed_at)) as avg_seconds
FROM ai_call_recordings
WHERE processing_status = 'completed'
GROUP BY hour
ORDER BY hour DESC
LIMIT 24;
```

## Troubleshooting

### Out of Memory

If you see CUDA OOM errors:

1. Reduce `MAX_CONCURRENT_JOBS` to 3-4
2. Use smaller analysis model (14B instead of 32B)
3. Check for memory leaks with `nvidia-smi`

### Slow Processing

1. Ensure models are on NVMe, not network storage
2. Check GPU utilisation (should be 80%+ during inference)
3. Consider reducing Whisper model size if transcription bottlenecked

### Model Loading Failures

1. Verify HuggingFace token has access to gated models
2. Check disk space for model cache
3. Ensure CUDA drivers match PyTorch version

## Scaling

### Horizontal (More Nodes)

Add more Thor nodes and increase deployment replicas:

```bash
kubectl scale deployment angel-intelligence-thor-batch --replicas=3
```

### Vertical (Faster Models)

With Thor's compute power, consider:
- FP8 quantization (faster than INT4)
- Larger batch sizes for Whisper
- Speculative decoding for analysis
