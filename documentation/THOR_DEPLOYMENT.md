# AGX Thor 128GB Deployment Guide

This guide provides a complete step-by-step process for deploying Angel Intelligence on NVIDIA AGX Thor nodes with 128GB unified memory in a **Thor-only architecture** (no Orin nodes).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GATEWAY NODE (x86_64)                              │
│  ┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌───────────┐  │
│  │   API POD       │  │ REDIS POD   │  │  MODEL STORAGE   │  │  INGRESS  │  │
│  │   FastAPI       │  │ RQ Queue    │  │  (NFS/PVC)       │  │ :443→API  │  │
│  │   routes.py     │  │ Job Storage │  │  /models/*       │  │           │  │
│  └─────────────────┘  └─────────────┘  └──────────────────┘  └───────────┘  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NVIDIA THOR AGX NODE (ARM64 - 128GB)                      │
│                           ~216 calls/hr capacity                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │               vLLM SERVICE (2 Pods) - ~80GB total                      │ │
│  │  ┌────────────────────────────┐  ┌────────────────────────────┐        │ │
│  │  │ vLLM Instance 1 (~40GB)    │  │ vLLM Instance 2 (~40GB)    │        │ │
│  │  │ Qwen2.5-72B-AWQ (Q4)       │  │ Qwen2.5-72B-AWQ (Q4)       │        │ │
│  │  │ :30800 OpenAI API          │  │ :30800 OpenAI API          │        │ │
│  │  └────────────────────────────┘  └────────────────────────────┘        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │          TRANSCRIPTION SERVICE (3 Pods) - ~30GB [BOTTLENECK]           │ │
│  │  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐        │ │
│  │  │ WhisperX ~10GB   │ │ WhisperX ~10GB   │ │ WhisperX ~10GB   │        │ │
│  │  │ large-v3 :30900  │ │ large-v3 :30900  │ │ large-v3 :30900  │        │ │
│  │  └──────────────────┘ └──────────────────┘ └──────────────────┘        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │ BATCH WORKERS (20) ~10GB   │  │ INTERACTIVE WORKERS (10) ~5GB        │  │
│  │ HTTP Orchestrators          │  │ Real-time WebSocket support          │  │
│  │ Call Transcription :30900   │  │ SQL Agent integration               │  │
│  │ Call vLLM :30800            │  │ Immediate response handling         │  │
│  └─────────────────────────────┘  └──────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │               TRAINING CRONJOB (Nightly @ 2AM)                         │ │
│  │   LoRA Fine-tuning → Updates /lora-adapters/ → vLLM Hot-Reload         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   MYSQL SERVER         │
                    │   (Separate Host)      │
                    │   mysql://db:3306      │
                    └────────────────────────┘
```

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Thor GPU Memory | 128GB unified |
| Thor Compute | ~2000 TOPS INT8 |
| Architecture | Blackwell-based (ARM64) |
| Gateway Node | x86_64, No GPU, 8GB+ RAM |
| Storage | NFS for models (~100GB) |
| Database | External MySQL Server |

## Memory Allocation (128GB Thor)

| Component | Memory | Notes |
|-----------|--------|-------|
| vLLM Instance 1 | ~40GB | Qwen2.5-72B-AWQ |
| vLLM Instance 2 | ~40GB | Qwen2.5-72B-AWQ |
| Transcription (3x) | ~30GB | WhisperX large-v3 |
| Workers (30x) | ~15GB | HTTP orchestrators |
| **Headroom** | ~3GB | System overhead |

## Performance Estimates

| Metric | Value |
|--------|-------|
| Throughput | ~216 calls/hr |
| Per 5-min call | ~1.5-2.5 min processing |
| Transcription (bottleneck) | ~72 calls/hr per pod |
| vLLM Analysis | ~180 calls/hr per pod |

---

# Step-by-Step Deployment Guide

## Prerequisites (Assumed Complete)

- [x] Kubernetes cluster setup (K3s recommended)
- [x] Thor node(s) added as worker nodes
- [x] NVIDIA device plugin installed
- [x] Gateway node configured

Verify your cluster:
```bash
kubectl get nodes
# Should show gateway node + Thor node(s)

kubectl describe nodes | grep nvidia.com/gpu
# Should show available GPUs on Thor nodes
```

---

## Phase 1: Prepare Infrastructure

### Step 1.1: Label Thor Nodes

Label your Thor nodes for pod scheduling:

```bash
# Using GPU product name (automatic with NVIDIA device plugin)
# Pods will schedule on nodes with nvidia.com/gpu.product=AGX-Thor

# OR use custom labels:
kubectl label node thor-node-01 angel.ai/tier=thor
kubectl label node thor-node-01 nvidia.com/gpu.product=AGX-Thor

# Verify labels
kubectl get nodes --show-labels | grep thor
```

### Step 1.2: Setup NFS Storage for Models

**On Gateway/Master Node (or separate NAS):**

```bash
# Install NFS server
sudo apt install nfs-kernel-server -y

# Create model storage directories
# Note: With vLLM, a single model serves both analysis AND chat requests
# No separate chat folder needed!
sudo mkdir -p /exports/angel-models/{whisper,analysis,adapters}
sudo chown -R nobody:nogroup /exports/angel-models
sudo chmod -R 777 /exports/angel-models

# Configure NFS exports
echo "/exports/angel-models *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

# Verify NFS is running
showmount -e localhost
```

**On Thor Node(s) - Install NFS Client:**

```bash
# Install NFS client
sudo apt install nfs-common -y

# Test mount (replace MASTER_IP)
sudo mkdir -p /mnt/test
sudo mount MASTER_IP:/exports/angel-models /mnt/test
ls /mnt/test
sudo umount /mnt/test
```

### Step 1.3: Create NFS PersistentVolume

Create `k8s/pv-models.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: angel-models-pv
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: 192.168.1.100  # Your NFS server IP
    path: /exports/angel-models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: angel-intelligence-thor-models
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  volumeName: angel-models-pv
```

Apply:
```bash
kubectl apply -f k8s/pv-models.yaml
```

---

## Phase 2: Pre-download Models

Models must be downloaded to NFS storage before deployment. Do this from your **control plane/gateway node** where NFS is hosted.

### Step 2.1: Download WhisperX Models

```bash
# On control plane (gateway/master node)
export MODELS_PATH=/exports/angel-models

# Ensure you have write permissions to the NFS share
sudo chown -R $USER:$USER $MODELS_PATH
# OR if using nobody:nogroup, download as root:
# sudo -E bash  (then run the commands below)

# Install Python venv package if not present
sudo apt install python3-venv python3-pip -y

# Create a temporary Python environment for downloads
python3 -m venv /tmp/model-download
source /tmp/model-download/bin/activate
pip install --upgrade pip
pip install huggingface-hub

# Create target directories first
mkdir -p $MODELS_PATH/whisper
mkdir -p $MODELS_PATH/analysis
mkdir -p $MODELS_PATH/adapters

# Download Whisper large-v3 using huggingface-hub
python3 << 'EOF'
import os
from huggingface_hub import snapshot_download

models_path = os.environ.get('MODELS_PATH', '/exports/angel-models')
snapshot_download(
    repo_id='Systran/faster-whisper-large-v3',
    local_dir=f'{models_path}/whisper/faster-whisper-large-v3'
)
print('Whisper large-v3 downloaded successfully')
EOF
```

### Step 2.2: Download vLLM Models

```bash
# Still on control plane (in the same venv)
# This is a large download (~35GB) - may take 30-60 minutes
python3 << 'EOF'
import os
from huggingface_hub import snapshot_download

models_path = os.environ.get('MODELS_PATH', '/exports/angel-models')
snapshot_download(
    repo_id='Qwen/Qwen2.5-72B-Instruct-AWQ',
    local_dir=f'{models_path}/analysis/Qwen2.5-72B-Instruct-AWQ'
)
print('Qwen2.5-72B-AWQ downloaded successfully')
EOF

# Cleanup temporary environment
deactivate
rm -rf /tmp/model-download
```

### Step 2.3: Accept HuggingFace Model Licenses

For speaker diarization, accept the pyannote license:
1. Go to https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the license terms
3. Note your HuggingFace token from https://huggingface.co/settings/tokens

### Note: Single Model for Analysis + Chat

With vLLM architecture, **one model serves both purposes**:
- **Analysis requests** → vLLM with analysis prompt + LoRA adapter
- **Chat/Interactive requests** → Same vLLM with chat prompt

No separate `chat` folder needed. If you have one from a previous setup, you can remove it:
```bash
rm -rf /exports/angel-models/chat
```

---

## Phase 3: Configure Kubernetes Secrets

### Step 3.1: Create Secrets File

```bash
# Copy the example
cp k8s/secret.yaml.example k8s/secret.yaml
```

### Step 3.2: Edit with Your Credentials

Edit `k8s/secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: angel-intelligence-secrets
  namespace: default
type: Opaque
stringData:
  # API Authentication (min 64 characters)
  # Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"
  api-auth-token: "YOUR_64_CHAR_TOKEN_HERE"
  
  # Database Configuration
  db-host: "your-mysql-host.example.com"
  db-port: "3306"
  db-database: "ai"
  db-username: "ai_user"
  db-password: "your-secure-password"
  
  # R2 Storage Configuration
  r2-endpoint: "https://your-account.r2.cloudflarestorage.com"
  r2-access-key-id: "your-access-key"
  r2-secret-access-key: "your-secret-key"
  r2-bucket: "call-recordings"
  
  # HuggingFace Token (for pyannote diarization)
  huggingface-token: "hf_your_token_here"
```

### Step 3.3: Apply Secrets

```bash
kubectl apply -f k8s/secret.yaml

# Verify
kubectl get secrets angel-intelligence-secrets
```

---

## Phase 4: Build and Push Docker Images

### Step 4.0: Setup Local Registry (Control Plane)

Set up a local Docker registry on the control plane so Thor nodes can pull images automatically:

```bash
# On control plane (gateway/master node)

# Start local registry (skip if already running)
docker run -d -p 5000:5000 --restart=always --name registry registry:2 2>/dev/null || echo "Registry already running"

# Configure Docker to allow insecure (HTTP) registry
export REGISTRY_IP=$(hostname -I | awk '{print $1}')
sudo tee /etc/docker/daemon.json << EOF
{
  "insecure-registries": ["${REGISTRY_IP}:5000", "localhost:5000"]
}
EOF

# Restart Docker to apply
sudo systemctl restart docker

# Verify registry is running
curl http://localhost:5000/v2/_catalog
# Should return: {"repositories":[]}

# Note your control plane IP (Thor needs this)
echo "Registry IP: ${REGISTRY_IP}"
```

**Configure K3s to trust the local registry:**

On each node (control plane + Thor), add the registry to containerd config:

```bash
# Create registries config
sudo mkdir -p /etc/rancher/k3s

sudo tee /etc/rancher/k3s/registries.yaml << EOF
mirrors:
  "192.168.9.50:5000":
    endpoint:
      - "http://192.168.9.50:5000"
EOF

# Replace CONTROL_PLANE_IP with your actual IP, e.g.:
# sudo sed -i 's/CONTROL_PLANE_IP/192.168.1.100/g' /etc/rancher/k3s/registries.yaml

# Restart K3s to apply
sudo systemctl restart k3s        # On control plane
sudo systemctl restart k3s-agent  # On Thor node
```

### Step 4.1: vLLM Image for ARM64

**Pull and push to local registry:**

```bash
# On control plane
export REGISTRY_IP=$(hostname -I | awk '{print $1}')

# Pull the Jetson vLLM image
docker pull dustynv/vllm:0.9.2-r36.4-cu128-24.04

# Tag and push to local registry
docker tag dustynv/vllm:0.9.2-r36.4-cu128-24.04 ${REGISTRY_IP}:5000/angel-intelligence:vllm-arm64
docker push ${REGISTRY_IP}:5000/angel-intelligence:vllm-arm64

# Verify
curl http://${REGISTRY_IP}:5000/v2/_catalog
# Should show: {"repositories":["angel-intelligence"]}
```

> **Note:** Choose the tag matching your JetPack/L4T version. Check https://hub.docker.com/r/dustynv/vllm/tags

**Option B: Build vLLM from source (on Thor node)**

```bash
# On Thor node (ARM64)
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Check if Dockerfile.arm exists, otherwise use main Dockerfile
# vLLM may have different Dockerfile names - check the repo
ls Dockerfile*

# Build (this takes 30-60 minutes on Thor)
docker build -t angel-intelligence:vllm-arm64 .

# Push to registry
docker tag angel-intelligence:vllm-arm64 your-registry/angel-intelligence:vllm-arm64
docker push your-registry/angel-intelligence:vllm-arm64
```

**Option C: Use NVIDIA NGC PyTorch base + pip install**

```bash
# Create a custom Dockerfile for vLLM on Jetson
cat > Dockerfile.vllm-jetson << 'EOF'
FROM nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.1-py3

RUN pip install --upgrade pip && \
    pip install vllm

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
EOF

# Build on Thor node
docker build -f Dockerfile.vllm-jetson -t angel-intelligence:vllm-arm64 .
```

> **Note:** vLLM ARM64 support is evolving. Check the [vLLM GitHub](https://github.com/vllm-project/vllm) and [dustynv/jetson-containers](https://github.com/dusty-nv/jetson-containers) for the latest options.

### Step 4.2: Build Pod-Specific Images for ARM64

Angel Intelligence uses modular Dockerfiles for minimal image sizes:

| Dockerfile | Pod | Size | Description |
|------------|-----|------|-------------|
| `Dockerfile.worker-jetson` | Worker | ~400MB | HTTP orchestrator |
| `Dockerfile.transcription-jetson` | Transcription | ~4GB | WhisperX + pyannote |
| `Dockerfile.api` | API | ~200MB | API gateway |

**Build and push to local registry:**

```bash
cd ~/Angel-Intelligence
export REGISTRY_IP=$(hostname -I | awk '{print $1}')

# Setup buildx for cross-platform builds (if not already done)
docker buildx create --use --name multiarch 2>/dev/null || docker buildx use multiarch

# Build Worker image (ARM64 for Thor)
# Use --load to load locally, then push with docker push (respects insecure registry config)
docker buildx build --platform linux/arm64 \
  -f Dockerfile.worker-jetson \
  -t ${REGISTRY_IP}:5000/angel-intelligence:worker-arm64 \
  --load .
docker push ${REGISTRY_IP}:5000/angel-intelligence:worker-arm64 # HERE

# Build Transcription image (ARM64 for Thor)
docker buildx build --platform linux/arm64 \
  -f Dockerfile.transcription-jetson \
  -t ${REGISTRY_IP}:5000/angel-intelligence:transcription-arm64 \
  --load .
docker push ${REGISTRY_IP}:5000/angel-intelligence:transcription-arm64

# Build API image (x86_64 for gateway - no buildx needed)
docker build -f Dockerfile.api \
  -t ${REGISTRY_IP}:5000/angel-intelligence:api .
docker push ${REGISTRY_IP}:5000/angel-intelligence:api

# Verify all images are in registry
curl http://${REGISTRY_IP}:5000/v2/angel-intelligence/tags/list
# Should show: {"name":"angel-intelligence","tags":["vllm-arm64","worker-arm64","transcription-arm64","api"]}
```

### Step 4.3: Update Kubernetes Manifests

Update image references in your deployment YAML files to use the local registry:

```yaml
# Replace CONTROL_PLANE_IP with your actual IP (e.g., 192.168.1.100)

# k8s/vllm-deployment.yaml - vLLM server
image: CONTROL_PLANE_IP:5000/angel-intelligence:vllm-arm64

# k8s/thor-deployment.yaml - Worker pods  
image: CONTROL_PLANE_IP:5000/angel-intelligence:worker-arm64

# k8s/transcription-deployment.yaml - Transcription pods
image: CONTROL_PLANE_IP:5000/angel-intelligence:transcription-arm64

# k8s/deployment.yaml - API pod (gateway node)
image: CONTROL_PLANE_IP:5000/angel-intelligence:api
```

When you deploy, K3s on Thor will automatically pull from the local registry.

---

## Phase 5: Deploy Services (In Order)

### Step 5.1: Apply ConfigMap

Review and customise `k8s/thor-deployment.yaml` ConfigMap section:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: angel-intelligence-thor-config
data:
  # Update these for your environment
  analysis-model: "Qwen/Qwen2.5-72B-Instruct-AWQ"
  whisper-model: "large-v3"
  pbx-live-url: "https://your-pbx.example.com/callrec/"
  pbx-archive-url: "https://your-archive.example.com/"
```

Apply the full thor-deployment.yaml (this includes the ConfigMap):
```bash
kubectl apply -f k8s/thor-deployment.yaml
```

### Step 5.2: Deploy vLLM Server (First)

The vLLM server provides shared LLM inference for all workers.

```bash
# Deploy vLLM
kubectl apply -f k8s/vllm-deployment.yaml

# Wait for model to load (takes 2-5 minutes)
kubectl wait --for=condition=ready pod -l app=vllm-server --timeout=600s

# Check logs for model loading progress
kubectl logs -f -l app=vllm-server

# Verify vLLM is responding
kubectl port-forward svc/vllm-server 8000:8000 &
curl http://localhost:8000/v1/models
```

Expected output:
```json
{
  "object": "list",
  "data": [
    {"id": "Qwen/Qwen2.5-72B-Instruct-AWQ", "object": "model", ...}
  ]
}
```

### Step 5.3: Deploy Transcription Service

The transcription service provides shared WhisperX for all workers.

```bash
# Deploy transcription pods (start with 1, scale up after verification)
kubectl apply -f k8s/transcription-deployment.yaml

# Wait for WhisperX model to load
kubectl wait --for=condition=ready pod -l component=transcription --timeout=300s

# Check logs
kubectl logs -f -l component=transcription

# Verify transcription service
kubectl port-forward svc/transcription-service 8001:8001 &
curl http://localhost:8001/internal/health
```

Expected output:
```json
{"status": "healthy", "model_loaded": true}
```

### Step 5.4: Update ConfigMap with Service URLs

Now that services are running, update the thor-deployment ConfigMap:

```bash
kubectl edit configmap angel-intelligence-thor-config
```

Set:
```yaml
data:
  llm-api-url: "http://vllm-server:8000/v1"
  transcription-service-url: "http://transcription-service:8001"
```

Or apply with patch:
```bash
kubectl patch configmap angel-intelligence-thor-config -p '{"data":{"llm-api-url":"http://vllm-server:8000/v1"}}'
```

### Step 5.5: Deploy Workers

```bash
# Deploy batch and interactive workers
kubectl apply -f k8s/thor-deployment.yaml

# Check pods are starting
kubectl get pods -l tier=thor

# Watch logs
kubectl logs -f -l component=batch-worker,tier=thor
kubectl logs -f -l component=interactive,tier=thor
```

---

## Phase 6: Deploy API and Redis (Gateway Node)

### Step 6.1: Deploy Redis

```bash
# Create Redis deployment if not in your manifests
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      nodeSelector:
        node-role.kubernetes.io/master: "true"  # Schedule on gateway
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
---
apiVersion: v1
kind: Service
metadata:
  name: redis
spec:
  selector:
    app: redis
  ports:
  - port: 6379
EOF
```

### Step 6.2: Deploy API Pod

```bash
# Deploy the API service (from main deployment.yaml or create separately)
kubectl apply -f k8s/deployment.yaml

# Verify API is running
kubectl get pods -l component=api
kubectl logs -f -l component=api
```

### Step 6.3: Setup Ingress

```bash
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: angel-intelligence-ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - ai.yourdomain.com
    secretName: angel-tls-secret
  rules:
  - host: ai.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: angel-intelligence-api
            port:
              number: 8000
EOF
```

---

## Phase 7: Verification and Testing

### Step 7.1: Verify All Pods Running

```bash
kubectl get pods -o wide

# Expected output:
# NAME                                            READY   STATUS    NODE
# vllm-server-xxx                                 1/1     Running   thor-node-01
# angel-intelligence-transcription-xxx            1/1     Running   thor-node-01
# angel-intelligence-thor-batch-xxx               1/1     Running   thor-node-01
# angel-intelligence-thor-interactive-xxx         1/1     Running   thor-node-01
# redis-xxx                                       1/1     Running   gateway-node
# angel-intelligence-api-xxx                      1/1     Running   gateway-node
```

### Step 7.2: Test vLLM API

```bash
kubectl port-forward svc/vllm-server 8000:8000 &

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 50
  }'
```

### Step 7.3: Test Transcription Service

```bash
kubectl port-forward svc/transcription-service 8001:8001 &
curl http://localhost:8001/internal/health
```

### Step 7.4: Test End-to-End API

```bash
# Get your API URL (Ingress or NodePort)
API_URL="https://ai.yourdomain.com"
API_TOKEN="your-api-token"

# Health check
curl -H "Authorization: Bearer $API_TOKEN" $API_URL/health

# Test call analysis endpoint
curl -X POST -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"recording_id": 12345}' \
  $API_URL/api/analyze
```

### Step 7.5: Monitor GPU Usage

```bash
# On Thor node
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv -l 5

# Or via kubectl
kubectl exec -it $(kubectl get pod -l app=vllm-server -o name) -- nvidia-smi
```

---

## Phase 8: Scaling and Optimisation

### Scale Transcription Service

The transcription service is typically the bottleneck. Scale based on call volume:

```bash
# Scale to 3 replicas (~216 calls/hr)
kubectl scale deployment angel-intelligence-transcription --replicas=3

# Check replica status
kubectl get pods -l component=transcription
```

| Replicas | VRAM Used | Calls/Hour |
|----------|-----------|------------|
| 1 | ~10GB | ~72 |
| 2 | ~20GB | ~144 |
| 3 | ~30GB | ~216 |

### Scale Workers

```bash
# Scale batch workers
kubectl scale deployment angel-intelligence-thor-batch --replicas=2

# Scale interactive workers
kubectl scale deployment angel-intelligence-thor-interactive --replicas=2
```

### Enable vLLM Load Balancing

For higher vLLM throughput, deploy multiple instances:

```bash
kubectl apply -f k8s/vllm-service-lb.yaml
```

---

## Monitoring and Maintenance

### View Logs

```bash
# All Thor components
kubectl logs -l tier=thor -f --all-containers

# Specific component
kubectl logs -l component=batch-worker,tier=thor -f
kubectl logs -l app=vllm-server -f
kubectl logs -l component=transcription -f
```

### Database Metrics

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

### Rolling Updates

```bash
# Build new images
docker build -f Dockerfile.worker-jetson -t your-registry/angel-intelligence:worker-v2.0.0 .
docker build -f Dockerfile.transcription-jetson -t your-registry/angel-intelligence:transcription-v2.0.0 .
docker push your-registry/angel-intelligence:worker-v2.0.0
docker push your-registry/angel-intelligence:transcription-v2.0.0

# Update worker deployment
kubectl set image deployment/angel-intelligence-thor-batch \
  worker=your-registry/angel-intelligence:worker-v2.0.0

# Update transcription deployment
kubectl set image deployment/angel-intelligence-transcription \
  transcription=your-registry/angel-intelligence:transcription-v2.0.0

# Watch rollout
kubectl rollout status deployment/angel-intelligence-thor-batch
kubectl rollout status deployment/angel-intelligence-transcription

# Rollback if needed
kubectl rollout undo deployment/angel-intelligence-thor-batch
```

---

## Troubleshooting

### Pod Not Starting on Thor Node

```bash
# Check node selector matches
kubectl describe pod <pod-name>

# Verify GPU is available
kubectl describe node thor-node-01 | grep nvidia.com/gpu

# Check node taint/toleration
kubectl get nodes -o json | jq '.items[].spec.taints'
```

### vLLM Out of Memory

```bash
# Reduce GPU memory utilization in vllm-deployment.yaml
GPU_MEMORY_UTILIZATION: "0.75"  # Down from 0.85

# Or reduce max sequences
MAX_NUM_SEQS: "16"  # Down from 32
```

### Transcription Service Slow

```bash
# Scale up replicas
kubectl scale deployment angel-intelligence-transcription --replicas=3

# Check if using GPU
kubectl exec -it <transcription-pod> -- python -c "import torch; print(torch.cuda.is_available())"
```

### Model Loading Failures

```bash
# Check NFS mount
kubectl exec -it <pod> -- ls -la /models

# Check disk space
kubectl exec -it <pod> -- df -h /models

# Verify HuggingFace token
kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.huggingface-token}' | base64 -d
```

### Workers Not Processing Jobs

```bash
# Check Redis connection
kubectl exec -it <worker-pod> -- python -c "
from redis import Redis
r = Redis(host='redis', port=6379)
print('Redis connected:', r.ping())
"

# Check database connection
kubectl exec -it <worker-pod> -- python -c "
from src.database import get_db_connection
db = get_db_connection()
print('DB connected successfully')
"
```

---

## Quick Reference Commands

```bash
# View all Angel Intelligence pods
kubectl get pods -l app=angel-intelligence -o wide

# View Thor-specific pods
kubectl get pods -l tier=thor -o wide

# View pod resource usage
kubectl top pods -l tier=thor

# Restart all Thor workers
kubectl rollout restart deployment -l tier=thor

# View events
kubectl get events --sort-by='.lastTimestamp' | grep angel

# Access pod shell
kubectl exec -it <pod-name> -- /bin/bash

# Port-forward for debugging
kubectl port-forward svc/vllm-server 8000:8000
kubectl port-forward svc/transcription-service 8001:8001
```

---

## Next Steps

- [MONITORING.md](MONITORING.md) - Setup observability and alerting
- [TRAINING.md](TRAINING.md) - Configure LoRA fine-tuning cronjob
- [TESTING.md](TESTING.md) - Run verification tests
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

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
