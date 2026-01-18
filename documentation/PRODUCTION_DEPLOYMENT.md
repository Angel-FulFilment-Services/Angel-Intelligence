# Production Deployment Guide

Deploy Angel Intelligence to a production Kubernetes cluster running on NVIDIA Jetson devices.

## Quick Start (Kubernetes)

```bash
# 1. Setup K3s cluster (if not already done)
# See detailed steps below

# 2. Build and push Docker image
docker build -f Dockerfile.jetson -t your-registry/angel-intelligence:latest .
docker push your-registry/angel-intelligence:latest

# 3. Configure secrets
cp k8s/secret.yaml.example k8s/secret.yaml
# Edit k8s/secret.yaml with your credentials (base64 encoded)

# 4. Apply Kubernetes manifests
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml

# 5. Verify deployment
kubectl get pods
kubectl logs -f deployment/angel-intelligence

# 6. Test
curl http://CLUSTER_IP/health
```

---

## Part 1: Kubernetes Installation

### Architecture

- **Kubernetes Master/Control Plane**: Separate server (x86/AMD64 recommended)
- **Worker Nodes**: NVIDIA Jetson devices (AGX Orin 32GB/64GB)
- **NFS Storage**: Can run on master server or separate NAS
- **Database**: External MySQL server

### Prerequisites

- 1 server for Kubernetes master (4+ CPU cores, 8GB+ RAM, Ubuntu 22.04)
- 3+ NVIDIA Jetson devices for workers (AGX Orin 32GB/64GB recommended)
- All devices on same network with static IPs
- Ubuntu 20.04/22.04 on all devices

### Step 1.1: Install K3s on Master Server

On your **separate Kubernetes server** (NOT Jetson):

```bash
# Install K3s master/control plane
curl -sfL https://get.k3s.io | sh -s - server \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --disable servicelb

# Get node token for workers
sudo cat /var/lib/rancher/k3s/server/node-token
# Save this token - you'll need it for Jetson worker nodes

# Verify master is running
sudo k3s kubectl get nodes

# Setup kubectl access
mkdir -p ~/.kube
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $USER ~/.kube/config
```

### Step 1.2: Add Jetson Devices as Worker Nodes

On **each Jetson device** (workers only):

```bash
# IMPORTANT: Replace the placeholders below with actual values:
# - MASTER_IP = Your Kubernetes server's IP address (e.g., 192.168.1.100)
# - NODE_TOKEN = Token from step 1.1 (starts with K10...)

# Fix for SSL certificate errors (if you see certificate verification failed):
# Option 1: Update CA certificates first (recommended)
sudo apt update
sudo apt install ca-certificates -y
sudo update-ca-certificates

# Option 2: If still failing, use -k flag to skip SSL verification
# curl -sfL -k https://get.k3s.io | K3S_URL=https://MASTER_IP:6443 \
#   K3S_TOKEN=NODE_TOKEN sh -

# Standard install command (replace MASTER_IP and NODE_TOKEN):
curl -sfL https://get.k3s.io | K3S_URL=https://MASTER_IP:6443 \
  K3S_TOKEN=NODE_TOKEN sh -

# Verify worker started
sudo systemctl status k3s-agent
```

**Common Issues:**

**SSL Certificate Error:**
```bash
# Update certificates
sudo apt install ca-certificates -y
sudo update-ca-certificates

# Or skip SSL check (less secure):
curl -sfL -k https://get.k3s.io | K3S_URL=https://MASTER_IP:6443 K3S_TOKEN=NODE_TOKEN sh -
```

**Connection refused:**
- Check master IP is correct: `ping MASTER_IP`
- Check port 6443 is open: `nc -zv MASTER_IP 6443`
- Check firewall on master: `sudo ufw status`

**Back on master server**, verify all workers joined:

```bash
kubectl get nodes
# Should show your master + all Jetson workers
```

### Step 1.3: Install NVIDIA Device Plugin

**On master server:**

```bash
# Install NVIDIA device plugin for GPU access in Jetson pods
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPUs are detected on Jetson nodes
kubectl describe nodes | grep nvidia.com/gpu
# Should show available GPUs on each Jetson node
```

### Step 1.4: Label Jetson Nodes

**On master server**, label Jetson nodes for scheduling:

```bash
# Label each Jetson node
kubectl label node jetson-1 node-type=jetson-gpu
kubectl label node jetson-2 node-type=jetson-gpu
kubectl label node jetson-3 node-type=jetson-gpu

# Verify labels
kubectl get nodes --show-labels | grep jetson-gpu
```

---

## Part 2: Deploy Angel Intelligence

### Step 2.1: Setup NFS Storage (Optional but Recommended)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pulse (Laravel)                         │
│                     Frontend Application                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTPS (Bearer Token Auth)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              K3s Cluster (LoadBalancer/Ingress)                 │
├─────────────────────────────────────────────────────────────────┤
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
│    │   Worker 1   │      │   Worker 2   │      │   Worker 3   │ │
│    │ Jetson Orin  │      │ Jetson Orin  │      │ Jetson Orin  │ │
│    │   8GB GPU    │      │   8GB GPU    │      │   8GB GPU    │ │
│    └──────────────┘      └──────────────┘      └──────────────┘ │
│           │                     │                     │         │
│           └─────────────────────┼─────────────────────┘         │
│                                 ▼                               │
│                    ┌─────────────────────────┐                  │
│                    │   Shared NFS Storage    │                  │
│                    │   /models (50GB)        │                  │
│                    └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │   MySQL Database       │
         │   Cloudflare R2        │
         └────────────────────────┘
```

## Prerequisites

1. **K3s Cluster**: Installed and running (see K3S_SETUP.md)
2. **kubectl**: Configured to connect to cluster
3. **Container Registry**: Local or DockerHub access
4. **NFS Server**: For shared model storage
5. **MySQL Server**: External database server
6. **Cloudflare R2**: For audio storage (optional)

---

## Part 2: Deploy Angel Intelligence

### Step 2.1: Setup NFS Storage (Recommended)

**On your Kubernetes master server** (or separate NAS):

```bash
# Install NFS server
sudo apt install nfs-kernel-server -y

# Create model storage
sudo mkdir -p /exports/angel-models
sudo chown -R nobody:nogroup /exports/angel-models
sudo chmod 777 /exports/angel-models

# Configure NFS exports (allow access from all cluster nodes)
echo "/exports/angel-models *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

# Verify NFS is running
showmount -e localhost
```

**On all Jetson worker nodes:**

```bash
# Install NFS client
sudo apt install nfs-common -y

# Test mount (replace MASTER_IP with your Kubernetes server IP)
sudo mkdir -p /mnt/test
sudo mount MASTER_IP:/exports/angel-models /mnt/test
ls /mnt/test
sudo umount /mnt/test
```

### Step 2.2: Build Docker Image

**On your Kubernetes master server:**

```bash
# Clone repository
cd ~
git clone https://github.com/Angel-FulFilment-Services/Angel-Intelligence.git
cd Angel-Intelligence

# Install docker to Kubernetes server
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh


# Build Jetson image (ARM64)
# Note: On x86 server, use buildx for cross-platform build
docker buildx create --use
docker buildx build --platform linux/arm64 \
  -f Dockerfile.jetson \
  -t angel-intelligence:jetson-latest \
  --load .

# OR build directly on one Jetson then export
# On Jetson:
# docker build -f Dockerfile.jetson -t angel-intelligence:jetson-latest .
# docker save angel-intelligence:jetson-latest | gzip > angel-jetson.tar.gz
# Copy to master server and load:
# docker load < angel-jetson.tar.gz
```

**OR use local registry on cluster:**

```bash
# On master node, create local registry
kubectl create -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docker-registry
spec:
  replicas: 1
  selector:
    matchLabels:
      app: docker-registry
  template:
    metadata:
      labels:
        app: docker-registry
    spec:
      containers:
      - name: registry
        image: registry:2
        ports:
        - containerPort: 5000
---
apiVersion: v1
kind: Service
metadata:
  name: docker-registry
spec:
  selector:
    app: docker-registry
  ports:
  - port: 5000
    targetPort: 5000
  type: NodePort
EOF

# Push to local registry
docker tag angel-intelligence:jetson-latest localhost:5000/angel-intelligence:jetson-latest
docker push localhost:5000/angel-intelligence:jetson-latest
```

### Step 2.3: Create Kubernetes Secrets

**On your Kubernetes master server:**

Create the secrets file from template:

```bash
cp k8s/secret.yaml.example k8s/secret.yaml
```

Edit `k8s/secret.yaml` with base64-encoded values:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: angel-intelligence-secrets
  namespace: default
type: Opaque
data:
  # Generate with: echo -n "your-value" | base64
  api-auth-token: <base64-encoded-64-char-token>
  db-host: <base64-encoded-host>
  db-database: <base64-encoded-database>
  db-username: <base64-encoded-username>
  db-password: <base64-encoded-password>
  r2-access-key: <base64-encoded-key>
  r2-secret-key: <base64-encoded-secret>
  huggingface-token: <base64-encoded-hf-token>
```

Apply secrets:

```bash
kubectl apply -f k8s/secret.yaml
```

## Step 2: Configure NFS for Shared Models

### On NFS Server

```bash
# Create model directories
sudo mkdir -p /exports/angel-models/{whisper,analysis,chat}
sudo chown -R nobody:nogroup /exports/angel-models

# Configure NFS exports
echo "/exports/angel-models *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -ra
```

### Create PersistentVolume

```yaml
# k8s/pv-models.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: angel-models-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteMany
  nfs:
    server: 192.168.1.100  # Your NFS server IP
    path: /exports/angel-models
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: angel-models-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  volumeName: angel-models-pv
```

Apply:

```bash
kubectl apply -f k8s/pv-models.yaml
```

## Step 3: Build and Push Container Image

### Using Local Registry

```bash
# Build for ARM64 (Jetson)
docker build --platform linux/arm64 -t localhost:5000/angel-intelligence:latest .

# Push to local registry
docker push localhost:5000/angel-intelligence:latest
```

### Using Docker Hub

```bash
# Login
docker login

# Build and push
docker build -t your-org/angel-intelligence:v1.0.0 .
docker push your-org/angel-intelligence:v1.0.0
```

## Step 4: Update ConfigMap

Edit `k8s/configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: angel-intelligence-config
data:
  ANGEL_ENV: "production"
  ANALYSIS_MODE: "audio"
  WHISPER_MODEL: "medium"
  USE_GPU: "true"
  USE_MOCK_MODELS: "false"
  POLL_INTERVAL_SECONDS: "30"
  MAX_CONCURRENT_JOBS: "4"
  MAX_RETRIES: "3"
  RETRY_DELAY_HOURS: "1"
  ENABLE_MODEL_HOT_RELOAD: "true"
  MODEL_RELOAD_CHECK_INTERVAL: "60"
  
  # PBX Recording Sources
  PBX_LIVE_URL: "https://pbx.angelfs.co.uk/callrec/"
  PBX_ARCHIVE_URL: "https://afs-pbx-callarchive.angelfs.co.uk/"
  
  # R2 Storage
  R2_ENDPOINT: "https://your-account.r2.cloudflarestorage.com"
  R2_BUCKET: "angel-call-recordings"
  
  # Model Paths (on NFS mount)
  MODELS_BASE_PATH: "/models"
  WHISPER_MODEL_PATH: "/models/whisper"
  ANALYSIS_MODEL_PATH: "/models/analysis"
  CHAT_MODEL_PATH: "/models/chat"
```

Apply:

```bash
kubectl apply -f k8s/configmap.yaml
```

## Step 5: Deploy Workers

Angel Intelligence supports two worker modes for optimal resource allocation:

### Worker Modes

| Mode | Purpose | Recommended Nodes |
|------|---------|-------------------|
| **batch** | Call processing (transcription, analysis) | 3+ Jetsons |
| **interactive** | Real-time AI (chat, summaries) | 1 Jetson |

### Deployment Strategy

For a 4-Jetson cluster, deploy:
- **3 batch workers** - Handle call queue (~900 calls/day capacity)
- **1 interactive node** - Handle chat and summary requests

Edit `k8s/deployment.yaml` replica counts:

```yaml
# Batch workers - handles call processing
apiVersion: apps/v1
kind: Deployment
metadata:
  name: angel-intelligence-batch-worker
spec:
  replicas: 3  # Adjust based on call volume
  template:
    spec:
      containers:
      - name: worker
        env:
        - name: WORKER_MODE
          value: "batch"
        resources:
          limits:
            nvidia.com/gpu: 1

---
# Interactive worker - handles real-time requests
apiVersion: apps/v1
kind: Deployment
metadata:
  name: angel-intelligence-interactive
spec:
  replicas: 1  # Start with 1, scale if needed
  template:
    spec:
      containers:
      - name: interactive
        command: ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
        env:
        - name: WORKER_MODE
          value: "interactive"
        resources:
          limits:
            nvidia.com/gpu: 1
```

Deploy:

```bash
kubectl apply -f k8s/deployment.yaml
```

### Scaling Guidelines

| Scenario | Batch Workers | Interactive |
|----------|---------------|-------------|
| Low volume (<300 calls/day) | 1 | 1 |
| Medium volume (300-900 calls/day) | 3 | 1 |
| High volume (900-1500 calls/day) | 5 | 1 |
| Heavy interactive usage | N | 2+ |

Scale deployments:

```bash
# Scale batch workers
kubectl scale deployment angel-intelligence-batch-worker --replicas=5

# Scale interactive workers
kubectl scale deployment angel-intelligence-interactive --replicas=2
```

## Step 6: Verify Deployment

```bash
# Check pods
kubectl get pods -l app=angel-intelligence

# View logs
kubectl logs -l app=angel-intelligence -f

# Check worker health
kubectl exec -it <pod-name> -- curl http://localhost:8000/health
```

## Step 7: Configure Ingress (Optional)

For external API access:

```yaml
# k8s/ingress.yaml
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
    - ai.angelfs.co.uk
    secretName: angel-tls-secret
  rules:
  - host: ai.angelfs.co.uk
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: angel-intelligence-api
            port:
              number: 8000
```

## Scaling

### Add More Workers

```bash
# Scale to 6 workers (if you have 6 nodes)
kubectl scale deployment angel-intelligence-worker --replicas=6
```

### Check Node Utilisation

```bash
kubectl top nodes
kubectl top pods -l app=angel-intelligence
```

## Updating

### Rolling Update

```bash
# Build new image
docker build -t localhost:5000/angel-intelligence:v1.1.0 .
docker push localhost:5000/angel-intelligence:v1.1.0

# Update deployment
kubectl set image deployment/angel-intelligence-worker \
  worker=localhost:5000/angel-intelligence:v1.1.0

# Watch rollout
kubectl rollout status deployment/angel-intelligence-worker
```

### Rollback

```bash
# View rollout history
kubectl rollout history deployment/angel-intelligence-worker

# Rollback to previous version
kubectl rollout undo deployment/angel-intelligence-worker
```

## Monitoring

### Log Aggregation

```bash
# View all worker logs
kubectl logs -l app=angel-intelligence,component=worker --all-containers -f

# Export logs
kubectl logs -l app=angel-intelligence > angel-logs.txt
```

### Health Checks

```bash
# Check all pods health
for pod in $(kubectl get pods -l app=angel-intelligence -o name); do
  echo "=== $pod ==="
  kubectl exec $pod -- curl -s http://localhost:8000/health | jq
done
```

### Database Metrics

```sql
-- Processing queue status
SELECT processing_status, COUNT(*) 
FROM ai_call_recordings 
GROUP BY processing_status;

-- Average processing time
SELECT DATE(processing_completed_at) as date,
       AVG(TIMESTAMPDIFF(SECOND, processing_started_at, processing_completed_at)) as avg_seconds
FROM ai_call_recordings
WHERE processing_status = 'completed'
GROUP BY DATE(processing_completed_at)
ORDER BY date DESC LIMIT 7;
```

## Troubleshooting

### Pod Not Starting

```bash
# Check events
kubectl describe pod <pod-name>

# Check node resources
kubectl describe node <node-name>
```

### GPU Not Available

```bash
# Check NVIDIA runtime
kubectl get nodes -o json | jq '.items[].status.allocatable["nvidia.com/gpu"]'

# Verify GPU access in pod
kubectl exec -it <pod-name> -- nvidia-smi
```

### Model Loading Issues

```bash
# Check NFS mount
kubectl exec -it <pod-name> -- ls -la /models

# Check model files
kubectl exec -it <pod-name> -- du -sh /models/*
```

### Database Connection Issues

```bash
# Test from pod
kubectl exec -it <pod-name> -- python -c "
from src.database import get_db_connection
db = get_db_connection()
print('Connected successfully')
"
```

## Security Checklist

- [ ] API_AUTH_TOKEN is 64+ characters
- [ ] Secrets are not in version control
- [ ] TLS enabled on Ingress
- [ ] Network policies restrict pod-to-pod traffic
- [ ] Database credentials use least privilege
- [ ] R2 bucket is not publicly accessible
- [ ] Worker pods cannot access Kubernetes API

## Performance Tuning

### Worker Settings

```env
# Adjust based on hardware
POLL_INTERVAL_SECONDS=30  # Lower for faster pickup
MAX_CONCURRENT_JOBS=4     # Match GPU memory
WHISPER_MODEL=medium      # small for less VRAM
```

### Database Optimisation

```sql
-- Add indexes for common queries
CREATE INDEX idx_processing_status ON ai_call_recordings(processing_status);
CREATE INDEX idx_call_date ON ai_call_recordings(call_date);
CREATE INDEX idx_client_ref ON ai_call_recordings(client_ref);
```

## Backup and Recovery

### Database Backup

```bash
# Daily backup script
mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASS ai > backup_$(date +%Y%m%d).sql
```

### Model Backup

```bash
# Backup trained models
rsync -av /exports/angel-models/ /backup/angel-models/
```

## Next Steps

- [Monitoring](MONITORING.md) - Set up observability
- [Testing Guide](TESTING.md) - Verify deployment
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues
