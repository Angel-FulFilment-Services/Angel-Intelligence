# Production Deployment Guide

Deploy Angel Intelligence to a production Kubernetes cluster running on NVIDIA Jetson devices.

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

## Step 1: Prepare Secrets

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

Edit `k8s/deployment.yaml` to set replica count and resource limits:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: angel-intelligence-worker
spec:
  replicas: 4  # One per Jetson node
  selector:
    matchLabels:
      app: angel-intelligence
      component: worker
  template:
    metadata:
      labels:
        app: angel-intelligence
        component: worker
    spec:
      containers:
      - name: worker
        image: localhost:5000/angel-intelligence:latest
        command: ["python", "-m", "src.worker.worker"]
        envFrom:
        - configMapRef:
            name: angel-intelligence-config
        - secretRef:
            name: angel-intelligence-secrets
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
        - name: temp
          mountPath: /tmp/angel
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: angel-models-pvc
      - name: temp
        emptyDir: {}
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app: angel-intelligence
                component: worker
            topologyKey: kubernetes.io/hostname
```

Deploy:

```bash
kubectl apply -f k8s/deployment.yaml
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
