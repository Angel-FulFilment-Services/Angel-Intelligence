# K3s Cluster Setup Guide for AI Workers

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              K3s Control Plane (HyperV VM)              │
│         - Manages cluster state                         │
│         - Schedules workloads                           │
│         - API server                                    │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┬───────────┬───────────┐
         │                       │           │           │
    ┌────▼─────┐          ┌─────▼────┐ ┌────▼─────┐ ┌──▼──────┐
    │ Jetson 1 │          │ Jetson 2 │ │ Jetson 3 │ │ Jetson 4│
    │ (Worker) │          │ (Worker) │ │ (Worker) │ │ (Worker)│
    │  + GPU   │          │  + GPU   │ │  + GPU   │ │  + GPU  │
    └──────────┘          └──────────┘ └──────────┘ └─────────┘
```

## Part 1: Setup Control Plane (HyperV VM)

### 1.1 Create Ubuntu VM in HyperV

1. **Download Ubuntu Server 22.04 LTS** (ARM64 if your host supports it, or x86_64)
   - https://ubuntu.com/download/server

2. **Create VM in HyperV Manager:**
   - Name: `k3s-control-plane`
   - Generation: 2
   - Memory: 2048 MB (2GB minimum, 4GB recommended)
   - CPU: 2 vCPUs
   - Disk: 20GB
   - Network: External Switch (so Jetsons can reach it)

3. **Install Ubuntu:**
   - Boot from ISO
   - Complete installation
   - Enable SSH server during installation

4. **Get VM's IP address:**
   ```bash
   ip addr show
   # Note the IP address (e.g., 192.168.1.100)
   ```

### 1.2 Install K3s Control Plane

SSH into your Ubuntu VM:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install K3s as server (control plane)
curl -sfL https://get.k3s.io | sh -s - server \
  --write-kubeconfig-mode 644 \
  --disable traefik \
  --node-taint CriticalAddonsOnly=true:NoExecute

# Verify installation
sudo kubectl get nodes

# Get node token (needed for workers to join)
sudo cat /var/lib/rancher/k3s/server/node-token
# Save this token! You'll need it for each Jetson
```

### 1.3 Setup Local Registry (for Docker images)

On the control plane VM:

```bash
# Install Docker
sudo apt install -y docker.io

# Run local registry
sudo docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name registry \
  -v /mnt/registry:/var/lib/registry \
  registry:2

# Configure K3s to use local registry
sudo mkdir -p /etc/rancher/k3s/
sudo tee /etc/rancher/k3s/registries.yaml > /dev/null <<EOF
mirrors:
  "localhost:5000":
    endpoint:
      - "http://$(hostname -I | awk '{print $1}'):5000"
EOF

# Restart K3s
sudo systemctl restart k3s
```

## Part 2: Setup Jetson Worker Nodes

### 2.1 Install K3s Agent on Each Jetson

On **each Jetson** (repeat for all 4):

```bash
# Install NVIDIA Container Toolkit (for GPU access)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure containerd for NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=containerd --set-as-default
sudo systemctl restart containerd

# Install K3s agent (worker)
# Replace YOUR_CONTROL_PLANE_IP and YOUR_TOKEN with values from Part 1
export K3S_URL="https://YOUR_CONTROL_PLANE_IP:6443"
export K3S_TOKEN="YOUR_TOKEN_FROM_CONTROL_PLANE"

curl -sfL https://get.k3s.io | sh -s - agent \
  --node-label nvidia.com/gpu=true

# Verify it joined the cluster
# (Run on control plane)
sudo kubectl get nodes
```

### 2.2 Install NVIDIA Device Plugin

On the **control plane** VM:

```bash
# Install NVIDIA device plugin for GPU scheduling
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Verify GPU is detected
kubectl get nodes -o json | grep -i nvidia
```

## Part 3: Deploy AI Workers

### 3.1 Build and Push Docker Image

On your **development machine** (Windows):

```powershell
# Build the image
cd C:\Users\lukes\Desktop\Voice\Voice
docker build -t ai-worker:latest .

# Tag for local registry (replace with your control plane IP)
docker tag ai-worker:latest YOUR_CONTROL_PLANE_IP:5000/ai-worker:latest

# Push to registry
docker push YOUR_CONTROL_PLANE_IP:5000/ai-worker:latest
```

**OR** build directly on the control plane:

```bash
# On control plane VM
git clone YOUR_REPO
cd Voice
sudo docker build -t localhost:5000/ai-worker:latest .
sudo docker push localhost:5000/ai-worker:latest
```

### 3.2 Create Kubernetes Secrets

On the **control plane** VM:

```bash
# Copy the example and edit with your credentials
cp k8s/secret.yaml.example k8s/secret.yaml
nano k8s/secret.yaml

# Apply secrets
kubectl apply -f k8s/secret.yaml
```

### 3.3 Deploy Workers

```bash
# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# Deploy workers
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -w

# View logs from a worker
kubectl logs -f <pod-name>

# Check GPU allocation
kubectl describe node <jetson-node-name> | grep -A 10 "Allocated resources"
```

## Part 4: Scaling & Management

### Add a New Jetson to Cluster

When you get a 5th, 6th, etc. Jetson:

```bash
# On the new Jetson, run the same K3s agent install from Part 2.1
export K3S_URL="https://YOUR_CONTROL_PLANE_IP:6443"
export K3S_TOKEN="YOUR_TOKEN"
curl -sfL https://get.k3s.io | sh -s - agent \
  --node-label nvidia.com/gpu=true

# On control plane, scale up the deployment
kubectl scale deployment ai-worker --replicas=5

# Kubernetes automatically schedules the new pod on the new Jetson
```

That's it! The new node is automatically utilized.

### Scale Down

```bash
kubectl scale deployment ai-worker --replicas=3
```

### Update Worker Code

```bash
# Build new image
docker build -t YOUR_CONTROL_PLANE_IP:5000/ai-worker:v2 .
docker push YOUR_CONTROL_PLANE_IP:5000/ai-worker:v2

# Update deployment
kubectl set image deployment/ai-worker ai-worker=YOUR_CONTROL_PLANE_IP:5000/ai-worker:v2

# Or edit deployment.yaml and apply
kubectl apply -f k8s/deployment.yaml
```

### Monitor Cluster

```bash
# View all resources
kubectl get all

# View worker logs
kubectl logs -f deployment/ai-worker

# View GPU usage on nodes
kubectl describe nodes | grep -i nvidia -A 5

# Get into a running pod
kubectl exec -it <pod-name> -- /bin/bash

# View events
kubectl get events --sort-by='.lastTimestamp'
```

## Part 5: Persistence & High Availability

### Model Cache Persistence

Models are cached on each Jetson at `/mnt/model-cache`. This prevents re-downloading models on every pod restart.

To pre-download models on all Jetsons:

```bash
# On each Jetson
sudo mkdir -p /mnt/model-cache
sudo chmod 777 /mnt/model-cache

# Run a test pod to download models
kubectl run test-download --image=YOUR_CONTROL_PLANE_IP:5000/ai-worker:latest \
  --restart=Never -- python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

### Control Plane Backup

```bash
# Backup K3s state
sudo k3s etcd-snapshot save

# Backups stored in: /var/lib/rancher/k3s/server/db/snapshots/
```

## Troubleshooting

### Pods not getting GPU

```bash
# Check if NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Check node labels
kubectl get nodes --show-labels | grep nvidia

# Manually label a node if needed
kubectl label nodes <node-name> nvidia.com/gpu=true
```

### Can't reach control plane from Jetsons

```bash
# On control plane, check firewall
sudo ufw status
sudo ufw allow 6443/tcp  # K3s API
sudo ufw allow 5000/tcp  # Docker registry

# Test connectivity from Jetson
curl -k https://YOUR_CONTROL_PLANE_IP:6443
```

### Pods stuck in Pending

```bash
# Check why
kubectl describe pod <pod-name>

# Common issues:
# - Not enough GPU nodes
# - Image pull failures
# - Resource constraints
```

## Quick Reference

```bash
# View cluster status
kubectl get nodes
kubectl get pods
kubectl get deployments

# Scale workers
kubectl scale deployment ai-worker --replicas=N

# Update image
kubectl set image deployment/ai-worker ai-worker=NEW_IMAGE

# View logs
kubectl logs -f deployment/ai-worker
kubectl logs -f <specific-pod-name>

# Delete everything
kubectl delete -f k8s/

# Restart deployment (recreate all pods)
kubectl rollout restart deployment/ai-worker

# Get control plane token (for adding nodes)
sudo cat /var/lib/rancher/k3s/server/node-token
```

## Network Requirements

- **Control Plane → Jetsons:** Ports 6443 (K3s API), 10250 (kubelet)
- **Jetsons → Control Plane:** Port 6443 (K3s API)
- **Development Machine → Control Plane:** Port 5000 (Docker Registry)
- All nodes should be on the same network or have proper routing configured

## Security Notes

1. The registry is HTTP (not HTTPS) for simplicity - fine for internal network
2. K3s token is sensitive - treat like a password
3. Kubernetes secrets are base64 encoded, not encrypted - consider using sealed-secrets for production
4. Consider setting up RBAC if multiple users will access the cluster
