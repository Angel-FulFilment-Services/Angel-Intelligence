# Initial setup script for K3s cluster (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "🚀 K3s AI Worker Initial Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if kubectl is available
try {
    kubectl version --client | Out-Null
    Write-Host "✅ kubectl found" -ForegroundColor Green
} catch {
    Write-Host "❌ kubectl not found. Please install kubectl first." -ForegroundColor Red
    exit 1
}

# Check if we can connect to cluster
try {
    kubectl cluster-info | Out-Null
    Write-Host "✅ Connected to cluster" -ForegroundColor Green
} catch {
    Write-Host "❌ Cannot connect to Kubernetes cluster." -ForegroundColor Red
    Write-Host "   Make sure you have K3s running and kubeconfig configured." -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Create secrets if they don't exist
Write-Host "🔐 Checking secrets..." -ForegroundColor Cyan
try {
    kubectl get secret ai-worker-secrets | Out-Null
    Write-Host "✅ Secrets already exist" -ForegroundColor Green
} catch {
    if (!(Test-Path "k8s\secret.yaml")) {
        Write-Host "⚠️  k8s\secret.yaml not found!" -ForegroundColor Yellow
        Write-Host "   Please copy k8s\secret.yaml.example to k8s\secret.yaml" -ForegroundColor Yellow
        Write-Host "   and fill in your credentials." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Creating secrets..." -ForegroundColor Cyan
    kubectl apply -f k8s\secret.yaml
    Write-Host "✅ Secrets created" -ForegroundColor Green
}

# Apply ConfigMap
Write-Host "⚙️  Applying ConfigMap..." -ForegroundColor Cyan
kubectl apply -f k8s\configmap.yaml

# Apply Deployment
Write-Host "🚀 Applying Deployment..." -ForegroundColor Cyan
kubectl apply -f k8s\deployment.yaml

Write-Host ""
Write-Host "⏳ Waiting for pods to be ready..." -ForegroundColor Cyan
kubectl wait --for=condition=ready pod -l app=ai-worker --timeout=300s

Write-Host ""
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Current status:" -ForegroundColor Cyan
kubectl get pods -l app=ai-worker

Write-Host ""
Write-Host "💡 Useful commands:" -ForegroundColor Yellow
Write-Host "   View logs:          kubectl logs -f deployment/ai-worker"
Write-Host "   Scale workers:      kubectl scale deployment ai-worker --replicas=N"
Write-Host "   Update workers:     .\scripts\deploy.ps1"
Write-Host "   Delete everything:  kubectl delete -f k8s\"
