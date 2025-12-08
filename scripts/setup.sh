#!/bin/bash
# Initial setup script for K3s cluster

set -e

echo "🚀 K3s AI Worker Initial Setup"
echo "================================"
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl first."
    exit 1
fi

# Check if we can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster."
    echo "   Make sure you have K3s running and kubeconfig configured."
    exit 1
fi

echo "✅ Connected to cluster"
echo ""

# Create namespace if it doesn't exist
echo "📦 Creating namespace (if needed)..."
kubectl create namespace default --dry-run=client -o yaml | kubectl apply -f -

# Create secrets if they don't exist
if ! kubectl get secret ai-worker-secrets &> /dev/null; then
    echo "🔐 Creating secrets..."
    if [ ! -f "k8s/secret.yaml" ]; then
        echo "⚠️  k8s/secret.yaml not found!"
        echo "   Please copy k8s/secret.yaml.example to k8s/secret.yaml"
        echo "   and fill in your credentials."
        exit 1
    fi
    kubectl apply -f k8s/secret.yaml
    echo "✅ Secrets created"
else
    echo "✅ Secrets already exist"
fi

# Apply ConfigMap
echo "⚙️  Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# Apply Deployment
echo "🚀 Applying Deployment..."
kubectl apply -f k8s/deployment.yaml

echo ""
echo "⏳ Waiting for pods to be ready..."
kubectl wait --for=condition=ready pod -l app=ai-worker --timeout=300s || true

echo ""
echo "✅ Setup complete!"
echo ""
echo "📊 Current status:"
kubectl get pods -l app=ai-worker
echo ""
echo "💡 Useful commands:"
echo "   View logs:          kubectl logs -f deployment/ai-worker"
echo "   Scale workers:      kubectl scale deployment ai-worker --replicas=N"
echo "   Update workers:     ./scripts/deploy.sh"
echo "   Delete everything:  kubectl delete -f k8s/"
