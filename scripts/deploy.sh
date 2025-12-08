#!/bin/bash
# Build and deploy AI worker to K3s cluster

set -e

# Configuration
REGISTRY="${K3S_REGISTRY:-localhost:5000}"
IMAGE_NAME="ai-worker"
VERSION="${VERSION:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "🔨 Building Docker image..."
docker build -t ${IMAGE_NAME}:${VERSION} .

echo "🏷️  Tagging image for registry..."
docker tag ${IMAGE_NAME}:${VERSION} ${FULL_IMAGE}

echo "📤 Pushing to registry at ${REGISTRY}..."
docker push ${FULL_IMAGE}

echo "📝 Updating deployment image..."
kubectl set image deployment/ai-worker ai-worker=${FULL_IMAGE}

echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/ai-worker

echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -l app=ai-worker

echo ""
echo "💡 View logs with: kubectl logs -f deployment/ai-worker"
