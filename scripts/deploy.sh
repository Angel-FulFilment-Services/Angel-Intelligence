#!/bin/bash
# Build and deploy Angel Intelligence to K3s cluster

set -e

# Configuration
REGISTRY="${K3S_REGISTRY:-localhost:5000}"
IMAGE_NAME="angel-intelligence"
VERSION="${VERSION:-latest}"
COMPONENT="${COMPONENT:-all}"  # 'api', 'worker', or 'all'
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${VERSION}"

echo "🔨 Building Docker image..."
docker build -t ${IMAGE_NAME}:${VERSION} .

echo "🏷️  Tagging image for registry..."
docker tag ${IMAGE_NAME}:${VERSION} ${FULL_IMAGE}

echo "📤 Pushing to registry at ${REGISTRY}..."
docker push ${FULL_IMAGE}

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "api" ]; then
    echo "📝 Updating API deployment..."
    kubectl set image deployment/angel-intelligence-api api=${FULL_IMAGE}
    kubectl rollout status deployment/angel-intelligence-api
fi

if [ "$COMPONENT" = "all" ] || [ "$COMPONENT" = "worker" ]; then
    echo "📝 Updating Worker deployment..."
    kubectl set image deployment/angel-intelligence-worker worker=${FULL_IMAGE}
    kubectl rollout status deployment/angel-intelligence-worker
fi

echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -l app=angel-intelligence

echo ""
echo "💡 View logs with:"
echo "   API:    kubectl logs -f deployment/angel-intelligence-api"
echo "   Worker: kubectl logs -f deployment/angel-intelligence-worker"
