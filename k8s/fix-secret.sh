#!/bin/bash
# Fix double-encoded secrets by recreating with plain values

echo "Fetching current secret values and decoding them..."

# Get current values (these are base64 encoded once by k8s)
API_TOKEN=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.api-auth-token}' | base64 -d)
DB_HOST=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-host}' | base64 -d)
DB_PORT=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-port}' | base64 -d)
DB_DATABASE=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-database}' | base64 -d)
DB_USERNAME=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-username}' | base64 -d)
DB_PASSWORD=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-password}' | base64 -d)
R2_ENDPOINT=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.r2-endpoint}' | base64 -d)
R2_ACCESS_KEY=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.r2-access-key-id}' | base64 -d)
R2_SECRET_KEY=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.r2-secret-access-key}' | base64 -d)
R2_BUCKET=$(kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.r2-bucket}' | base64 -d)

echo "Current db-port value: $DB_PORT"
echo ""

# Check if values are still base64 encoded (double-encoded)
if [[ "$DB_PORT" == *"="* ]] || [[ ! "$DB_PORT" =~ ^[0-9]+$ ]]; then
    echo "⚠️  Values are DOUBLE-ENCODED. Decoding again..."
    DB_PORT=$(echo "$DB_PORT" | base64 -d)
    echo "Decoded db-port: $DB_PORT"
    echo ""
    echo "Decoding all other values..."
    API_TOKEN=$(echo "$API_TOKEN" | base64 -d)
    DB_HOST=$(echo "$DB_HOST" | base64 -d)
    DB_DATABASE=$(echo "$DB_DATABASE" | base64 -d)
    DB_USERNAME=$(echo "$DB_USERNAME" | base64 -d)
    DB_PASSWORD=$(echo "$DB_PASSWORD" | base64 -d)
    R2_ENDPOINT=$(echo "$R2_ENDPOINT" | base64 -d)
    R2_ACCESS_KEY=$(echo "$R2_ACCESS_KEY" | base64 -d)
    R2_SECRET_KEY=$(echo "$R2_SECRET_KEY" | base64 -d)
    R2_BUCKET=$(echo "$R2_BUCKET" | base64 -d)
fi

echo "✓ Values decoded successfully"
echo ""
echo "Deleting old secret..."
kubectl delete secret angel-intelligence-secrets

echo ""
echo "Creating new secret with correct values..."
kubectl create secret generic angel-intelligence-secrets \
  --from-literal=api-auth-token="$API_TOKEN" \
  --from-literal=db-host="$DB_HOST" \
  --from-literal=db-port="$DB_PORT" \
  --from-literal=db-database="$DB_DATABASE" \
  --from-literal=db-username="$DB_USERNAME" \
  --from-literal=db-password="$DB_PASSWORD" \
  --from-literal=r2-endpoint="$R2_ENDPOINT" \
  --from-literal=r2-access-key-id="$R2_ACCESS_KEY" \
  --from-literal=r2-secret-access-key="$R2_SECRET_KEY" \
  --from-literal=r2-bucket="$R2_BUCKET"

echo ""
echo "✓ Secret recreated successfully!"
echo ""
echo "Verifying - db-port should be '3306':"
kubectl get secret angel-intelligence-secrets -o jsonpath='{.data.db-port}' | base64 -d
echo ""
echo ""
echo "Restart deployments to pick up the fixed secret:"
echo "  kubectl rollout restart deployment angel-intelligence-api"
echo "  kubectl rollout restart deployment angel-intelligence-interactive"
