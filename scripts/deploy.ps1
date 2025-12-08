# Build and deploy AI worker to K3s cluster (PowerShell)

param(
    [string]$Registry = $env:K3S_REGISTRY ?? "localhost:5000",
    [string]$Version = "latest"
)

$ImageName = "ai-worker"
$FullImage = "${Registry}/${ImageName}:${Version}"

Write-Host "🔨 Building Docker image..." -ForegroundColor Cyan
docker build -t "${ImageName}:${Version}" .

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "🏷️  Tagging image for registry..." -ForegroundColor Cyan
docker tag "${ImageName}:${Version}" $FullImage

Write-Host "📤 Pushing to registry at ${Registry}..." -ForegroundColor Cyan
docker push $FullImage

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}

Write-Host "📝 Updating deployment image..." -ForegroundColor Cyan
kubectl set image deployment/ai-worker "ai-worker=$FullImage"

Write-Host "⏳ Waiting for rollout to complete..." -ForegroundColor Cyan
kubectl rollout status deployment/ai-worker

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Pod status:" -ForegroundColor Cyan
kubectl get pods -l app=ai-worker

Write-Host ""
Write-Host "💡 View logs with: kubectl logs -f deployment/ai-worker" -ForegroundColor Yellow
