# Build and deploy Angel Intelligence to K3s cluster (PowerShell)

param(
    [string]$Registry = $env:K3S_REGISTRY ?? "localhost:5000",
    [string]$Version = "latest",
    [string]$Component = "all"  # 'api', 'worker', or 'all'
)

$ImageName = "angel-intelligence"
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

if ($Component -eq "all" -or $Component -eq "api") {
    Write-Host "📝 Updating API deployment..." -ForegroundColor Cyan
    kubectl set image deployment/angel-intelligence-api "api=$FullImage"
    kubectl rollout status deployment/angel-intelligence-api
}

if ($Component -eq "all" -or $Component -eq "worker") {
    Write-Host "📝 Updating Worker deployment..." -ForegroundColor Cyan
    kubectl set image deployment/angel-intelligence-worker "worker=$FullImage"
    kubectl rollout status deployment/angel-intelligence-worker
}

Write-Host "✅ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Pod status:" -ForegroundColor Cyan
kubectl get pods -l app=angel-intelligence

Write-Host ""
Write-Host "💡 View logs with:" -ForegroundColor Yellow
Write-Host "   API:    kubectl logs -f deployment/angel-intelligence-api"
Write-Host "   Worker: kubectl logs -f deployment/angel-intelligence-worker"
