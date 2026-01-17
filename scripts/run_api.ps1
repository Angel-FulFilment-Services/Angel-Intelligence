# Run Angel Intelligence API server locally (PowerShell)

Write-Host "🚀 Starting Angel Intelligence API..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "📦 Activating virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
    } else {
        Write-Host "⚠️  Virtual environment not found. Please run:" -ForegroundColor Red
        Write-Host "   python -m venv venv" -ForegroundColor Yellow
        Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}

# Set development mode if not set
if (-not $env:ANGEL_ENV) {
    $env:ANGEL_ENV = "development"
}

Write-Host "📍 Environment: $env:ANGEL_ENV" -ForegroundColor Cyan
Write-Host ""

# Run the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

