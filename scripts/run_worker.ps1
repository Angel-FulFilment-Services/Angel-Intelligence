# Run Angel Intelligence Worker locally (PowerShell)

Write-Host "Starting Angel Intelligence Worker..." -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
    } else {
        Write-Host "Virtual environment not found. Please run:" -ForegroundColor Red
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

if (-not $env:WORKER_ID) {
    $env:WORKER_ID = "local-dev"
}

# Set GPU selection - MUST be set before PyTorch loads
# On dual-GPU systems (Intel iGPU + NVIDIA dGPU), the iGPU may be device 0
# Set this to the NVIDIA GPU index (check with: nvidia-smi -L)
if (-not $env:CUDA_VISIBLE_DEVICES) {
    # Check if .env has a value, otherwise default to "1" for NVIDIA on dual-GPU systems
    if (Test-Path ".env") {
        $envFile = Get-Content ".env"
        $cudaLine = $envFile | Where-Object { $_ -match "^CUDA_VISIBLE_DEVICES=" }
        if ($cudaLine) {
            $env:CUDA_VISIBLE_DEVICES = ($cudaLine -split "=")[1].Trim()
        }
    }
}

Write-Host "Environment: $env:ANGEL_ENV" -ForegroundColor Cyan
Write-Host "Worker ID: $env:WORKER_ID" -ForegroundColor Cyan
if ($env:CUDA_VISIBLE_DEVICES) {
    Write-Host "CUDA GPU(s): $env:CUDA_VISIBLE_DEVICES" -ForegroundColor Cyan
}
Write-Host ""

# Run the worker
python -m src.worker.worker
