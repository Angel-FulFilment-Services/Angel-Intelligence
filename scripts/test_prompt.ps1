# Test LLM Prompt Generation (PowerShell)
# 
# Tests the full LLM prompt generation for a call recording
# WITHOUT marking it as processing or completed.
#
# Usage:
#   .\scripts\test_prompt.ps1                    # Test next pending recording
#   .\scripts\test_prompt.ps1 -Id 123            # Test specific recording by ID
#   .\scripts\test_prompt.ps1 -Apex ABC123       # Test specific recording by apex_id
#   .\scripts\test_prompt.ps1 -Id 123 -Save      # Save prompt to file

param(
    [int]$Id,
    [string]$Apex,
    [switch]$Save,
    [switch]$Quiet
)

Write-Host "Angel Intelligence - Prompt Tester" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    if (Test-Path "venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & ".\venv\Scripts\Activate.ps1"
    } elseif (Test-Path ".venv\Scripts\Activate.ps1") {
        Write-Host "Activating virtual environment..." -ForegroundColor Yellow
        & ".\.venv\Scripts\Activate.ps1"
    } else {
        Write-Host "Virtual environment not found. Please run:" -ForegroundColor Red
        Write-Host "   python -m venv venv" -ForegroundColor Yellow
        Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
        Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}

# Build arguments
$pythonArgs = @()

if ($Id) {
    $pythonArgs += "--id"
    $pythonArgs += $Id
}

if ($Apex) {
    $pythonArgs += "--apex"
    $pythonArgs += $Apex
}

if ($Save) {
    $pythonArgs += "--save"
}

if ($Quiet) {
    $pythonArgs += "--quiet"
}

# Run the script
python scripts/test_prompt.py @pythonArgs
