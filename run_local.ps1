<#
.SYNOPSIS
    run_local.ps1 - CaseLens Local Development Orchestrator

.DESCRIPTION
    Launches the decoupled CaseLens application:
    1. Starts the FastAPI backend service (src/api.py) on port 8000.
    2. Waits for the backend health check endpoint (http://localhost:8000/api/v1/health) to be ready.
    3. Launches the Streamlit frontend dashboard (app/dashboard.py) on port 8501.
    4. Automatically terminates the background FastAPI server when Streamlit closes or upon Ctrl+C.
#>

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   CaseLens: The Leaky Pipeline - Local Service Launcher   " -ForegroundColor Cyan
Write-Host "   Streamlit Frontend -> FastAPI Backend -> DuckDB/JSON    " -ForegroundColor DarkCyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we are in the project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check Python environment
$PythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonExe) {
    Write-Error "Python executable not found in PATH. Please install Python 3.10+."
    exit 1
}

$ApiPort = 8000
$HealthUrl = "http://localhost:$ApiPort/api/v1/health"
$ApiProcess = $null

try {
    Write-Host "[1/3] Starting FastAPI Backend on port $ApiPort..." -ForegroundColor Yellow
    $ApiProcess = Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "$ApiPort", "--log-level", "info" `
        -PassThru `
        -NoNewWindow

    Write-Host "[2/3] Waiting for FastAPI backend to become ready..." -ForegroundColor Yellow
    $Ready = $false
    $MaxAttempts = 30
    $Attempt = 0

    while (-not $Ready -and $Attempt -lt $MaxAttempts) {
        Start-Sleep -Milliseconds 800
        $Attempt++
        try {
            $Response = Invoke-RestMethod -Uri $HealthUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
            if ($Response -and ($Response.status -eq "ok" -or $Response.status -eq "healthy" -or $Response -match "ok")) {
                $Ready = $true
            }
        } catch {
            Write-Host "  ... waiting for API (attempt $Attempt/$MaxAttempts)" -ForegroundColor DarkGray
        }

        if ($ApiProcess.HasExited) {
            Write-Error "FastAPI backend process exited prematurely with code $($ApiProcess.ExitCode)."
            exit 1
        }
    }

    if (-not $Ready) {
        Write-Error "Timed out waiting for FastAPI backend at $HealthUrl."
        exit 1
    }

    Write-Host "[+] FastAPI Backend is healthy and listening on http://localhost:$ApiPort" -ForegroundColor Green
    Write-Host ""
    Write-Host "[3/3] Launching Streamlit Frontend Dashboard..." -ForegroundColor Yellow
    Write-Host "Press Ctrl+C in this console to stop both Streamlit and FastAPI backend." -ForegroundColor DarkGray
    Write-Host ""

    # Start Streamlit in the foreground so console output and Ctrl+C are properly handled
    & $PythonExe -m streamlit run app/dashboard.py
}
finally {
    Write-Host ""
    Write-Host "[*] Shutting down services..." -ForegroundColor Yellow
    if ($ApiProcess -and -not $ApiProcess.HasExited) {
        Write-Host "Stopping FastAPI backend process (PID $($ApiProcess.Id))..." -ForegroundColor DarkGray
        Stop-Process -Id $ApiProcess.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "[+] All CaseLens local services stopped cleanly." -ForegroundColor Green
}
