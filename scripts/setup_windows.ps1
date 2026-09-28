# PkVision Windows Setup Script
# Run on windows-dev after transferring files
# PowerShell: .\scripts\setup_windows.ps1

Write-Host "PkVision Windows Setup" -ForegroundColor Green
Write-Host "======================" -ForegroundColor Green

# Extract frames
$framesDir = "C:\Users\pc\pkvision\data\v5_full_training\frames"
if (-not (Test-Path $framesDir)) {
    Write-Host "Extracting training frames..."
    Set-Location "C:\Users\pc\pkvision"
    tar xzf v5_frames.tar.gz -C data\v5_full_training
    $count = (Get-ChildItem -Path $framesDir -Recurse -Filter "*.npy").Count
    Write-Host "Extracted $count frame files"
} else {
    $count = (Get-ChildItem -Path $framesDir -Recurse -Filter "*.npy").Count
    Write-Host "Frames already extracted: $count files"
}

# Create Python venv with CUDA support
$venvPath = "C:\Users\pc\pkvision\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating Python virtual environment..."
    python -m venv $venvPath
    & "$venvPath\Scripts\pip.exe" install torch torchvision transformers numpy opencv-python-headless accelerate scipy
    Write-Host "Dependencies installed"
} else {
    Write-Host "Venv already exists"
}

# Verify CUDA
Write-Host ""
Write-Host "Checking CUDA..."
& "$venvPath\Scripts\python.exe" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

Write-Host ""
Write-Host "Ready! Run training with:"
Write-Host "  cd C:\Users\pc\pkvision"
Write-Host "  .\venv\Scripts\python.exe scripts\train_metric_local.py --epochs 30 --batch-size 8 --device cuda"
