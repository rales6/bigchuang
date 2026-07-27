param(
    [ValidateSet("cpu", "cu126", "cu128")]
    [string]$TorchBackend = "cpu",
    [switch]$WithDev
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv")) {
    py -3.11 -m venv .venv
}

$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel

if ($TorchBackend -eq "cpu") {
    & $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
} else {
    & $Python -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$TorchBackend"
}

& $Python -m pip install -r requirements\qwen.txt
& $Python -m pip install -r requirements\yolo.txt

# Ultralytics installs opencv-python. Replace it with the contrib build so CSRT is available.
& $Python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless 2>$null
& $Python -m pip install --upgrade opencv-contrib-python
& $Python -m pip install -r requirements\base.txt
& $Python -m pip install -e .

if ($WithDev) {
    & $Python -m pip install -r requirements\dev.txt
}

& $Python -c "import cv2, torch; print('OpenCV', cv2.__version__); print('CSRT', hasattr(cv2,'TrackerCSRT_create') or hasattr(getattr(cv2,'legacy',object()),'TrackerCSRT_create')); print('CUDA', torch.cuda.is_available())"
Write-Host "Environment ready. Activate with: .\.venv\Scripts\Activate.ps1"
