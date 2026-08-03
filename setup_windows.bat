@echo off
REM Windows setup helper for Gemma 4 E4B Unsloth QLoRA
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating .venv ...
  python -m venv .venv
)

echo [2/5] Upgrading pip ...
.\.venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/5] Installing requirements ...
.\.venv\Scripts\pip.exe install -r requirements.txt

echo [4/5] Forcing CUDA PyTorch 2.11 + cu126 ...
.\.venv\Scripts\pip.exe install --force-reinstall torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\pip.exe install "fsspec>=2023.1.0,<=2025.9.0"

echo [5/5] Verifying ...
.\.venv\Scripts\python.exe -c "import torch; from unsloth import FastLanguageModel; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo.
echo Done. Activate with: .\.venv\Scripts\Activate.ps1
echo Then run: python prepare_dataset.py ^&^& python train.py
endlocal
