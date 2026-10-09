@echo off
:: hand-ar-tracker - install dependencies

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%"

:: force using python or py launcher
set "PY=python"
where py >nul 2>&1 && set "PY=py -3.12"
where py >nul 2>&1 || (
    where python >nul 2>&1 || (
        echo [ERROR] Python not found. Install Python 3.12 from python.org
        pause
        exit /b 1
    )
)

echo [INFO] Using: %PY%
%PY% --version

set "VENV_DIR=venv"

:create_venv
echo [INFO] Creating virtual environment "%VENV_DIR%"...
%PY% -m venv "%VENV_DIR%"
if errorlevel 1 (
    :: fallback to just python if py -312 fails
    if not "%PY%"=="python" (
        echo [WARNING] Failed with py -3.12, trying default python interpreter...
        set "PY=python"
        goto create_venv
    )
    echo [ERROR] Failed to create virtual environment "%VENV_DIR%".
    pause
    exit /b 1
)

:: upgrade pip
echo [INFO] Upgrading pip...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

:: install core dependencies
echo.
echo [INFO] Installing core dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies.
    pause
    exit /b 1
)

:: install pytorch with cuda support (auto-detects nvidia gpu)
echo.
echo [INFO] Installing PyTorch with CUDA support (auto-detects NVIDIA GPU)...
"%VENV_DIR%\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [WARNING] CUDA PyTorch install failed. Installing CPU???only PyTorch...
    "%VENV_DIR%\Scripts\python.exe" -m pip install torch torchvision torchaudio
)

:: verify installation
echo.
echo [INFO] Verifying installation...
"%VENV_DIR%\Scripts\python.exe" -c "import mediapipe as mp; print('[OK] MediaPipe', mp.__version__)"
"%VENV_DIR%\Scripts\python.exe" -c "import cv2, yaml, numpy; print('[OK] OpenCV', cv2.__version__, 'NumPy', numpy.__version__)"
"%VENV_DIR%\Scripts\python.exe" -c "import torch; print('[OK] PyTorch', torch.__version__, 'CUDA available:', torch.cuda.is_available()); import torch as t; [print('  GPU:', t.cuda.get_device_name(i)) for i in range(t.cuda.device_count())] if t.cuda.is_available() else None"

echo.
echo [SUCCESS] Installation complete! Run start.bat to launch the tracker.
pause
