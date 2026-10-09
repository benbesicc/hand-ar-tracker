@echo off
:: hand-ar-tracker - start script

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if not exist "%PROJECT_DIR%" (
    echo [ERROR] Project folder not found: %PROJECT_DIR%
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

set "VENV_DIR=venv"

:: verify venv python works (catches broken venv pointing to missing python etc)
set "NEED_INSTALL=0"
if not exist "%VENV_DIR%\Scripts\python.exe" set "NEED_INSTALL=1"
if "%NEED_INSTALL%"=="0" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
    if errorlevel 1 set "NEED_INSTALL=1"
)

if "%NEED_INSTALL%"=="1" (
    echo [INFO] Virtual environment "%VENV_DIR%" missing or broken - running install.bat...
    call install.bat
    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo [ERROR] install.bat failed to set up venv.
        pause
        exit /b 1
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
"%VENV_DIR%\Scripts\python.exe" main.py %*
set EXITCODE=%ERRORLEVEL%

if %EXITCODE% neq 0 (
    echo.
    echo [ERROR] main.py exited with code %EXITCODE%
) else (
    echo.
    echo [INFO] Tracker finished successfully.
)
pause
