@echo off
echo ======================================
echo   RiceVision AI - Backend Startup
echo ======================================
echo.
echo Starting FastAPI backend on http://localhost:8000
echo Press Ctrl+C to stop.
echo.

set SCRIPT_DIR=%~dp0
set VENV_PYTHON=%SCRIPT_DIR%..\ResNet50\.venv\Scripts\python.exe

if not exist "%VENV_PYTHON%" (
    echo ERROR: Python venv not found at:
    echo   %VENV_PYTHON%
    echo Make sure the ResNet50 project venv exists.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --app-dir "%SCRIPT_DIR%"

