@echo off
echo ======================================
echo   RiceVision AI - Frontend Startup
echo ======================================
echo.
echo Starting Vite dev server on http://localhost:5173
echo Press Ctrl+C to stop.
echo.

set SCRIPT_DIR=%~dp0

cd /d "%SCRIPT_DIR%frontend"
npm run dev

