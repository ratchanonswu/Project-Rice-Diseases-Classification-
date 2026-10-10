@echo off
echo ====================================================
echo   RiceVision AI (Static Showcase) - Preview Server
echo ====================================================
echo.
echo Starting Vite preview on http://localhost:4173 ...
echo.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"
npm run preview

