@echo off
title TrustVoice Backend Server
echo ===================================================
echo   Starting TrustVoice AI Backend Server
echo ===================================================
echo.
cd /d "%~dp0"

echo [1] Checking network...
ipconfig | findstr "IPv4"
echo.

echo [1.5] Ensuring port 8000 is free...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
timeout /t 1 /nobreak >nul
echo.

echo [2] Starting server on http://0.0.0.0:8000 ...
python -m backend.main

echo.
pause
