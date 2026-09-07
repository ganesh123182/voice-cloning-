@echo off
echo ========================================================
echo   Starting Deepfake Voice Detector Setup...
echo ========================================================
echo.

REM Navigate to the project directory (where this .bat file is located)
cd /d "%~dp0"

echo [1/3] Checking dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Make sure Python is installed.
    pause
    exit /b %errorlevel%
)
echo [OK] Dependencies ready.
echo.

REM Kill any leftover process on port 8001 before starting
echo [2/3] Checking for stale processes on port 8001...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do (
    echo Killing stale process PID %%a on port 8001...
    taskkill /F /PID %%a >nul 2>&1
)
echo [OK] Port 8001 is free.
echo.

echo [3/3] Starting Server...
echo The server will open in a few seconds. Do not close this window.
echo.

REM Open browser after a short delay (runs in background)
start /b cmd /c "timeout /t 4 /nobreak >nul & start "" http://127.0.0.1:8001"

REM Start server in foreground (keeps window open)
python -m backend.main
pause

