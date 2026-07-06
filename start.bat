@echo off
chcp 65001 >nul
echo.
echo ==================== AI Faces Pro ====================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.8+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python check passed

if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
echo ==================== Starting Server ====================
echo.
echo Access:
echo   Local: http://127.0.0.1:8005
echo   LAN: http://your-ip:8005
echo.
echo Press Ctrl+C to stop
echo.

python main.py

pause