#!/bin/bash
set -e

echo ""
echo "==================== AI Faces Pro ===================="
echo ""

if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[!] Error: Python not found. Please install Python 3.8+"
        echo "    Download: https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "[OK] Python check passed"

if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "[OK] Virtual environment created"
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Installing dependencies (using Alibaba mirror)..."
pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[WARN] Failed to install via Alibaba mirror, trying default..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi
echo "[OK] Dependencies installed"

echo ""
echo "[INFO] Checking for port 8005..."
PID=$(lsof -ti:8005 2>/dev/null || netstat -tlnp | grep :8005 | awk '{print $7}' | cut -d'/' -f1 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "[INFO] Killing process on port 8005 (PID: $PID)..."
    kill -9 $PID 2>/dev/null || true
    sleep 1
fi
echo "[OK] Port 8005 is ready"

echo ""
echo "==================== Starting Server ===================="
echo ""
echo "Access:"
echo "  Local: https://127.0.0.1:8005"
echo "  LAN: https://your-ip:8005"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python main.py