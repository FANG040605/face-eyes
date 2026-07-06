#!/bin/bash
set -e

echo ""
echo "==================== AI Faces Pro 启动器 ===================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "[!] 错误：未找到 Python，请先安装 Python 3.8+"
        echo "    下载地址：https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "[+] Python 版本检查通过"

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "[*] 创建虚拟环境..."
    $PYTHON_CMD -m venv venv
    echo "[+] 虚拟环境创建成功"
fi

# 激活虚拟环境
echo "[*] 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "[*] 安装/更新依赖..."
pip install -r requirements.txt
echo "[+] 依赖安装完成"

echo ""
echo "==================== 启动服务 ===================="
echo ""
echo "服务地址："
echo "  - 本地：http://127.0.0.1:8005"
echo "  - 局域网：http://本机IP:8005"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动服务
python main.py