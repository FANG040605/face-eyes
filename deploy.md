# AI Faces Pro 部署手册

> 人脸识别系统安装部署步骤

---

## 一、环境准备

### 1.1 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 双核 2.0GHz | 四核 2.5GHz+ |
| 内存 | 4GB | 8GB+ |
| 硬盘 | 1GB 可用空间 | 10GB+ |
| 摄像头 | USB 摄像头（可选） | USB 高清摄像头 |

### 1.2 软件要求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 编程语言 |
| pip | 20.0+ | 包管理工具 |
| Git | 2.0+ | 版本控制（可选） |

### 1.3 操作系统支持

| 操作系统 | 版本 | 状态 |
|----------|------|------|
| Windows | 10/11 | ✅ 支持 |
| macOS | 10.15+ | ✅ 支持 |
| Linux | Ubuntu 18.04+ | ✅ 支持 |

---

## 二、Windows 部署

### 2.1 步骤一：安装 Python

1. 下载 Python 3.8+ 安装包：https://www.python.org/downloads/windows/
2. 运行安装程序，勾选「Add Python to PATH」
3. 完成安装后验证：

```powershell
python --version
pip --version
```

### 2.2 步骤二：创建虚拟环境

```powershell
# 创建项目目录
mkdir d:\ai-faces-practice
cd d:\ai-faces-practice

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate
```

### 2.3 步骤三：安装依赖

```powershell
# 安装核心依赖
pip install fastapi uvicorn python-multipart pillow numpy opencv-python face-recognition cryptography
```

### 2.4 步骤四：部署项目文件

将以下文件复制到项目目录中：

| 文件 | 说明 |
|------|------|
| `main.py` | 主程序代码 |
| `index.html` | 前端页面 |
| `models/deploy.prototxt` | DNN 模型配置 |
| `models/res10_300x300_ssd_iter_140000_fp16.caffemodel` | DNN 模型权重 |

### 2.5 步骤五：启动服务

```powershell
cd d:\face-eyes

# 方式1：HTTP 模式（开发环境）
venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8005

# 方式2：HTTPS 模式（生产环境，自动生成证书）
venv\Scripts\python.exe main.py
```

### 2.6 步骤六：验证部署

1. 打开浏览器访问：`http://127.0.0.1:8005`
2. 检查页面是否正常加载
3. 测试人脸注册和识别功能

---

## 三、Linux 部署

### 3.1 步骤一：安装 Python

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip python3-venv
```

### 3.2 步骤二：创建虚拟环境

```bash
mkdir -p /opt/ai-faces-practice
cd /opt/ai-faces-practice

python3 -m venv venv
source venv/bin/activate
```

### 3.3 步骤三：安装依赖

```bash
# 安装系统依赖（face-recognition 需要）
sudo apt install build-essential cmake libopenblas-dev liblapack-dev python3-dev

# 安装 Python 依赖
pip install fastapi uvicorn python-multipart pillow numpy opencv-python face-recognition cryptography
```

### 3.4 步骤四：部署项目文件

```bash
mkdir -p /opt/face-eyes
# 将项目文件复制到该目录
```

### 3.5 步骤五：启动服务

```bash
cd /opt/face-eyes
source venv/bin/activate

# 启动服务
uvicorn main:app --host 0.0.0.0 --port 8005
```

---

## 四、macOS 部署

### 4.1 步骤一：安装 Python

```bash
# 使用 Homebrew 安装
brew install python3

# 验证安装
python3 --version
```

### 4.2 步骤二：创建虚拟环境

```bash
mkdir -p ~/ai-faces-practice
cd ~/ai-faces-practice

python3 -m venv venv
source venv/bin/activate
```

### 4.3 步骤三：安装依赖

```bash
pip install fastapi uvicorn python-multipart pillow numpy opencv-python face-recognition cryptography
```

### 4.4 步骤四：启动服务

```bash
cd ~/face-eyes
uvicorn main:app --host 0.0.0.0 --port 8005
```

---

## 五、局域网部署

### 5.1 配置网络

确保服务器和客户端在同一局域网内：

1. 服务器连接到局域网（有线或无线）
2. 获取服务器 IP 地址：

**Windows**:
```powershell
ipconfig
# 查找 IPv4 地址，如 192.168.1.105
```

**Linux/macOS**:
```bash
ifconfig
# 或
ip addr show
```

### 5.2 客户端访问

| 设备 | 访问地址 |
|------|----------|
| Windows | `http://192.168.1.105:8005` |
| macOS | `http://192.168.1.105:8005` |
| Android | `http://192.168.1.105:8005` |
| iOS | `http://192.168.1.105:8005` |

### 5.3 移动端摄像头权限

移动端使用本机摄像头需要 HTTPS：

```powershell
# 使用 HTTPS 模式启动
venv\Scripts\python.exe main.py
```

访问地址：`https://192.168.1.105:8005`

---

## 六、生产环境部署

### 6.1 使用 Nginx 反向代理

创建 Nginx 配置文件 `/etc/nginx/sites-available/ai-faces`:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8005;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 6.2 配置 Systemd 服务

创建 `/etc/systemd/system/ai-faces.service`:

```ini
[Unit]
Description=AI Faces Pro Face Recognition Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/face-eyes
Environment="PATH=/opt/face-eyes/venv/bin"
ExecStart=/opt/face-eyes/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8005
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-faces
sudo systemctl start ai-faces
```

---

## 七、防火墙配置

### 7.1 Windows 防火墙

```powershell
# 允许端口 8005
New-NetFirewallRule -DisplayName "AI Faces Pro" -Direction Inbound -Protocol TCP -LocalPort 8005 -Action Allow
```

### 7.2 Linux 防火墙

```bash
# Ubuntu/Debian
sudo ufw allow 8005/tcp
sudo ufw enable

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8005/tcp
sudo firewall-cmd --reload
```

---

## 八、故障排除

### 8.1 常见问题

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| 端口被占用 | 8005 端口已被使用 | 修改端口或关闭占用程序 |
| 摄像头无法打开 | 权限不足或被占用 | 关闭其他使用摄像头的程序 |
| face_recognition 安装失败 | 缺少系统依赖 | 安装 cmake、build-essential |
| 图片无法上传 | 文件大小限制 | 检查上传文件大小 |
| 手机无法访问 | 网络配置问题 | 确保在同一局域网 |

### 8.2 日志查看

```powershell
# Windows：查看控制台输出

# Linux：查看 systemd 日志
sudo journalctl -u ai-faces -f
```

---

**版本**: v2.0