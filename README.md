# AI Faces Pro 人脸识别系统

> 基于 FastAPI + OpenCV + face_recognition 的高性能人脸识别 Web 服务

---

## 一、项目概述

### 1.1 功能特性

| 功能模块 | 说明 | 状态 |
|----------|------|------|
| 人脸注册入库 | 支持图片上传注册新人员，持久化存储 | ✅ 已实现 |
| 图片人脸识别 | 上传图片进行批量识别，支持多人脸 | ✅ 已实现 |
| 电脑摄像头实时识别 | 基于 MJPEG 流的低延迟实时检测 | ✅ 已实现 |
| 手机摄像头实时识别 | 移动端调用本机摄像头，支持 HTTPS | ✅ 已实现 |
| 局域网部署 | 支持 HTTPS 加密访问，跨设备兼容 | ✅ 已实现 |
| 双线程优化 | 采集与检测分离，画面流畅无卡顿 | ✅ 已实现 |
| 人脸库持久化 | 重启后自动加载已注册人脸 | ✅ 已实现 |

### 1.2 技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| 后端框架 | FastAPI | 0.138+ |
| 服务器 | Uvicorn | 0.49+ |
| 人脸检测 | OpenCV DNN | 4.13+ |
| 人脸识别 | face_recognition | 1.3.0+ |
| 图像处理 | PIL/Pillow | 10.x |
| 向量计算 | NumPy | 1.24+ |
| SSL 证书 | cryptography | 42.x |

### 1.3 核心优势

- **高性能**: 双线程架构，采集与检测分离
- **高精度**: 双重验证匹配算法（距离阈值 + 差异检测）
- **跨平台**: 支持 Windows/macOS/Linux，移动端访问
- **持久化**: 人脸数据持久存储，重启不丢失
- **易部署**: 一键启动，自动生成 SSL 证书

---

## 二、一键快速开始

### 2.1 前置条件

- **必须安装**: Python 3.8+（[下载地址](https://www.python.org/downloads/)）
- **可选**: 摄像头（用于实时识别功能）

### 2.2 启动方式

#### Windows

```powershell
# 克隆项目
git clone https://github.com/FANG040605/face-eyes.git
cd face-eyes

# 运行启动器（PowerShell）
.\start.bat

# 或者直接双击 start.bat 文件
```

**或者手动运行**:

```powershell
# 创建虚拟环境（首次运行）
python -m venv venv

# 激活虚拟环境
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

#### macOS / Linux

```bash
# 克隆项目
git clone https://github.com/FANG040605/face-eyes.git
cd face-eyes

# 运行启动器
chmod +x start.sh
./start.sh
```

**或者手动运行**:

```bash
# 创建虚拟环境（首次运行）
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py
```

### 2.3 访问服务

| 访问方式 | 地址 |
|----------|------|
| 本地 | `http://127.0.0.1:8005` |
| 局域网 | `http://192.168.x.x:8005` |
| 手机 | 连接同一 Wi-Fi，访问局域网地址 |

---

## 三、项目结构

```
face-eyes/
├── main.py                                   # 主程序入口（核心代码）
├── index.html                                # 前端页面（单页应用）
├── requirements.txt                          # Python 依赖列表
├── start.bat                                 # Windows 一键启动器
├── start.sh                                  # macOS/Linux 一键启动器
├── test_performance.py                       # 性能测试脚本
├── deploy.prototxt                           # DNN 模型配置文件
├── res10_300x300_ssd_iter_140000_fp16.caffemodel  # DNN 模型权重
├── storage/                                  # 持久化存储目录（自动创建）
│   ├── known_face/                           # 已注册人脸库
│   ├── upload_temp/                          # 上传临时目录
│   ├── camera_capture/                       # 摄像头抓拍
│   ├── face_meta/                            # 人脸元数据
│   └── dnn_models/                           # DNN 模型缓存
├── test_samples/                             # 测试样本图片
├── cert.pem                                  # SSL 证书（自动生成）
└── key.pem                                   # SSL 私钥（自动生成）
```

---

## 四、使用说明

### 4.1 注册人脸

1. 访问首页
2. 点击「人脸注册」标签页
3. 输入人员姓名
4. 上传包含清晰人脸的图片（JPG/PNG）
5. 点击「提交注册」

### 4.2 图片识别

1. 访问首页
2. 点击「图片识别」标签页
3. 上传待识别图片
4. 点击「开始识别」
5. 查看识别结果（人脸框、姓名、置信度）

### 4.3 实时摄像头识别

1. 访问首页
2. 点击「实时摄像头」标签页
3. 选择摄像头来源：
   - **电脑摄像头**: 使用服务器端电脑的摄像头
   - **本机摄像头**: 使用当前设备（手机/平板）的摄像头
4. 点击「启动摄像头」
5. 实时显示识别结果（绿色框=已注册，红色框=未知）

---

## 五、配置参数

### 5.1 检测参数

| 参数 | 范围 | 默认值 | 说明 |
|------|------|--------|------|
| `confidence_threshold` | 0.1-0.95 | 0.70 | DNN 人脸检测置信度阈值 |
| `match_threshold` | 0.2-0.6 | 0.45 | 人脸识别匹配阈值（越小越严格） |
| `detection_interval` | 100-1000ms | 100ms | 检测间隔 |

### 5.2 调优建议

| 场景 | 调整策略 |
|------|----------|
| 误识别多 | 降低 `match_threshold`（如 0.40） |
| 漏识别多 | 提高 `match_threshold`（如 0.50） |
| 画面卡顿 | 增大 `detection_interval`（如 400ms） |
| 检测不灵敏 | 降低 `confidence_threshold`（如 0.60） |

---

## 六、相关文档

| 文档 | 说明 |
|------|------|
| [API接口文档](api.md) | 所有 API 接口详细说明 |
| [部署手册](deploy.md) | 安装部署步骤 |
| [测试报告](test_report.md) | 功能测试和性能测试结果 |

---

## 七、许可证

MIT License

---

**版本**: v2.0  
**作者**: AI Faces Team