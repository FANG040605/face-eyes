# AI Faces Pro API 接口文档

> 人脸识别系统 RESTful API 详细说明

---

## 一、接口概览

| 分类 | 接口 | 方法 | 说明 |
|------|------|------|------|
| 页面路由 | `/` | GET | 首页 |
| 视频流 | `/video` | GET | MJPEG 视频流 |
| 人脸注册 | `/register` | POST | 单张人脸注册 |
| 人脸注册 | `/register_batch` | POST | 批量人脸注册 |
| 人脸删除 | `/delete_face` | POST | 删除已注册人脸 |
| 图片识别 | `/predict_img` | POST | 图片人脸识别 |
| 人脸库查询 | `/known_faces` | GET | 获取已注册人脸列表 |
| 摄像头控制 | `/start_camera` | POST | 启动服务器摄像头 |
| 摄像头控制 | `/stop_camera` | POST | 停止服务器摄像头 |
| 摄像头控制 | `/capture` | GET | 抓拍当前帧 |
| 配置管理 | `/set_config` | POST | 设置检测配置 |
| 配置管理 | `/get_config` | GET | 获取当前配置 |

---

## 二、页面路由

### 2.1 GET /

**说明**: 返回系统首页（单页应用）

**请求**: 无参数

**响应**: HTML 页面

---

## 三、人脸注册接口

### 3.1 POST /register

**说明**: 注册单张人脸图片到人脸库

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 人员姓名 |
| `file` | file | 是 | 人脸图片（JPG/PNG） |

**请求示例**:

```bash
curl -X POST http://localhost:8005/register \
  -F "name=张三" \
  -F "file=@zhangsan.jpg"
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "【张三】持久注册成功（当前 1 张图片，质量 0.85）",
  "quality": 0.85,
  "total_images": 1
}
```

**失败响应** (200):

```json
{
  "success": false,
  "msg": "注册失败: 图片中未检测到人脸"
}
```

### 3.2 POST /register_batch

**说明**: 批量注册多张人脸图片

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 人员姓名 |
| `files` | file[] | 是 | 多张人脸图片 |

**请求示例**:

```bash
curl -X POST http://localhost:8005/register_batch \
  -F "name=张三" \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg" \
  -F "files=@face3.jpg"
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "【张三】批量持久注册成功，融合 3 张图片",
  "images_used": 3,
  "avg_quality": 0.82
}
```

---

## 四、人脸删除接口

### 4.1 POST /delete_face

**说明**: 删除已注册的人脸数据

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 要删除的人员姓名 |

**请求示例**:

```bash
curl -X POST http://localhost:8005/delete_face \
  -F "name=张三"
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "【张三】删除成功"
}
```

---

## 五、图片识别接口

### 5.1 POST /predict_img

**说明**: 上传图片进行人脸识别，支持多人脸检测

**请求参数**:

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 待识别图片（JPG/PNG） |

**请求示例**:

```bash
curl -X POST http://localhost:8005/predict_img \
  -F "file=@test.jpg"
```

**成功响应** (200):

```json
{
  "face_count": 2,
  "results": [
    {
      "name": "张三",
      "confidence": 0.95,
      "distance": 0.35,
      "location": [100, 200, 300, 100]
    },
    {
      "name": "未知人员",
      "confidence": 0.80,
      "distance": 0.55,
      "location": [150, 250, 350, 150]
    }
  ],
  "annotated_img": "data:image/jpeg;base64,..."
}
```

**响应字段说明**:

| 字段 | 说明 |
|------|------|
| `face_count` | 检测到的人脸数量 |
| `results[].name` | 识别出的姓名，"未知人员"表示未匹配 |
| `results[].confidence` | 识别置信度 (0-1) |
| `results[].distance` | 与匹配人脸的距离 (越小越相似) |
| `results[].location` | 人脸位置 [top, right, bottom, left] |
| `annotated_img` | 标注后的图片（Base64 编码） |

---

## 六、人脸库查询接口

### 6.1 GET /known_faces

**说明**: 获取已注册的人脸列表

**请求参数**: 无

**请求示例**:

```bash
curl http://localhost:8005/known_faces
```

**成功响应** (200):

```json
{
  "faces": [
    {
      "name": "张三",
      "image_count": 3,
      "updated": 1783332717.09
    },
    {
      "name": "李四",
      "image_count": 1,
      "updated": 1783332750.12
    }
  ],
  "count": 2
}
```

---

## 七、摄像头控制接口

### 7.1 POST /start_camera

**说明**: 启动服务器端电脑摄像头

**请求参数**: 无

**请求示例**:

```bash
curl -X POST http://localhost:8005/start_camera
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "摄像头已启动"
}
```

### 7.2 POST /stop_camera

**说明**: 停止服务器端电脑摄像头

**请求参数**: 无

**请求示例**:

```bash
curl -X POST http://localhost:8005/stop_camera
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "摄像头已停止"
}
```

### 7.3 GET /video

**说明**: 获取 MJPEG 视频流（实时摄像头画面）

**请求参数**: 无

**请求示例**:

```bash
# 在浏览器中访问
http://localhost:8005/video
```

**响应**: MJPEG 视频流

### 7.4 GET /capture

**说明**: 抓拍当前摄像头画面

**请求参数**: 无

**请求示例**:

```bash
curl http://localhost:8005/capture --output capture.jpg
```

**响应**: JPEG 图片文件

---

## 八、配置管理接口

### 8.1 POST /set_config

**说明**: 设置检测参数

**请求参数**:

| 参数 | 类型 | 必填 | 范围 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| `confidence_threshold` | float | 否 | 0.1-0.95 | 0.70 | 人脸检测置信度阈值 |
| `match_threshold` | float | 否 | 0.2-0.6 | 0.45 | 人脸识别匹配阈值 |

**请求示例**:

```bash
curl -X POST http://localhost:8005/set_config \
  -F "confidence_threshold=0.70" \
  -F "match_threshold=0.45"
```

**成功响应** (200):

```json
{
  "success": true,
  "msg": "配置更新成功"
}
```

### 8.2 GET /get_config

**说明**: 获取当前检测配置

**请求参数**: 无

**请求示例**:

```bash
curl http://localhost:8005/get_config
```

**成功响应** (200):

```json
{
  "confidence_threshold": 0.70,
  "detection_interval": 100,
  "match_threshold": 0.45,
  "encoding_workers": 4,
  "batch_size": 8,
  "min_face_quality": 0.6,
  "max_faces_per_person": 10,
  "use_gpu": false,
  "encoding_model": "small"
}
```

---

## 九、错误码说明

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误 |
| 404 | 接口不存在 |
| 500 | 服务器内部错误 |

---

**版本**: v2.0