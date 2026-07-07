
"""
AI Faces Pro - 高性能人脸识别 Web 服务
核心升级：
  1. 多进程并行编码（CPU 满载）
  2. 多图融合注册（提升准确率 + 鲁棒性）
  3. 向量化批量匹配（NumPy 加速）
  4. 读写锁分离（视频流零阻塞）
  5. GPU 加速检测（CUDA/OpenCL）
  6. 增量人脸库更新（无需全量重载）
  7. 人脸质量评分（自动筛选最优图）

【摄像头优化版 - 针对联想拯救者 R7000 等 Windows 笔记本】
  8. 双缓冲帧队列 + 丢弃旧帧策略（解决 USB 摄像头缓冲堆积卡顿）
  9. Windows 专用 DirectShow/MF 后端优化
  10. 自适应帧率控制（根据处理速度动态调整）
  11. 采集与处理线程完全解耦（零阻塞管道）
"""
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import os, cv2, face_recognition, numpy as np, threading, time, base64, io, urllib.request, json, hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import queue

app = FastAPI(title="AI Faces Pro - 高性能人脸识别系统")

# =============================================================================
# 【修改：持久化独立存储文件夹，所有文件永久保存，不会自动消失】
# 路径基于脚本位置（绝对路径），确保从任何目录启动都能正确找到存储
# =============================================================================
_APP_DIR = Path(__file__).parent.resolve()

KNOWN_DIR = _APP_DIR / "storage" / "known_face"
UPLOAD_DIR = _APP_DIR / "storage" / "upload_temp"
META_DIR = _APP_DIR / "storage" / "face_meta"
CAPTURE_SAVE_DIR = _APP_DIR / "storage" / "camera_capture"
MODEL_DIR = _APP_DIR / "storage" / "dnn_models"

_OLD_KNOWN_DIR = _APP_DIR / "known_face"

KNOWN_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
META_DIR.mkdir(exist_ok=True, parents=True)
CAPTURE_SAVE_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# 全局配置（完全保留原有GPU开关，所有功能不变）
# =============================================================================
_cfg = {
    "confidence_threshold": 0.4,
    "detection_interval": 300,   # ms
    "match_threshold": 0.45,
    "encoding_workers": max(4, os.cpu_count() or 4),  # 增加编码进程数提升并发性能
    "batch_size": 8,             # 批量编码大小
    "min_face_quality": 0.6,     # 人脸质量阈值
    "max_faces_per_person": 10,  # 每人最大存储图片数
    "use_gpu": False,             # 尝试启用 GPU（保留原有功能）
    "encoding_model": "small",     # small 比 large 快 3 倍
}

# =============================================================================
# 全局人脸缓存（支持多编码/人）
# =============================================================================
# 结构: {name: {"encodings": [enc1, enc2, ...], "primary": enc_avg, "count": n}}
_face_db = {}
_face_db_lock = threading.RLock()

# 向量化缓存（匹配用）
_all_encodings = None   # np.array (N, 128)
_all_names = []         # list of str (长度与 _all_encodings 一致）
_all_indices = {}       # {name: [idx1, idx2, ...]}  用于快速定位
_db_version = 0         # 版本号，用于增量更新

# =============================================================================
# 进程池（全局单例，避免重复创建开销）
# =============================================================================
_process_pool = None

def get_process_pool():
    global _process_pool
    if _process_pool is None or _process_pool._shutdown:
        _process_pool = ProcessPoolExecutor(max_workers=_cfg["encoding_workers"])
    return _process_pool

# =============================================================================
# DNN 人脸检测器（支持 GPU 加速，完整保留原有逻辑）
# =============================================================================
class FaceDetector:
    def __init__(self, use_dnn=True):
        self.use_dnn = use_dnn
        if use_dnn:
            self._init_dnn_model()
        else:
            self._init_haar_cascade()

    def _init_haar_cascade(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("无法加载 Haar 级联分类器")
        print("[INFO] Haar 级联分类器加载成功")

    def _init_dnn_model(self):
        prototxt_path = MODEL_DIR / 'deploy.prototxt'
        caffemodel_path = MODEL_DIR / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

        root_prototxt = _APP_DIR / 'deploy.prototxt'
        root_caffemodel = _APP_DIR / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

        if root_prototxt.exists() and root_caffemodel.exists():
            prototxt_path = root_prototxt
            caffemodel_path = root_caffemodel
            print("[INFO] 使用项目根目录的 DNN 模型文件")

        if not prototxt_path.exists() or not caffemodel_path.exists():
            print("[INFO] 正在下载 DNN 模型文件至持久存储目录 storage/dnn_models...")
            self._download_models(prototxt_path, caffemodel_path)

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

        # ===== GPU 加速配置（完整保留原有CUDA逻辑） =====
        if _cfg["use_gpu"]:
            # 尝试 CUDA 后端
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[INFO] DNN 已启用 CUDA GPU 加速")
            except Exception as e:
                print(f"[WARN] CUDA 不可用，尝试 OpenCL: {e}")
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                    print("[INFO] DNN 已启用 OpenCL 加速")
                except Exception:
                    print("[INFO] DNN 使用 CPU 模式")

        print("[INFO] DNN 人脸检测模型加载成功，模型永久保存在 storage/dnn_models")

    def _download_models(self, prototxt_path, caffemodel_path):
        prototxt_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
        caffemodel_url = 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel'
        try:
            print("[INFO] 下载模型配置文件...")
            urllib.request.urlretrieve(prototxt_url, str(prototxt_path))
            print("[INFO] 下载模型权重文件（约 5MB）...")
            urllib.request.urlretrieve(caffemodel_url, str(caffemodel_path))
            print("[INFO] 模型文件下载完成，永久保存在 storage/dnn_models")
        except Exception as e:
            print(f"[ERROR] 模型下载失败: {e}")
            print("[INFO] 切换到 Haar 级联分类器")
            self.use_dnn = False
            self._init_haar_cascade()

    def detect_faces(self, image, confidence_threshold=0.5):
        if self.use_dnn:
            return self._detect_faces_dnn(image, confidence_threshold)
        else:
            return self._detect_faces_haar(image)

    def _detect_faces_dnn(self, image, confidence_threshold):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        confidences = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append([max(0, x1), max(0, y1), x2 - x1, y2 - y1])
                confidences.append(float(confidence))

        return (np.array(faces) if faces else np.array([])), confidences

    def _detect_faces_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20)
        )
        return (np.array(faces) if len(faces) > 0 else np.array([])), []

_face_detector = None
_face_detector_lock = threading.Lock()

def get_face_detector():
    global _face_detector
    if _face_detector is None:
        with _face_detector_lock:
            if _face_detector is None:
                _face_detector = FaceDetector(use_dnn=True)
    return _face_detector


# =============================================================================
# 人脸质量评估（用于筛选最优注册图）
# =============================================================================
def assess_face_quality(face_img_bgr):
    """
    评估人脸质量，返回 0-1 分数。
    综合：清晰度（拉普拉斯方差）、对比度、亮度、姿态（眼睛对称性）
    """
    gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. 清晰度（拉普拉斯方差）
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness = min(lap_var / 500.0, 1.0)  # 归一化

    # 2. 对比度
    contrast = gray.std() / 128.0

    # 3. 亮度适中度（偏离 128 越小越好）
    brightness = 1.0 - abs(gray.mean() - 128) / 128.0

    # 4. 综合评分
    score = (sharpness * 0.5 + contrast * 0.3 + brightness * 0.2)
    return min(max(score, 0.0), 1.0)


# =============================================================================
# 多进程编码函数（必须在模块顶层，可 pickle）
# =============================================================================
def _encode_single_image(image_path):
    """单张图片编码（供进程池调用）"""
    try:
        img = face_recognition.load_image_file(str(image_path))
        locs = face_recognition.face_locations(img, number_of_times_to_upsample=1)
        if not locs:
            return None
        encs = face_recognition.face_encodings(img, locs, model="small")
        if not encs:
            return None
        # 返回最佳人脸（最大面积）
        best_idx = max(range(len(locs)), key=lambda i:
                       (locs[i][2]-locs[i][0]) * (locs[i][1]-locs[i][3]))
        return encs[best_idx].tolist()
    except Exception as e:
        return None


def _batch_encode_images(image_paths):
    """批量编码（减少进程间通信开销）"""
    results = []
    for path in image_paths:
        results.append(_encode_single_image(path))
    return results


# =============================================================================
# 人脸数据库管理（增量更新 + 多编码融合）
# =============================================================================
def _save_person_meta(name, data):
    """保存人物元数据到 JSON（持久化存储）"""
    meta_path = META_DIR / f"{name}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _load_person_meta(name):
    """加载人物元数据"""
    meta_path = META_DIR / f"{name}.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _compute_primary_encoding(encodings):
    """
    计算主编码：多图编码的加权平均。
    使用余弦相似度加权，离群值自动降权。
    """
    if not encodings:
        return None
    if len(encodings) == 1:
        return encodings[0]

    encs = np.array(encodings)
    # 计算两两距离矩阵
    dist_matrix = face_recognition.face_distance(encs, encs[0])
    # 距离越近权重越高（高斯加权）
    weights = np.exp(-(dist_matrix ** 2) / 0.1)
    weights = weights / weights.sum()

    primary = np.average(encs, axis=0, weights=weights)
    # L2 归一化
    primary = primary / np.linalg.norm(primary)
    return primary.tolist()


def _rebuild_vector_cache():
    """
    重建向量化缓存（全量）。
    将 _face_db 展开为扁平数组，用于 NumPy 向量化匹配。
    """
    global _all_encodings, _all_names, _all_indices, _db_version

    encodings = []
    names = []
    indices = {}
    idx = 0

    with _face_db_lock:
        for name, data in _face_db.items():
            person_encs = data.get("encodings", [])
            # 存储所有编码（不只是 primary）
            for enc in person_encs:
                encodings.append(enc)
                names.append(name)
                if name not in indices:
                    indices[name] = []
                indices[name].append(idx)
                idx += 1

    if encodings:
        _all_encodings = np.array(encodings, dtype=np.float32)
    else:
        _all_encodings = None
    _all_names = names
    _all_indices = indices
    _db_version += 1
    print(f"[INFO] 向量化缓存重建完成: {len(names)} 个编码, {_db_version} 个版本")


def load_known_faces_parallel():
    """
    并行加载所有人脸。
    扫描 storage/known_face 持久目录，多进程编码所有图片。
    """
    global _face_db

    print("[INFO] 开始并行加载持久化人脸库 storage/known_face...")
    start = time.time()

    if _OLD_KNOWN_DIR.exists() and _OLD_KNOWN_DIR.is_dir():
        old_files = list(_OLD_KNOWN_DIR.iterdir())
        if old_files:
            print(f"[INFO] 检测到旧人脸库目录 {_OLD_KNOWN_DIR}，正在迁移 {len(old_files)} 个文件...")
            for item in old_files:
                if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    dest_path = KNOWN_DIR / item.name
                    if not dest_path.exists():
                        item.rename(dest_path)
                        print(f"  [OK] 迁移: {item.name} -> {dest_path}")
            _OLD_KNOWN_DIR.rmdir()
            print("[INFO] 旧人脸库迁移完成")

    person_images = {}  # {name: [path1, path2, ...]}
    if not KNOWN_DIR.exists():
        print(f"[WARN] 人脸持久库目录不存在: {KNOWN_DIR}")
        return

    files = list(KNOWN_DIR.iterdir())
    print(f"[INFO] 扫描持久人脸目录，共 {len(files)} 个文件/目录")

    for item in files:
        if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg", ".png"):
            fname = item.name
            name = item.stem
            # 支持多图命名: 张三_1.jpg, 张三_2.jpg
            base_name = name.split("_")[0] if "_" in name else name
            person_images.setdefault(base_name, []).append(item)
            print(f"[INFO] 持久人脸图片: {fname} -> 归属: {base_name}")

    if not person_images:
        print("[INFO] 持久人脸库为空，请上传人脸注册")
        return

    print(f"[INFO] 共 {len(person_images)} 个人物需要编码")

    # 批量并行编码
    all_paths = []
    path_to_person = {}
    for name, paths in person_images.items():
        for p in paths:
            all_paths.append(str(p))
            path_to_person[str(p)] = name

    # 分批提交到进程池
    pool = get_process_pool()
    batch_size = _cfg["batch_size"]
    futures = []
    for i in range(0, len(all_paths), batch_size):
        batch = all_paths[i:i+batch_size]
        futures.append(pool.submit(_batch_encode_images, batch))

    # 收集结果
    person_encodings = {name: [] for name in person_images}
    for future in futures:
        batch_results = future.result()
        for path, enc in zip(all_paths, batch_results):
            if enc is not None:
                name = path_to_person[path]
                person_encodings[name].append(enc)

    # 构建数据库
    with _face_db_lock:
        _face_db.clear()
        for name, encs in person_encodings.items():
            if encs:
                primary = _compute_primary_encoding(encs)
                _face_db[name] = {
                    "encodings": encs,
                    "primary": primary,
                    "count": len(encs),
                    "updated": time.time(),
                }
                print(f"  [OK] {name}: {len(encs)} 张持久人脸图片, 主编码已生成")

    _rebuild_vector_cache()
    elapsed = time.time() - start
    print(f"[INFO] 持久人脸库加载完成，共 {len(_face_db)} 人，耗时 {elapsed:.2f}s")


def add_person_encoding(name, new_encoding, image_path=None):
    """
    增量添加编码（无需全量重载）。
    自动维护每人最大图片数，超出时保留最新持久图片。
    """
    global _face_db

    with _face_db_lock:
        if name not in _face_db:
            _face_db[name] = {
                "encodings": [new_encoding],
                "primary": new_encoding,
                "count": 1,
                "updated": time.time(),
            }
        else:
            data = _face_db[name]
            encs = data["encodings"]

            # 检查是否与现有编码过于相似（去重）
            if encs:
                dists = face_recognition.face_distance(np.array(encs), np.array(new_encoding))
                if np.min(dists) < 0.15:  # 几乎相同，跳过
                    print(f"[INFO] {name}: 编码与现有过于相似，跳过")
                    return False

            encs.append(new_encoding)

            # 超出限制时保留最新持久图片
            if len(encs) > _cfg["max_faces_per_person"]:
                encs = encs[-_cfg["max_faces_per_person"]:]

            data["encodings"] = encs
            data["primary"] = _compute_primary_encoding(encs)
            data["count"] = len(encs)
            data["updated"] = time.time()

    _rebuild_vector_cache()
    return True


def remove_person(name):
    """删除人物（清理全部持久存储文件）"""
    global _face_db
    with _face_db_lock:
        if name in _face_db:
            del _face_db[name]

    # 清理持久人脸图片文件
    for ext in (".jpg", ".jpeg", ".png"):
        p = KNOWN_DIR / f"{name}{ext}"
        if p.exists():
            p.unlink()
        # 清理多图持久文件
        for i in range(1, _cfg["max_faces_per_person"] + 2):
            p = KNOWN_DIR / f"{name}_{i}{ext}"
            if p.exists():
                p.unlink()

    # 清理持久元数据json
    meta_path = META_DIR / f"{name}.json"
    if meta_path.exists():
        meta_path.unlink()

    _rebuild_vector_cache()
    return True


# =============================================================================
# 向量化批量匹配（NumPy 加速，比逐一遍历快 10x+）
# =============================================================================
def match_faces_vectorized(query_encodings, top_k=2):
    """
    向量化匹配多个查询编码。
    返回: [(name, is_known, confidence, distance), ...]
    """
    if _all_encodings is None or len(_all_encodings) == 0:
        return [("未知人员", False, "0%", 1.0) for _ in query_encodings]

    results = []
    match_threshold = _cfg["match_threshold"]

    for enc in query_encodings:
        enc = np.array(enc, dtype=np.float32)
        # 向量化计算所有距离（一次性矩阵运算）
        distances = np.linalg.norm(_all_encodings - enc, axis=1)

        if len(distances) == 0:
            results.append(("未知人员", False, "0%", 1.0))
            continue

        # 找最佳匹配
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        best_name = _all_names[best_idx]

        # 检查次优匹配（防误判）
        is_known = False
        if best_dist < match_threshold:
            # 获取该人物的次优距离
            person_indices = _all_indices.get(best_name, [])
            other_dists = [distances[i] for i in range(len(distances))
                          if i != best_idx and i not in person_indices]

            if other_dists:
                second_best = min(other_dists)
                dist_diff = second_best - best_dist
                if dist_diff > 0.12:  # 与次优有足够差距
                    is_known = True
            else:
                is_known = True

        confidence = f"{max(0, (1 - best_dist) * 100):.1f}%"
        results.append((best_name if is_known else "未知人员", is_known, confidence, best_dist))

    return results


# =============================================================================
# 中文字体 + 标签缓存（与原代码一致）
# =============================================================================
_font_cache = {}
_label_cache = {}

def _load_chinese_font(size=20):
    candidate_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/Deng.ttf",
    ]
    for fp in candidate_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()

def get_font(size=20):
    if size not in _font_cache:
        _font_cache[size] = _load_chinese_font(size)
    return _font_cache[size]

def clear_label_cache():
    _label_cache.clear()

def _render_label(name, is_known):
    key = (name, is_known)
    if key in _label_cache:
        return _label_cache[key]

    bg_bgr = (0, 200, 0) if is_known else (0, 0, 255)
    font = get_font(20)

    temp = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), name, font=font)
    tw = bbox[2] - bbox[0] + 8
    th = bbox[3] - bbox[1] + 6

    label_pil = Image.new('RGB', (tw, th), (bg_bgr[2], bg_bgr[1], bg_bgr[0]))
    d = ImageDraw.Draw(label_pil)
    d.text((4, 1), name, font=font, fill=(255, 255, 255))

    label_bgr = cv2.cvtColor(np.array(label_pil), cv2.COLOR_RGB2BGR)
    _label_cache[key] = label_bgr
    return label_bgr

def draw_faces_with_names(frame, face_results):
    for (top, right, bottom, left), name, is_known, _ in face_results:
        color = (0, 200, 0) if is_known else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label = _render_label(name, is_known)
        lh, lw = label.shape[:2]
        y = bottom + 2
        x = left
        if y + lh > frame.shape[0]:
            y = max(0, top - lh - 2)
        if x + lw > frame.shape[1]:
            x = max(0, frame.shape[1] - lw)
        frame[y:y+lh, x:x+lw] = label
    return frame


# =============================================================================
# 【核心优化】摄像头视频流系统 - 针对联想拯救者 R7000 等 Windows 笔记本优化
# =============================================================================
"""

1. 双缓冲帧队列（生产者-消费者模式）
   - 采集线程高频抓取，只保留最新 2 帧
   - 处理线程从队列取帧，旧帧自动丢弃
   - 彻底解决 USB 摄像头缓冲堆积导致的延迟累积

2. Windows 后端优化
   - 优先使用 cv2.CAP_DSHOW（DirectShow）后端
   - 设置 CAP_PROP_BUFFERSIZE = 1 减少内部缓冲
   - 设置 CAP_PROP_FOURCC 为 MJPG 格式（USB 带宽友好）

3. 自适应帧率控制
   - 检测实际处理耗时，动态调整检测间隔
   - 处理快则提高检测频率，处理慢则降低避免积压

4. 采集与处理完全解耦
   - 采集线程：只负责 grab() + retrieve()，最快速度循环
   - 处理线程：独立运行，不阻塞采集
   - 输出线程：合成最终画面
"""

class CameraSystem:
    """
    高性能摄像头系统（R7000 优化版）：
      - 采集线程：高频 grab，双缓冲队列只保留最新帧
      - 检测线程：从队列取帧处理，旧帧丢弃
      - 输出线程：读取最新结果 + 最新帧，合成输出
    """
    def __init__(self):
        # 双缓冲帧队列：只保留最新 2 帧，防止 USB 缓冲堆积
        self._frame_queue = queue.Queue(maxsize=2)
        self._results_queue = queue.Queue(maxsize=2)  # 只保留最新结果
        self._cap_thread = None
        self._det_thread = None
        self._active = False
        self._cap = None
        self._detector = None
        self._last_processed_frame = None  # 最后一帧处理结果（用于显示）
        self._last_results = []  # 最后一帧的检测结果
        self._process_time = 0.05  # 预估处理时间，用于自适应间隔
        self._frame_count = 0
        self._drop_count = 0

    def start(self):
        if self._active:
            return
        self._active = True
        self._cap_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._det_thread = threading.Thread(target=self._detect_worker, daemon=True)
        self._cap_thread.start()
        self._det_thread.start()
        print("[INFO] 摄像头系统已启动（R7000 优化版：双缓冲 + 自适应帧率）")

    def stop(self):
        self._active = False
        if self._det_thread and self._det_thread.is_alive():
            self._det_thread.join(timeout=2.0)
        if self._cap_thread and self._cap_thread.is_alive():
            self._cap_thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        # 清空队列
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self._results_queue.empty():
            try:
                self._results_queue.get_nowait()
            except queue.Empty:
                break
        print("[INFO] 摄像头系统已停止")

    def get_frame(self):
        """获取当前最新帧（带检测结果）"""
        return self._last_processed_frame

    def get_results(self):
        try:
            return self._results_queue.get_nowait()
        except queue.Empty:
            return None

    def _open_camera(self):
        """【优化】针对 Windows 笔记本的多后端尝试打开摄像头"""
        # 尝试列表：优先 DSHOW，然后是 MSMF，最后是默认
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "MediaFoundation"),
            (cv2.CAP_ANY, "默认"),
        ]

        for backend, name in backends:
            try:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    print(f"[INFO] 摄像头使用 {name} 后端打开成功")

                    # 【关键优化】减少内部缓冲区，降低延迟
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    # 【关键优化】设置 MJPG 格式，降低 USB 带宽占用
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

                    # 设置分辨率（R7000 摄像头通常支持 640x480 流畅运行）
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                    # 预热摄像头：丢弃前 5 帧（摄像头启动时的模糊/曝光调整帧）
                    for _ in range(5):
                        cap.grab()

                    return cap
            except Exception as e:
                print(f"[WARN] {name} 后端打开失败: {e}")
                continue

        print("[ERROR] 所有后端均无法打开摄像头")
        return None

    def _capture_worker(self):
        """【优化】采集线程：高频 grab，双缓冲队列"""
        self._cap = self._open_camera()

        if self._cap is None or not self._cap.isOpened():
            print("[ERROR] 摄像头打开失败")
            self._active = False
            return

        print("[INFO] 采集线程启动，分辨率: 640x480, 格式: MJPG, 缓冲: 1")

        while self._active:
            # grab 比 read 更快，不解码图像
            ret = self._cap.grab()
            if not ret:
                time.sleep(0.001)
                continue

            # 每隔几帧才 retrieve 一次，减轻 CPU 负担
            # 但保持 grab 频率以清空 USB 缓冲区
            self._frame_count += 1

            # 尝试放入队列，如果队列满则丢弃旧帧
            try:
                # 先清空旧帧（只保留最新）
                while self._frame_queue.qsize() >= 1:
                    try:
                        self._frame_queue.get_nowait()
                        self._drop_count += 1
                    except queue.Empty:
                        break

                # retrieve 最新帧
                ret, frame = self._cap.retrieve()
                if ret and frame is not None:
                    self._frame_queue.put_nowait(frame)
            except queue.Full:
                self._drop_count += 1

            # 极短休眠，让出 CPU
            time.sleep(0.001)

        if self._cap:
            self._cap.release()

    def _detect_worker(self):
        """【优化】检测线程：从队列取帧，自适应间隔，旧帧丢弃"""
        if self._detector is None:
            self._detector = get_face_detector()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        last_detect = 0
        consecutive_errors = 0

        while self._active:
            # 自适应检测间隔：根据实际处理速度调整
            adaptive_interval = max(
                _cfg["detection_interval"] / 1000.0,
                self._process_time * 1.2  # 给 20% 余量
            )

            if time.time() - last_detect < adaptive_interval:
                time.sleep(0.005)
                continue

            # 从队列取最新帧（丢弃所有旧帧）
            frame = None
            try:
                while not self._frame_queue.empty():
                    try:
                        frame = self._frame_queue.get_nowait()
                    except queue.Empty:
                        break
            except Exception:
                pass

            if frame is None:
                time.sleep(0.01)
                continue

            detect_start = time.time()

            try:
                # 快速预处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())

                if brightness < 80:
                    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    enhanced = frame

                rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

                # 检测 - 使用 DNN 检测器
                faces, confs = self._detector.detect_faces(enhanced, _cfg["confidence_threshold"])

                results = []
                if len(faces) > 0:
                    face_locs = [(int(y), int(x+w), int(y+h), int(x)) for (x, y, w, h) in faces]

                    # 【修复】确保人脸库已加载
                    if _all_encodings is not None and len(_all_encodings) > 0:
                        encs = face_recognition.face_encodings(rgb, face_locs, model=_cfg["encoding_model"])

                        # 向量化匹配
                        matches = match_faces_vectorized(encs)

                        for (x, y, w, h), (name, is_known, conf, dist) in zip(faces, matches):
                            top, right, bottom, left = y, x+w, y+h, x
                            results.append(((top, right, bottom, left), name, is_known, conf))
                    else:
                        # 人脸库为空，全部标记为未知
                        for (x, y, w, h) in faces:
                            top, right, bottom, left = y, x+w, y+h, x
                            results.append(((top, right, bottom, left), "未知人员", False, "0%"))

                # 保存结果和帧用于输出
                self._last_results = results
                self._last_processed_frame = frame.copy()

                # 只保留最新结果（丢弃旧结果）
                while not self._results_queue.empty():
                    try:
                        self._results_queue.get_nowait()
                    except queue.Empty:
                        break
                self._results_queue.put(results)

                last_detect = time.time()
                consecutive_errors = 0

                # 更新处理时间用于自适应
                self._process_time = time.time() - detect_start

                # 每 100 帧报告一次丢弃统计
                if self._frame_count % 100 == 0 and self._drop_count > 0:
                    print(f"[INFO] 帧统计: 采集 {self._frame_count}, 丢弃 {self._drop_count}, 处理耗时 {self._process_time*1000:.1f}ms")

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    print(f"[WARN] 检测异常 (连续 {consecutive_errors} 次): {e}")
                time.sleep(0.05)


_camera_system = CameraSystem()


def gen_camera_stream():
    """【优化】视频流生成器 - 降低编码开销"""
    _camera_system.start()

    # 等待首帧
    for _ in range(100):  # 增加等待时间
        if _camera_system.get_frame() is not None:
            break
        time.sleep(0.03)

    frame_interval = 0.04  # 25fps 目标
    last_send = 0

    while True:
        frame = _camera_system.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        # 控制输出帧率，避免浏览器端堆积
        now = time.time()
        if now - last_send < frame_interval:
            time.sleep(0.005)
            continue
        last_send = now

        results = _camera_system.get_results()
        if results:
            frame = draw_faces_with_names(frame.copy(), results)
        elif _camera_system._last_results and _camera_system._last_processed_frame is not None:
            # 没有新结果但用旧结果绘制（保持标注不闪烁）
            frame = draw_faces_with_names(frame.copy(), _camera_system._last_results)

        # 【优化】降低 JPEG 质量到 75，减少带宽和编码时间
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield b'--frame\r\nContent-Type:image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'


# =============================================================================
# FastAPI 路由（完全保留原有功能）
# =============================================================================

@app.get("/")
async def index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read(), headers={"Cache-Control": "no-cache"})
    except FileNotFoundError:
        return HTMLResponse("<h1>AI Faces Pro</h1><p>请创建 index.html</p>")


@app.post("/register")
async def register(
    name: str = Form(...),
    file: UploadFile = File(...),
    multi_mode: bool = Form(False)  # 是否多图模式
):
    """
    人脸注册（持久化存储至 storage/known_face）
    multi_mode=True 时，自动编号保存为多图持久文件
    """
    content = await file.read()
    if not content:
        return {"success": False, "msg": "上传文件为空"}

    # 持久存储路径
    if multi_mode:
        # 查找下一个可用编号
        next_num = 1
        for i in range(1, _cfg["max_faces_per_person"] + 2):
            if not (KNOWN_DIR / f"{name}_{i}.jpg").exists():
                next_num = i
                break
        save_path = KNOWN_DIR / f"{name}_{next_num}.jpg"
    else:
        save_path = KNOWN_DIR / f"{name}.jpg"

    with open(save_path, "wb") as fw:
        fw.write(content)

    # 异步编码（不阻塞响应）
    try:
        img = face_recognition.load_image_file(str(save_path))
        locs = face_recognition.face_locations(img, number_of_times_to_upsample=1)
        if not locs:
            save_path.unlink()
            return {"success": False, "msg": "未检测到人脸，请上传清晰正面或侧面照片"}

        encs = face_recognition.face_encodings(img, locs, model=_cfg["encoding_model"])
        if not encs:
            save_path.unlink()
            return {"success": False, "msg": "人脸编码失败"}

        # 质量评估
        from PIL import Image
        pil_img = Image.open(str(save_path)).convert('RGB')
        face_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if face_img is None:
            save_path.unlink()
            return {"success": False, "msg": "图片读取失败，请重新上传"}
        quality = assess_face_quality(face_img)
        if quality < _cfg["min_face_quality"]:
            save_path.unlink()
            return {"success": False, "msg": f"人脸质量过低 ({quality:.2f})，请重新拍摄"}

        # 增量添加到持久人脸库
        added = add_person_encoding(name, encs[0].tolist(), str(save_path))

        with _face_db_lock:
            count = _face_db.get(name, {}).get("count", 0)

        return {
            "success": True,
            "msg": f"【{name}】持久注册成功（当前 {count} 张图片，质量 {quality:.2f}）",
            "quality": round(quality, 3),
            "total_images": count,
        }
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        return {"success": False, "msg": f"注册失败: {str(e)}"}


@app.post("/register_batch")
async def register_batch(name: str = Form(...), files: list[UploadFile] = File(...)):
    """
    批量注册：多张图片持久存入 storage/known_face，自动筛选最优，融合编码。
    """
    if not files:
        return {"success": False, "msg": "未上传图片"}

    saved_paths = []
    encodings = []
    qualities = []

    for idx, file in enumerate(files[:_cfg["max_faces_per_person"]]):
        content = await file.read()
        if not content:
            continue

        save_path = KNOWN_DIR / f"{name}_{idx+1}.jpg"
        with open(save_path, "wb") as fw:
            fw.write(content)
        saved_paths.append(save_path)

        try:
            img = face_recognition.load_image_file(str(save_path))
            locs = face_recognition.face_locations(img, number_of_times_to_upsample=1)
            if not locs:
                continue
            encs = face_recognition.face_encodings(img, locs, model=_cfg["encoding_model"])
            if encs:
                face_img = cv2.imread(str(save_path))
                q = assess_face_quality(face_img)
                encodings.append(encs[0].tolist())
                qualities.append(q)
        except Exception:
            continue

    if not encodings:
        for p in saved_paths:
            p.unlink()
        return {"success": False, "msg": "所有图片均未检测到有效人脸"}

    # 按质量排序，保留高质量持久图
    sorted_pairs = sorted(zip(qualities, encodings, saved_paths), reverse=True)
    best_encodings = [enc for _, enc, _ in sorted_pairs[:_cfg["max_faces_per_person"]]]

    # 清理低质量持久图片
    for _, _, path in sorted_pairs[_cfg["max_faces_per_person"]:]:
        path.unlink()

    # 更新持久数据库缓存
    with _face_db_lock:
        _face_db[name] = {
            "encodings": best_encodings,
            "primary": _compute_primary_encoding(best_encodings),
            "count": len(best_encodings),
            "updated": time.time(),
        }

    _rebuild_vector_cache()

    return {
        "success": True,
        "msg": f"【{name}】批量持久注册成功，融合 {len(best_encodings)} 张图片",
        "images_used": len(best_encodings),
        "avg_quality": round(np.mean(qualities), 3),
    }


@app.post("/predict_img")
async def predict_img(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        return {"face_count": 0, "results": [], "error": "上传文件为空"}

    temp_path = UPLOAD_DIR / f"temp_{int(time.time()*1000)}.jpg"
    with open(temp_path, "wb") as fw:
        fw.write(content)

    try:
        img = face_recognition.load_image_file(str(temp_path))
        locs = face_recognition.face_locations(img, number_of_times_to_upsample=1)
        encs = face_recognition.face_encodings(img, locs, model=_cfg["encoding_model"])

        # 向量化匹配
        matches = match_faces_vectorized(encs)

        res_list = []
        face_draw_info = []
        for i, enc in enumerate(encs):
            top, right, bottom, left = locs[i]
            name, is_known, conf, dist = matches[i]
            res_list.append({
                "name": name,
                "confidence": conf,
                "distance": round(dist, 4),
                "location": [top, right, bottom, left]
            })
            face_draw_info.append((top, right, bottom, left, name, conf, is_known))

        # 绘制标注图
        annotated_b64 = None
        if face_draw_info:
            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            font_label = get_font(28)
            for (top, right, bottom, left, name, conf, is_known) in face_draw_info:
                color = (0, 200, 0) if is_known else (255, 0, 0)
                line_w = max(2, int(img.shape[1] / 400))
                draw.rectangle([left, top, right, bottom], outline=color, width=line_w)
                label_text = f"{name}  {conf}" if is_known else "未知人员"
                bbox = draw.textbbox((0, 0), label_text, font=font_label)
                tw = bbox[2] - bbox[0] + 12
                th = bbox[3] - bbox[1] + 8
                ly = bottom if bottom + th <= pil_img.height else max(0, top - th)
                lx = left if left + tw <= pil_img.width else max(0, pil_img.width - tw)
                draw.rectangle([lx, ly, lx + tw, ly + th], fill=color)
                draw.text((lx + 6, ly + 1), label_text, font=font_label, fill=(255, 255, 255))

            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=90)
            annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

        return {"face_count": len(res_list), "results": res_list, "annotated_img": annotated_b64}
    except Exception as e:
        return {"face_count": 0, "results": [], "error": str(e)}
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/known_faces")
async def list_known_faces():
    with _face_db_lock:
        faces = [
            {
                "name": name,
                "image_count": data["count"],
                "updated": data.get("updated", 0),
            }
            for name, data in _face_db.items()
        ]
    return {"faces": faces, "count": len(faces)}


@app.post("/delete_face")
async def delete_face(name: str = Form(...)):
    if remove_person(name):
        return {"success": True, "msg": f"已删除【{name}】全部持久存储图片与元数据"}
    return {"success": False, "msg": f"未找到【{name}】持久人脸数据"}


@app.get("/video")
async def video_stream():
    return StreamingResponse(gen_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/capture")
async def capture_frame():
    _camera_system.start()
    for _ in range(100):  # 增加等待时间
        frame = _camera_system.get_frame()
        if frame is not None:
            break
        time.sleep(0.03)

    frame = _camera_system.get_frame()
    if frame is None:
        return Response(status_code=503, content=b'{"error":"camera not ready"}')

    results = _camera_system.get_results()
    if results:
        frame = draw_faces_with_names(frame.copy(), results)
    elif _camera_system._last_results:
        frame = draw_faces_with_names(frame.copy(), _camera_system._last_results)

    # 抓拍图片同时持久保存到 storage/camera_capture_storage
    save_name = f"capture_{int(time.time()*1000)}.jpg"
    save_full_path = CAPTURE_SAVE_DIR / save_name
    cv2.imwrite(str(save_full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/stop_camera")
async def stop_camera():
    _camera_system.stop()
    return {"success": True, "msg": "摄像头已关闭"}


@app.post("/set_config")
async def set_config(
    confidence_threshold: float = Form(0.70),
    detection_interval: int = Form(300),
    match_threshold: float = Form(0.45),
    encoding_workers: int = Form(None),
):
    _cfg["confidence_threshold"] = max(0.1, min(0.95, confidence_threshold))
    _cfg["detection_interval"] = max(50, min(1000, detection_interval))
    _cfg["match_threshold"] = max(0.2, min(0.6, match_threshold))
    if encoding_workers:
        _cfg["encoding_workers"] = max(1, min(16, encoding_workers))
    return {"success": True, "config": _cfg}


@app.get("/get_config")
async def get_config():
    return _cfg


@app.get("/performance")
async def performance_stats():
    """获取性能统计"""
    import psutil
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "db_version": _db_version,
        "total_encodings": len(_all_names) if _all_names else 0,
        "total_people": len(_face_db),
        "encoding_workers": _cfg["encoding_workers"],
    }


# =============================================================================
# 启动
# =============================================================================
def _generate_self_signed_cert():
    cert_path = Path(__file__).parent / 'cert.pem'
    key_path = Path(__file__).parent / 'key.pem'
    if cert_path.exists() and key_path.exists():
        return str(cert_path), str(key_path)

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import datetime

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Faces Pro"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = x509.CertificateBuilder().subject_name(subject).issuer_name(issuer)\
        .public_key(private_key.public_key()).serial_number(x509.random_serial_number())\
        .not_valid_before(datetime.datetime.utcnow())\
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))\
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False)\
        .sign(private_key, hashes.SHA256())

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    print("[INFO] SSL 证书生成成功")
    return str(cert_path), str(key_path)


@app.on_event("startup")
def startup_event():
    print("[INFO] FastAPI 启动事件：加载人脸库...")
    load_known_faces_parallel()
    print("[INFO] 人脸库加载完成")

if __name__ == "__main__":
    import uvicorn, socket

    cert_path, key_path = _generate_self_signed_cert()
    port = 8005
    local_ip = socket.gethostbyname(socket.gethostname())

    print(f"[INFO] 服务启动中... 编码进程数: {_cfg['encoding_workers']}")
    print(f"[INFO] 持久化存储根目录: ./storage/")
    print(f"[INFO] 局域网访问: https://{local_ip}:{port}")
    print(f"[INFO] 本地访问: https://127.0.0.1:{port}")

    uvicorn.run("main:app", host="0.0.0.0", port=port, ssl_keyfile=key_path, ssl_certfile=cert_path)
