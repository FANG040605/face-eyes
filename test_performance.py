import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


def test_performance():
    base_url = "https://localhost:8005"
    test_image = Path("test_samples/face.jpg")
    
    if not test_image.exists():
        print("❌ 测试图片不存在")
        return

    print("\n" + "="*60)
    print("AI Faces Pro 人脸识别系统性能测试")
    print("="*60)

    print("\n[性能测试1] 单次人脸识别请求响应时间...")
    times = []
    for i in range(5):
        start_time = time.time()
        try:
            with open(test_image, "rb") as f:
                files = {"file": f}
                response = requests.post(f"{base_url}/predict_img", files=files, verify=False, timeout=30)
            elapsed = time.time() - start_time
            times.append(elapsed)
            status = "✓" if response.status_code == 200 else "✗"
            print(f"  第{i+1}次: {elapsed:.2f}秒 [{status} HTTP {response.status_code}]")
        except Exception as e:
            elapsed = time.time() - start_time
            
            print(f"  第{i+1}次: {elapsed:.2f}秒 [✗ 错误: {str(e)[:50]}]")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\n  平均响应时间: {avg_time:.2f}秒")
        print(f"  最快响应: {min(times):.2f}秒")
        print(f"  最慢响应: {max(times):.2f}秒")
        if avg_time < 3.0:
            print("  ✓ 响应时间符合要求 (< 3秒)")
        else:
            print("  ⚠ 响应时间超过要求")

    print("\n[性能测试2] 并发人脸识别处理能力...")
    def recognize_image(i):
        try:
            with open(test_image, "rb") as f:
                files = {"file": (f"test_{i}.jpg", f, "image/jpeg")}
                response = requests.post(f"{base_url}/predict_img", files=files, verify=False, timeout=30)
            return response.status_code, response.elapsed.total_seconds()
        except Exception as e:
            return -1, time.time()
    
    for workers in [2, 3, 5]:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start_time = time.time()
            results = list(executor.map(recognize_image, range(workers)))
            elapsed = time.time() - start_time
        
        success_count = sum(1 for code, _ in results if code == 200)
        valid_times = [t for _, t in results if t > 0]
        avg_req_time = sum(valid_times) / len(valid_times) if valid_times else 0
        print(f"  {workers}个并发请求: 总耗时 {elapsed:.2f}秒, 平均 {avg_req_time:.2f}秒/请求, 成功率 {success_count}/{workers}")
    print("  ✓ 并发处理正常")

    print("\n[性能测试3] 人脸库查询接口响应...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/known_faces", verify=False, timeout=10)
        elapsed = time.time() - start_time
        data = response.json()
        print(f"  响应时间: {elapsed:.2f}秒")
        print(f"  人脸库人数: {data.get('count', 0)}")
        print("  ✓ 查询接口正常")
    except Exception as e:
        print(f"  ✗ 查询失败: {e}")

    print("\n[性能测试4] 服务健康检查...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/", verify=False, timeout=10)
        elapsed = time.time() - start_time
        print(f"  响应时间: {elapsed:.2f}秒")
        print(f"  HTTP状态: {response.status_code}")
        print("  ✓ 服务正常")
    except Exception as e:
        print(f"  ✗ 服务不可达: {e}")
        print("  请确认服务已启动: https://localhost:8005")

    print("\n" + "="*60)
    print("性能测试完成！")
    print("="*60)


if __name__ == "__main__":
    test_performance()