import requests
import json
import time
import subprocess

BASE_URL = "http://localhost:5000"

def test_api():
    print("测试API服务...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"健康检查: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        response = requests.get(f"{BASE_URL}/model_info", timeout=5)
        print(f"\n模型信息查询: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
        data = {
            "features": [5.1, 3.5, 1.4, 0.2]
        }
        response = requests.post(f"{BASE_URL}/predict", json=data, timeout=5)
        print(f"\n预测接口: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务，请确保服务已启动")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    print("请先在另一个终端运行: python api_service.py")
    print("然后运行: python test_api.py")
