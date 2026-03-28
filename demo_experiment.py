#!/usr/bin/env python3
"""
实验工具完整演示脚本
展示从初始化到训练、查看结果的完整流程
"""
import os
import sys
import subprocess
import time


def run_command(cmd):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"执行命令: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("错误输出:", result.stderr)
    
    print(f"\n命令完成，返回码: {result.returncode}")
    return result.returncode


def main():
    """主演示函数"""
    print("🚀 顾客偏好预测实验工具演示")
    print("="*60)
    
    # 1. 创建示例数据
    print("\n1. 创建示例训练数据...")
    if run_command("python create_sample_data.py") != 0:
        print("❌ 创建数据失败")
        return
    
    # 2. 初始化实验
    print("\n2. 初始化实验目录...")
    if run_command("python main.py init --name 'demo_customer_preference' --description '顾客偏好预测演示实验'") != 0:
        print("❌ 初始化失败")
        return
    
    # 3. 查看生成的配置文件
    print("\n3. 查看配置文件内容:")
    config_file = "experiment_config.yaml"
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            print(f.read())
    
    # 4. 开始训练
    print("\n4. 开始模型训练...")
    if run_command("python main.py train") != 0:
        print("❌ 训练失败")
        return
    
    # 5. 查看实验列表
    print("\n5. 查看实验列表...")
    if run_command("python main.py list --sort-by accuracy") != 0:
        print("❌ 查看列表失败")
        return
    
    # 6. 演示完成
    print("\n✅ 演示完成!")
    print("\n📌 可用的后续命令:")
    print("  - python main.py list --status completed  # 查看已完成的实验")
    print("  - python main.py compare <exp_id1> <exp_id2>  # 对比实验")
    print("  - python main.py clean --keep-top-n 5  # 清理实验")
    print("\n📂 生成的文件:")
    print("  - experiment_config.yaml  # 实验配置")
    print("  - data/train.csv  # 训练数据")
    print("  - experiments/  # 实验追踪数据")
    print("  - models/  # 训练好的模型")


if __name__ == "__main__":
    main()
