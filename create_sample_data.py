#!/usr/bin/env python3
"""
创建示例训练数据脚本
用于测试顾客偏好预测实验工具
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification


def create_sample_data(output_path: str = "data/train.csv", n_samples: int = 1000):
    """创建示例训练数据"""
    # 创建分类数据集
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # 创建DataFrame
    feature_names = [f"feature_{i+1}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df["preference"] = y
    
    # 添加一些有意义的列名
    df = df.rename(columns={
        "feature_1": "age",
        "feature_2": "income",
        "feature_3": "spending_score",
        "feature_4": "visit_frequency",
        "feature_5": "average_order_value"
    })
    
    # 保存到CSV
    df.to_csv(output_path, index=False)
    print(f"示例数据已创建: {output_path}")
    print(f"数据形状: {df.shape}")
    print(f"特征列: {list(df.columns[:-1])}")
    print(f"目标列: preference")
    
    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    create_sample_data()
