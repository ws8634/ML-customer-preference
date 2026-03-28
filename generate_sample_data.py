#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os


def generate_sample_data(output_path: str = "data/customers.csv"):
    """生成示例顾客数据集"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.randint(30000, 150000, n_samples),
        "purchase_history": np.random.randint(1, 50, n_samples),
        "browsing_time": np.random.uniform(5, 120, n_samples),
        "items_viewed": np.random.randint(1, 30, n_samples),
        "preference": np.random.randint(0, 2, n_samples)
    }

    df = pd.DataFrame(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"示例数据集已生成: {output_path}")
    print(f"数据集大小: {len(df)} 行 x {len(df.columns)} 列")
    print(f"前5行:\n{df.head()}")


if __name__ == "__main__":
    generate_sample_data()
