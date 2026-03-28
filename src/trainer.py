import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import os


class ModelTrainer:
    """模型训练器，负责训练和评估模型"""

    SUPPORTED_MODELS = {
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "logistic_regression": LogisticRegression
    }

    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器

        Args:
            config: 训练配置
        """
        self.config = config
        self.model = None

    def load_data(self, dataset_path: str, features: list, target: str) -> pd.DataFrame:
        """
        加载数据集

        Args:
            dataset_path: 数据集路径
            features: 特征列列表
            target: 目标列名

        Returns:
            加载的数据集
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"数据集不存在: {dataset_path}")

        df = pd.read_csv(dataset_path)

        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"数据集缺少必要列: {missing_cols}")

        return df

    def split_data(self, df: pd.DataFrame, features: list, target: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        划分训练集和测试集

        Args:
            df: 数据集
            features: 特征列列表
            target: 目标列名
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df[features]
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def create_model(self, model_type: str, hyperparameters: Dict[str, Any]) -> Any:
        """
        创建模型实例

        Args:
            model_type: 模型类型
            hyperparameters: 超参数

        Returns:
            模型实例
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: {list(self.SUPPORTED_MODELS.keys())}")

        model_class = self.SUPPORTED_MODELS[model_type]
        return model_class(**hyperparameters)

    def train(self, dataset_path: str, features: list, target: str, model_type: str, hyperparameters: Dict[str, Any]) -> Dict[str, float]:
        """
        训练模型并返回评估指标

        Args:
            dataset_path: 数据集路径
            features: 特征列列表
            target: 目标列名
            model_type: 模型类型
            hyperparameters: 超参数

        Returns:
            评估指标字典
        """
        df = self.load_data(dataset_path, features, target)
        X_train, X_test, y_train, y_test = self.split_data(df, features, target)

        self.model = self.create_model(model_type, hyperparameters)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        }

        if y_pred_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
            except:
                metrics["roc_auc"] = 0.0

        return metrics

    def save_model(self, model_path: str) -> None:
        """
        保存模型到文件

        Args:
            model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型，请先训练")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
