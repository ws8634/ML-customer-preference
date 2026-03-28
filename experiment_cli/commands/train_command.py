"""
训练命令模块
负责执行模型训练并自动记录参数和指标
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from experiment_cli.commands.base import BaseCommand
from experiment_cli.core.config_manager import ConfigManager
from experiment_cli.core.experiment_manager import ExperimentManager
from experiment_cli.core.models import Metrics, ExperimentConfig


class TrainCommand(BaseCommand):
    """训练命令类"""
    
    def execute(self, config_path: str = "experiment_config.yaml",
                tracking_uri: str = "./experiments") -> Dict[str, Any]:
        """
        执行训练命令
        
        Args:
            config_path: 配置文件路径
            tracking_uri: 实验追踪路径
            
        Returns:
            Dict[str, Any]: 训练结果信息
        """
        self.print_highlight("开始模型训练实验")
        
        # 加载配置
        try:
            config_manager = ConfigManager(config_path)
            config = config_manager.load_config()
        except Exception as e:
            self.print_error(f"加载配置失败: {e}")
            return {"success": False, "error": str(e)}
        
        # 创建实验
        experiment_manager = ExperimentManager(tracking_uri)
        experiment = experiment_manager.create_experiment(config)
        
        self.print_info(f"实验ID: {experiment.experiment_id}")
        self.print_info(f"实验名称: {experiment.name}")
        
        try:
            # 标记实验开始
            experiment.start()
            experiment_manager.update_experiment(experiment)
            
            # 加载数据
            self.print_info("加载训练数据...")
            X_train, X_test, y_train, y_test = self._load_data(config)
            
            # 训练模型
            self.print_info("训练模型...")
            model = self._train_model(config, X_train, y_train)
            
            # 评估模型
            self.print_info("评估模型...")
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # 保存模型
            self.print_info("保存模型...")
            model_path = self._save_model(model, experiment.experiment_id, config)
            
            # 标记实验完成
            experiment.complete(metrics, artifacts=[model_path])
            experiment_manager.update_experiment(experiment)
            
            self.print_success("训练完成!")
            self._print_metrics(metrics)
            
            return {
                "success": True,
                "experiment_id": experiment.experiment_id,
                "metrics": self._metrics_to_dict(metrics),
                "model_path": model_path
            }
            
        except Exception as e:
            experiment.fail(str(e))
            experiment_manager.update_experiment(experiment)
            self.print_error(f"训练失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _load_data(self, config: ExperimentConfig) -> tuple:
        """加载并预处理数据"""
        data_config = config.data
        
        # 加载训练数据
        train_df = pd.read_csv(data_config.train_path)
        
        # 分离特征和目标
        X = train_df.drop(columns=[data_config.target_column])
        y = train_df[data_config.target_column]
        
        # 如果有测试数据则使用，否则分割训练数据
        if data_config.test_path and os.path.exists(data_config.test_path):
            test_df = pd.read_csv(data_config.test_path)
            X_test = test_df.drop(columns=[data_config.target_column])
            y_test = test_df[data_config.target_column]
            return X, X_test, y, y_test
        else:
            return train_test_split(
                X, y, 
                test_size=data_config.test_size,
                random_state=config.model.random_state
            )
    
    def _train_model(self, config: ExperimentConfig, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """训练模型"""
        model_type = config.model.model_type
        params = config.model.parameters
        
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier(**params, random_state=config.model.random_state)
        elif model_type == "LogisticRegression":
            model = LogisticRegression(**params, random_state=config.model.random_state)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def _evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Metrics:
        """评估模型性能"""
        y_pred = model.predict(X_test)
        
        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 计算AUC-ROC（如果是二分类问题）
        auc_roc = None
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] == 2:
                    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        except Exception:
            pass
        
        return Metrics(
            accuracy=float(accuracy),
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            auc_roc=float(auc_roc) if auc_roc is not None else None
        )
    
    def _save_model(self, model: Any, experiment_id: str, config: ExperimentConfig) -> str:
        """保存训练好的模型"""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"model_{experiment_id}.pkl"
        joblib.dump(model, model_path)
        
        return str(model_path)
    
    def _print_metrics(self, metrics: Metrics) -> None:
        """打印评估指标"""
        self.print_highlight("\n模型评估指标:")
        self.console.print(f"  Accuracy:  [metric]{metrics.accuracy:.4f}[/metric]")
        self.console.print(f"  Precision: [metric]{metrics.precision:.4f}[/metric]")
        self.console.print(f"  Recall:    [metric]{metrics.recall:.4f}[/metric]")
        self.console.print(f"  F1 Score:  [metric]{metrics.f1_score:.4f}[/metric]")
        if metrics.auc_roc is not None:
            self.console.print(f"  AUC-ROC:   [metric]{metrics.auc_roc:.4f}[/metric]")
    
    def _metrics_to_dict(self, metrics: Metrics) -> Dict[str, float]:
        """将指标转换为字典"""
        result = {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score
        }
        if metrics.auc_roc is not None:
            result["auc_roc"] = metrics.auc_roc
        return result
