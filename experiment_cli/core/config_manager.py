"""
配置管理模块
负责加载、验证和保存实验配置
"""
import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from experiment_cli.core.models import ExperimentConfig, ModelConfig, DataConfig


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: str = "experiment_config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
    
    def load_config(self) -> ExperimentConfig:
        """
        从YAML文件加载配置
        
        Returns:
            ExperimentConfig: 实验配置对象
            
        Raises:
            FileNotFoundError: 如果配置文件不存在
            ValueError: 如果配置格式无效
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        return self._parse_config(config_dict)
    
    def save_config(self, config: ExperimentConfig) -> None:
        """
        保存配置到YAML文件
        
        Args:
            config: 要保存的实验配置对象
        """
        config_dict = self._serialize_config(config)
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def create_default_config(self, name: str = "customer_preference_experiment",
                            description: str = "顾客偏好预测实验") -> ExperimentConfig:
        """
        创建默认配置
        
        Args:
            name: 实验名称
            description: 实验描述
            
        Returns:
            ExperimentConfig: 默认实验配置对象
        """
        model_config = ModelConfig(
            model_type="RandomForestClassifier",
            parameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2
            }
        )
        
        data_config = DataConfig(
            train_path="data/train.csv",
            test_path="data/test.csv",
            target_column="preference",
            test_size=0.2
        )
        
        return ExperimentConfig(
            name=name,
            description=description,
            model=model_config,
            data=data_config,
            tags=["customer_preference", "classification"]
        )
    
    def _parse_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """
        解析配置字典为ExperimentConfig对象
        
        Args:
            config_dict: 配置字典
            
        Returns:
            ExperimentConfig: 解析后的实验配置对象
            
        Raises:
            ValueError: 如果配置格式无效
        """
        try:
            model_dict = config_dict["model"]
            model_config = ModelConfig(
                model_type=model_dict["model_type"],
                parameters=model_dict["parameters"],
                random_state=model_dict.get("random_state", 42)
            )
            
            data_dict = config_dict["data"]
            data_config = DataConfig(
                train_path=data_dict["train_path"],
                test_path=data_dict.get("test_path"),
                target_column=data_dict.get("target_column", "preference"),
                test_size=data_dict.get("test_size", 0.2)
            )
            
            return ExperimentConfig(
                name=config_dict["name"],
                description=config_dict.get("description", ""),
                model=model_config,
                data=data_config,
                tracking_uri=config_dict.get("tracking_uri", "./experiments"),
                tags=config_dict.get("tags", [])
            )
        except KeyError as e:
            raise ValueError(f"配置缺少必要字段: {e}")
    
    def _serialize_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        序列化ExperimentConfig对象为字典
        
        Args:
            config: 实验配置对象
            
        Returns:
            Dict[str, Any]: 序列化后的配置字典
        """
        return {
            "name": config.name,
            "description": config.description,
            "model": {
                "model_type": config.model.model_type,
                "parameters": config.model.parameters,
                "random_state": config.model.random_state
            },
            "data": {
                "train_path": config.data.train_path,
                "test_path": config.data.test_path,
                "target_column": config.data.target_column,
                "test_size": config.data.test_size
            },
            "tracking_uri": config.tracking_uri,
            "tags": config.tags
        }
