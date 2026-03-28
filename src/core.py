from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
import hashlib
import shutil


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    experiment_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    dataset_path: str
    features: List[str]
    target: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float]
    model_path: str
    status: str
    started_at: str
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "model_path": self.model_path,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class Command(ABC):
    """命令模式基类"""

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行命令"""
        pass


class ExperimentManager:
    """实验管理器，负责实验的持久化和查询"""

    EXPERIMENTS_DIR = "experiments"
    CONFIG_FILE = "config.json"
    RESULTS_FILE = "results.json"
    MODELS_DIR = "models"

    def __init__(self, root_dir: str = "."):
        """
        初始化实验管理器

        Args:
            root_dir: 根目录路径
        """
        self.root_dir = root_dir
        self.experiments_dir = os.path.join(root_dir, self.EXPERIMENTS_DIR)
        self.models_dir = os.path.join(root_dir, self.MODELS_DIR)

    def initialize_project(self) -> None:
        """初始化项目目录结构"""
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        default_config = {
            "default_model_type": "random_forest",
            "default_hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "features": ["age", "income", "purchase_history"],
            "target": "preference"
        }

        config_path = os.path.join(self.root_dir, self.CONFIG_FILE)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

    def is_initialized(self) -> bool:
        """检查项目是否已初始化"""
        return os.path.exists(os.path.join(self.root_dir, self.CONFIG_FILE))

    def load_config(self) -> Dict[str, Any]:
        """加载项目配置"""
        config_path = os.path.join(self.root_dir, self.CONFIG_FILE)
        with open(config_path, 'r') as f:
            return json.load(f)

    def generate_experiment_id(self, config: ExperimentConfig) -> str:
        """生成实验ID"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{hash_suffix}"

    def save_experiment(self, result: ExperimentResult) -> None:
        """保存实验结果"""
        exp_dir = os.path.join(self.experiments_dir, result.experiment_id)
        os.makedirs(exp_dir, exist_ok=True)

        result_path = os.path.join(exp_dir, self.RESULTS_FILE)
        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """加载实验结果"""
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        result_path = os.path.join(exp_dir, self.RESULTS_FILE)

        if not os.path.exists(result_path):
            return None

        with open(result_path, 'r') as f:
            data = json.load(f)

        config_data = data["config"]
        config = ExperimentConfig(
            experiment_name=config_data["experiment_name"],
            model_type=config_data["model_type"],
            hyperparameters=config_data["hyperparameters"],
            dataset_path=config_data["dataset_path"],
            features=config_data["features"],
            target=config_data["target"],
            created_at=config_data["created_at"]
        )

        return ExperimentResult(
            experiment_id=data["experiment_id"],
            config=config,
            metrics=data["metrics"],
            model_path=data["model_path"],
            status=data["status"],
            started_at=data["started_at"],
            completed_at=data["completed_at"]
        )

    def list_experiments(self) -> List[ExperimentResult]:
        """列出所有实验"""
        experiments = []
        if not os.path.exists(self.experiments_dir):
            return experiments

        for exp_id in os.listdir(self.experiments_dir):
            exp = self.load_experiment(exp_id)
            if exp:
                experiments.append(exp)

        return experiments

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验"""
        exp_dir = os.path.join(self.experiments_dir, experiment_id)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            return True
        return False
