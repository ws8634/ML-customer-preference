"""
数据模型定义模块
包含实验、配置、指标等核心数据结构
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class ModelConfig:
    """模型配置数据类"""
    model_type: str
    parameters: Dict[str, Any]
    random_state: int = 42


@dataclass
class DataConfig:
    """数据配置数据类"""
    train_path: str
    test_path: Optional[str] = None
    target_column: str = "preference"
    test_size: float = 0.2


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    name: str
    description: str
    model: ModelConfig
    data: DataConfig
    tracking_uri: str = "./experiments"
    tags: List[str] = field(default_factory=list)


@dataclass
class Metrics:
    """评估指标数据类"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class Experiment:
    """实验数据类"""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: str = "created"  # created, running, completed, failed
    config: Optional[ExperimentConfig] = None
    metrics: Metrics = field(default_factory=Metrics)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def start(self) -> None:
        """标记实验开始"""
        self.status = "running"
        self.started_at = datetime.now()
        self.updated_at = self.started_at

    def complete(self, metrics: Metrics, artifacts: Optional[List[str]] = None) -> None:
        """标记实验完成"""
        self.status = "completed"
        self.metrics = metrics
        if artifacts:
            self.artifacts = artifacts
        self.ended_at = datetime.now()
        self.updated_at = self.ended_at
        if self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()

    def fail(self, error_message: str) -> None:
        """标记实验失败"""
        self.status = "failed"
        self.tags.append(f"error:{error_message}")
        self.ended_at = datetime.now()
        self.updated_at = self.ended_at
        if self.started_at:
            self.duration_seconds = (self.ended_at - self.started_at).total_seconds()
