"""
实验管理核心模块
负责实验的创建、执行、追踪和管理
"""
import os
import json
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from experiment_cli.core.models import Experiment, ExperimentConfig, Metrics


class ExperimentManager:
    """实验管理器类"""
    
    def __init__(self, tracking_uri: str = "./experiments"):
        """
        初始化实验管理器
        
        Args:
            tracking_uri: 实验数据存储路径
        """
        self.tracking_uri = Path(tracking_uri)
        self._ensure_directory()
    
    def _ensure_directory(self) -> None:
        """确保实验目录存在"""
        self.tracking_uri.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(self, config: ExperimentConfig) -> Experiment:
        """
        创建新实验
        
        Args:
            config: 实验配置
            
        Returns:
            Experiment: 创建的实验对象
        """
        experiment = Experiment(
            name=config.name,
            description=config.description,
            config=config,
            tags=config.tags
        )
        
        # 创建实验目录
        experiment_dir = self.tracking_uri / experiment.experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # 保存实验元数据
        self._save_experiment(experiment)
        
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        根据ID获取实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            Optional[Experiment]: 实验对象，如果不存在则返回None
        """
        experiment_dir = self.tracking_uri / experiment_id
        metadata_path = experiment_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        return self._load_experiment(metadata_path)
    
    def list_experiments(self, status: Optional[str] = None, 
                        tags: Optional[List[str]] = None) -> List[Experiment]:
        """
        列出所有实验
        
        Args:
            status: 可选的状态过滤器
            tags: 可选的标签过滤器
            
        Returns:
            List[Experiment]: 实验列表
        """
        experiments = []
        
        for experiment_dir in self.tracking_uri.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            metadata_path = experiment_dir / "metadata.json"
            if not metadata_path.exists():
                continue
                
            try:
                experiment = self._load_experiment(metadata_path)
                
                # 应用过滤器
                if status and experiment.status != status:
                    continue
                if tags and not any(tag in experiment.tags for tag in tags):
                    continue
                    
                experiments.append(experiment)
            except Exception:
                continue
        
        # 按创建时间倒序排序
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        return experiments
    
    def update_experiment(self, experiment: Experiment) -> None:
        """
        更新实验信息
        
        Args:
            experiment: 要更新的实验对象
        """
        experiment.updated_at = datetime.now()
        self._save_experiment(experiment)
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """
        删除实验
        
        Args:
            experiment_id: 要删除的实验ID
            
        Returns:
            bool: 是否成功删除
        """
        experiment_dir = self.tracking_uri / experiment_id
        
        if not experiment_dir.exists():
            return False
            
        shutil.rmtree(experiment_dir)
        return True
    
    def get_best_experiments(self, metric: str = "accuracy", 
                           top_n: int = 5,
                           status: str = "completed") -> List[Experiment]:
        """
        获取最佳实验（按指定指标排序）
        
        Args:
            metric: 排序指标名称
            top_n: 返回前N个实验
            status: 实验状态过滤器
            
        Returns:
            List[Experiment]: 排序后的实验列表
        """
        experiments = self.list_experiments(status=status)
        
        # 过滤出有指定指标的实验
        valid_experiments = []
        for exp in experiments:
            metric_value = getattr(exp.metrics, metric, None)
            if metric_value is not None:
                valid_experiments.append(exp)
        
        # 按指标降序排序
        valid_experiments.sort(
            key=lambda x: getattr(x.metrics, metric),
            reverse=True
        )
        
        return valid_experiments[:top_n]
    
    def compare_experiments(self, experiment_ids: List[str]) -> Tuple[List[Experiment], Dict[str, List[Any]]]:
        """
        比较多个实验
        
        Args:
            experiment_ids: 要比较的实验ID列表
            
        Returns:
            Tuple[List[Experiment], Dict[str, List[Any]]]: 实验对象列表和比较数据
        """
        experiments = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                experiments.append(exp)
        
        # 构建比较数据
        comparison_data = {
            "experiment_id": [exp.experiment_id for exp in experiments],
            "name": [exp.name for exp in experiments],
            "status": [exp.status for exp in experiments],
            "accuracy": [exp.metrics.accuracy for exp in experiments],
            "precision": [exp.metrics.precision for exp in experiments],
            "recall": [exp.metrics.recall for exp in experiments],
            "f1_score": [exp.metrics.f1_score for exp in experiments],
            "duration": [exp.duration_seconds for exp in experiments],
            "created_at": [exp.created_at.isoformat() for exp in experiments]
        }
        
        return experiments, comparison_data
    
    def clean_experiments(self, keep_top_n: int = 10, 
                         metric: str = "accuracy",
                         dry_run: bool = False) -> Tuple[int, List[str]]:
        """
        清理过期实验，保留头部模型
        
        Args:
            keep_top_n: 保留的最佳实验数量
            metric: 用于排序的指标
            dry_run: 是否仅模拟删除
            
        Returns:
            Tuple[int, List[str]]: 删除的实验数量和删除的实验ID列表
        """
        best_experiments = self.get_best_experiments(metric=metric, top_n=keep_top_n)
        best_ids = {exp.experiment_id for exp in best_experiments}
        
        all_experiments = self.list_experiments()
        to_delete = [exp for exp in all_experiments if exp.experiment_id not in best_ids]
        
        deleted_count = 0
        deleted_ids = []
        
        for exp in to_delete:
            if not dry_run:
                if self.delete_experiment(exp.experiment_id):
                    deleted_count += 1
                    deleted_ids.append(exp.experiment_id)
            else:
                deleted_count += 1
                deleted_ids.append(exp.experiment_id)
        
        return deleted_count, deleted_ids
    
    def _save_experiment(self, experiment: Experiment) -> None:
        """保存实验到文件"""
        experiment_dir = self.tracking_uri / experiment.experiment_id
        metadata_path = experiment_dir / "metadata.json"
        
        # 转换为可序列化的字典
        experiment_dict = {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "description": experiment.description,
            "status": experiment.status,
            "config": self._config_to_dict(experiment.config) if experiment.config else None,
            "metrics": self._metrics_to_dict(experiment.metrics),
            "artifacts": experiment.artifacts,
            "tags": experiment.tags,
            "created_at": experiment.created_at.isoformat(),
            "updated_at": experiment.updated_at.isoformat(),
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "ended_at": experiment.ended_at.isoformat() if experiment.ended_at else None,
            "duration_seconds": experiment.duration_seconds
        }
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(experiment_dict, f, indent=2, ensure_ascii=False)
    
    def _load_experiment(self, metadata_path: Path) -> Experiment:
        """从文件加载实验"""
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 解析配置
        config = None
        if data["config"]:
            from experiment_cli.core.models import ModelConfig, DataConfig, ExperimentConfig
            model_config = ModelConfig(**data["config"]["model"])
            data_config = DataConfig(**data["config"]["data"])
            config = ExperimentConfig(
                name=data["config"]["name"],
                description=data["config"]["description"],
                model=model_config,
                data=data_config,
                tracking_uri=data["config"]["tracking_uri"],
                tags=data["config"]["tags"]
            )
        
        # 解析指标
        metrics = Metrics(**data["metrics"])
        
        # 解析日期时间
        created_at = datetime.fromisoformat(data["created_at"])
        updated_at = datetime.fromisoformat(data["updated_at"])
        started_at = datetime.fromisoformat(data["started_at"]) if data["started_at"] else None
        ended_at = datetime.fromisoformat(data["ended_at"]) if data["ended_at"] else None
        
        return Experiment(
            experiment_id=data["experiment_id"],
            name=data["name"],
            description=data["description"],
            status=data["status"],
            config=config,
            metrics=metrics,
            artifacts=data["artifacts"],
            tags=data["tags"],
            created_at=created_at,
            updated_at=updated_at,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=data["duration_seconds"]
        )
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
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
    
    def _metrics_to_dict(self, metrics: Metrics) -> Dict[str, Any]:
        """将指标对象转换为字典"""
        return {
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "auc_roc": metrics.auc_roc,
            "custom_metrics": metrics.custom_metrics
        }
