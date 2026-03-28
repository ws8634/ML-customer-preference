from typing import Any, Dict
from datetime import datetime
import os
from src.core import Command, ExperimentManager, ExperimentConfig, ExperimentResult
from src.trainer import ModelTrainer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


class TrainCommand(Command):
    """执行模型训练命令"""

    def __init__(self, console: Console):
        """
        初始化命令

        Args:
            console: Rich控制台对象
        """
        self.console = console

    def execute(self, **kwargs) -> Any:
        """
        执行训练命令

        Args:
            **kwargs: 训练参数
                - experiment_name: 实验名称
                - model_type: 模型类型
                - dataset_path: 数据集路径
                - hyperparameters: 超参数字典
                - features: 特征列表
                - target: 目标列名
                - path: 项目根目录路径

        Returns:
            训练结果
        """
        root_dir = kwargs.get("path", ".")
        manager = ExperimentManager(root_dir)

        if not manager.is_initialized():
            self.console.print("[red]错误: 项目未初始化，请先运行 init 命令[/red]")
            return None

        project_config = manager.load_config()

        experiment_name = kwargs.get("experiment_name", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model_type = kwargs.get("model_type", project_config.get("default_model_type", "random_forest"))
        dataset_path = kwargs.get("dataset_path", "data/customers.csv")
        hyperparameters = kwargs.get("hyperparameters", project_config.get("default_hyperparameters", {}))
        features = kwargs.get("features", project_config.get("features", []))
        target = kwargs.get("target", project_config.get("target", "preference"))

        if not os.path.exists(dataset_path):
            self.console.print(f"[red]错误: 数据集不存在: {dataset_path}[/red]")
            return None

        config = ExperimentConfig(
            experiment_name=experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters,
            dataset_path=dataset_path,
            features=features,
            target=target
        )

        experiment_id = manager.generate_experiment_id(config)
        model_path = os.path.join(manager.models_dir, f"{experiment_id}.pkl")

        self.console.print(f"[cyan]开始训练实验: {experiment_name}[/cyan]")
        self.console.print(f"  实验ID: {experiment_id}")
        self.console.print(f"  模型类型: {model_type}")
        self.console.print(f"  数据集: {dataset_path}")
        self.console.print(f"  超参数: {hyperparameters}")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("训练中...", total=None)

                started_at = datetime.now().isoformat()
                trainer = ModelTrainer(project_config)
                metrics = trainer.train(dataset_path, features, target, model_type, hyperparameters)
                trainer.save_model(model_path)
                completed_at = datetime.now().isoformat()

                progress.update(task, completed=True)

            result = ExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=metrics,
                model_path=model_path,
                status="completed",
                started_at=started_at,
                completed_at=completed_at
            )

            manager.save_experiment(result)

            self.console.print("[green]✓ 训练完成![/green]")
            self.console.print("  评估指标:")
            for metric, value in metrics.items():
                self.console.print(f"    {metric}: {value:.4f}")
            self.console.print(f"  模型保存至: {model_path}")

            return result

        except Exception as e:
            self.console.print(f"[red]训练失败: {str(e)}[/red]")
            return None
