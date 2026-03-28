from typing import Any, List
from datetime import datetime
from src.core import Command, ExperimentManager, ExperimentResult
from rich.console import Console
from rich.table import Table


class ListCommand(Command):
    """列出历史实验命令"""

    def __init__(self, console: Console):
        """
        初始化命令

        Args:
            console: Rich控制台对象
        """
        self.console = console

    def execute(self, **kwargs) -> Any:
        """
        执行列出实验命令

        Args:
            **kwargs: 命令参数
                - sort_by: 排序方式 (time/metric)
                - metric: 排序指标 (当sort_by=metric时)
                - ascending: 是否升序
                - path: 项目根目录路径

        Returns:
            实验列表
        """
        root_dir = kwargs.get("path", ".")
        manager = ExperimentManager(root_dir)

        if not manager.is_initialized():
            self.console.print("[red]错误: 项目未初始化，请先运行 init 命令[/red]")
            return []

        experiments = manager.list_experiments()

        if not experiments:
            self.console.print("[yellow]暂无实验记录[/yellow]")
            return []

        sort_by = kwargs.get("sort_by", "time")
        metric_name = kwargs.get("metric", "accuracy")
        ascending = kwargs.get("ascending", False)

        experiments = self._sort_experiments(experiments, sort_by, metric_name, ascending)

        self._display_experiments(experiments, metric_name)

        return experiments

    def _sort_experiments(self, experiments: List[ExperimentResult], sort_by: str, metric_name: str, ascending: bool) -> List[ExperimentResult]:
        """
        对实验进行排序

        Args:
            experiments: 实验列表
            sort_by: 排序方式
            metric_name: 指标名称
            ascending: 是否升序

        Returns:
            排序后的实验列表
        """
        if sort_by == "time":
            experiments.sort(key=lambda x: x.started_at, reverse=not ascending)
        elif sort_by == "metric":
            experiments.sort(key=lambda x: x.metrics.get(metric_name, 0), reverse=not ascending)

        return experiments

    def _display_experiments(self, experiments: List[ExperimentResult], metric_name: str) -> None:
        """
        显示实验列表

        Args:
            experiments: 实验列表
            metric_name: 主要指标名称
        """
        table = Table(title=f"实验历史 (共 {len(experiments)} 个实验)")
        table.add_column("序号", style="cyan", width=4)
        table.add_column("实验ID", style="magenta", width=30)
        table.add_column("实验名称", style="green", width=20)
        table.add_column("模型类型", style="yellow", width=15)
        table.add_column(f"{metric_name}", style="blue", width=10)
        table.add_column("状态", style="white", width=10)
        table.add_column("开始时间", style="dim", width=20)

        for idx, exp in enumerate(experiments, 1):
            metric_value = exp.metrics.get(metric_name, 0)
            status_style = "green" if exp.status == "completed" else "red"

            table.add_row(
                str(idx),
                exp.experiment_id,
                exp.config.experiment_name,
                exp.config.model_type,
                f"{metric_value:.4f}",
                f"[{status_style}]{exp.status}[/{status_style}]",
                exp.started_at[:19]
            )

        self.console.print(table)
