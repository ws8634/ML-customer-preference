from typing import Any, List
from datetime import datetime, timedelta
from src.core import Command, ExperimentManager, ExperimentResult
from rich.console import Console
from rich.prompt import Confirm


class CleanCommand(Command):
    """清理过期实验命令"""

    def __init__(self, console: Console):
        """
        初始化命令

        Args:
            console: Rich控制台对象
        """
        self.console = console

    def execute(self, **kwargs) -> Any:
        """
        执行清理命令

        Args:
            **kwargs: 命令参数
                - keep_top: 保留顶部模型数量
                - days: 保留N天内的实验
                - metric: 排序指标
                - dry_run: 试运行模式
                - force: 强制删除（不确认）
                - path: 项目根目录路径

        Returns:
            清理结果
        """
        root_dir = kwargs.get("path", ".")
        manager = ExperimentManager(root_dir)

        if not manager.is_initialized():
            self.console.print("[red]错误: 项目未初始化，请先运行 init 命令[/red]")
            return []

        keep_top = kwargs.get("keep_top", 5)
        days = kwargs.get("days", None)
        metric = kwargs.get("metric", "accuracy")
        dry_run = kwargs.get("dry_run", False)
        force = kwargs.get("force", False)

        experiments = manager.list_experiments()

        if not experiments:
            self.console.print("[yellow]暂无实验记录[/yellow]")
            return []

        experiments_to_delete = self._get_experiments_to_delete(experiments, keep_top, days, metric)

        if not experiments_to_delete:
            self.console.print("[green]没有需要清理的实验[/green]")
            return []

        self._display_clean_plan(experiments_to_delete, keep_top, days, metric)

        if dry_run:
            self.console.print("[yellow]试运行模式：不会实际删除任何实验[/yellow]")
            return experiments_to_delete

        if not force:
            confirm = Confirm.ask(f"[red]确认要删除 {len(experiments_to_delete)} 个实验吗？[/red]")
            if not confirm:
                self.console.print("[yellow]已取消清理[/yellow]")
                return []

        deleted = []
        for exp in experiments_to_delete:
            if manager.delete_experiment(exp.experiment_id):
                deleted.append(exp)
                self.console.print(f"[red]已删除: {exp.experiment_id}[/red]")

        self.console.print(f"[green]✓ 清理完成，共删除 {len(deleted)} 个实验[/green]")

        return deleted

    def _get_experiments_to_delete(self, experiments: List[ExperimentResult], keep_top: int, days: int, metric: str) -> List[ExperimentResult]:
        """
        获取需要删除的实验列表

        Args:
            experiments: 所有实验
            keep_top: 保留顶部数量
            days: 保留天数
            metric: 排序指标

        Returns:
            需要删除的实验列表
        """
        to_delete = []

        if days is not None:
            cutoff_date = datetime.now() - timedelta(days=days)
            for exp in experiments:
                exp_date = datetime.fromisoformat(exp.started_at)
                if exp_date < cutoff_date:
                    to_delete.append(exp)

        experiments.sort(key=lambda x: x.metrics.get(metric, 0), reverse=True)
        top_experiments = experiments[:keep_top]
        top_ids = {exp.experiment_id for exp in top_experiments}

        for exp in experiments:
            if exp.experiment_id not in top_ids and exp not in to_delete:
                to_delete.append(exp)

        return to_delete

    def _display_clean_plan(self, experiments: List[ExperimentResult], keep_top: int, days: int, metric: str) -> None:
        """
        显示清理计划

        Args:
            experiments: 要删除的实验列表
            keep_top: 保留顶部数量
            days: 保留天数
            metric: 排序指标
        """
        self.console.print(f"[cyan]清理计划:[/cyan]")
        self.console.print(f"  保留指标 {metric} 排名前 {keep_top} 的实验")
        if days is not None:
            self.console.print(f"  保留 {days} 天内的实验")
        self.console.print(f"  将删除 {len(experiments)} 个实验:")

        for exp in experiments:
            self.console.print(f"    - {exp.experiment_id} ({exp.config.experiment_name}, {metric}: {exp.metrics.get(metric, 0):.4f})")
