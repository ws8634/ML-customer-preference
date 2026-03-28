from typing import Any, List, Tuple
from src.core import Command, ExperimentManager, ExperimentResult
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


class CompareCommand(Command):
    """对比实验差异命令"""

    def __init__(self, console: Console):
        """
        初始化命令

        Args:
            console: Rich控制台对象
        """
        self.console = console

    def execute(self, **kwargs) -> Any:
        """
        执行对比命令

        Args:
            **kwargs: 命令参数
                - experiment_ids: 要对比的实验ID列表
                - path: 项目根目录路径

        Returns:
            对比结果
        """
        root_dir = kwargs.get("path", ".")
        manager = ExperimentManager(root_dir)

        if not manager.is_initialized():
            self.console.print("[red]错误: 项目未初始化，请先运行 init 命令[/red]")
            return None

        experiment_ids = kwargs.get("experiment_ids", [])

        if len(experiment_ids) < 2:
            self.console.print("[red]错误: 请至少提供两个实验ID进行对比[/red]")
            return None

        experiments = []
        for exp_id in experiment_ids:
            exp = manager.load_experiment(exp_id)
            if exp:
                experiments.append(exp)
            else:
                self.console.print(f"[yellow]警告: 实验 {exp_id} 不存在，已跳过[/yellow]")

        if len(experiments) < 2:
            self.console.print("[red]错误: 有效实验数量不足2个，无法对比[/red]")
            return None

        self._display_comparison(experiments)

        return experiments

    def _display_comparison(self, experiments: List[ExperimentResult]) -> None:
        """
        显示实验对比结果

        Args:
            experiments: 实验列表
        """
        self.console.print(Panel.fit("[cyan]实验对比分析[/cyan]"))

        self._display_config_comparison(experiments)
        self._display_metrics_comparison(experiments)
        self._display_hyperparameters_comparison(experiments)

    def _display_config_comparison(self, experiments: List[ExperimentResult]) -> None:
        """
        显示配置对比

        Args:
            experiments: 实验列表
        """
        table = Table(title="配置对比")
        table.add_column("属性", style="cyan", width=20)
        for exp in experiments:
            table.add_column(exp.experiment_id[:15], style="magenta", width=20)

        config_fields = [
            ("实验名称", lambda e: e.config.experiment_name),
            ("模型类型", lambda e: e.config.model_type),
            ("数据集", lambda e: e.config.dataset_path),
            ("目标列", lambda e: e.config.target),
            ("状态", lambda e: e.status),
            ("开始时间", lambda e: e.started_at[:19])
        ]

        for field_name, field_getter in config_fields:
            row = [field_name]
            for exp in experiments:
                row.append(field_getter(exp))
            table.add_row(*row)

        self.console.print(table)

    def _display_metrics_comparison(self, experiments: List[ExperimentResult]) -> None:
        """
        显示指标对比

        Args:
            experiments: 实验列表
        """
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())

        table = Table(title="指标对比")
        table.add_column("指标", style="cyan", width=15)
        for exp in experiments:
            table.add_column(exp.experiment_id[:15], style="magenta", width=15)

        for metric in sorted(all_metrics):
            row = [metric]
            values = []
            for exp in experiments:
                value = exp.metrics.get(metric, 0.0)
                values.append(value)
                row.append(f"{value:.4f}")

            row = self._highlight_best_value(row, values)
            table.add_row(*row)

        self.console.print(table)

    def _display_hyperparameters_comparison(self, experiments: List[ExperimentResult]) -> None:
        """
        显示超参数对比

        Args:
            experiments: 实验列表
        """
        all_params = set()
        for exp in experiments:
            all_params.update(exp.config.hyperparameters.keys())

        if not all_params:
            return

        table = Table(title="超参数对比")
        table.add_column("超参数", style="cyan", width=20)
        for exp in experiments:
            table.add_column(exp.experiment_id[:15], style="magenta", width=20)

        for param in sorted(all_params):
            row = [param]
            values = []
            for exp in experiments:
                value = exp.config.hyperparameters.get(param, "-")
                values.append(str(value))
                row.append(str(value))

            table.add_row(*row)

        self.console.print(table)

    def _highlight_best_value(self, row: List[str], values: List[float]) -> List[str]:
        """
        高亮最佳值（假设越大越好）

        Args:
            row: 原始行数据
            values: 数值列表

        Returns:
            高亮后的行数据
        """
        max_idx = values.index(max(values))
        new_row = [row[0]]
        for i, val in enumerate(row[1:]):
            if i == max_idx:
                new_row.append(f"[green]{val}[/green]")
            else:
                new_row.append(val)
        return new_row
