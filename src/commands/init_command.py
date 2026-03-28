from typing import Any
from src.core import Command, ExperimentManager
from rich.console import Console


class InitCommand(Command):
    """初始化实验目录结构命令"""

    def __init__(self, console: Console):
        """
        初始化命令

        Args:
            console: Rich控制台对象
        """
        self.console = console

    def execute(self, **kwargs) -> Any:
        """
        执行初始化命令

        Args:
            **kwargs: 可选参数
                - path: 项目根目录路径

        Returns:
            初始化结果
        """
        root_dir = kwargs.get("path", ".")
        manager = ExperimentManager(root_dir)

        if manager.is_initialized():
            self.console.print(f"[yellow]项目已经初始化: {root_dir}[/yellow]")
            return False

        manager.initialize_project()
        self.console.print(f"[green]✓ 项目初始化成功[/green]")
        self.console.print(f"  根目录: {root_dir}")
        self.console.print(f"  实验目录: {manager.experiments_dir}")
        self.console.print(f"  模型目录: {manager.models_dir}")
        self.console.print(f"  配置文件: {manager.CONFIG_FILE}")
        return True
