"""
实验对比命令模块
负责对比多个实验的差异
"""
from typing import Dict, Any, List, Tuple
from rich.table import Table
from rich.box import SIMPLE
from rich.panel import Panel
from rich.columns import Columns
from experiment_cli.commands.base import BaseCommand
from experiment_cli.core.experiment_manager import ExperimentManager
from experiment_cli.core.models import Experiment


class CompareCommand(BaseCommand):
    """实验对比命令类"""
    
    def execute(self, experiment_ids: List[str],
                tracking_uri: str = "./experiments") -> Dict[str, Any]:
        """
        执行实验对比命令
        
        Args:
            experiment_ids: 要对比的实验ID列表
            tracking_uri: 实验追踪路径
            
        Returns:
            Dict[str, Any]: 对比结果信息
        """
        if len(experiment_ids) < 2:
            self.print_error("需要至少两个实验ID进行对比")
            return {"success": False, "error": "需要至少两个实验ID"}
        
        self.print_highlight(f"对比 {len(experiment_ids)} 个实验...")
        
        experiment_manager = ExperimentManager(tracking_uri)
        experiments, comparison_data = experiment_manager.compare_experiments(experiment_ids)
        
        if not experiments:
            self.print_error("没有找到有效的实验")
            return {"success": False, "error": "没有找到有效的实验"}
        
        # 打印对比结果
        self._print_comparison(experiments, comparison_data)
        
        return {
            "success": True,
            "compared_experiments": len(experiments),
            "comparison_data": comparison_data
        }
    
    def _print_comparison(self, experiments: List[Experiment], comparison_data: Dict[str, List[Any]]) -> None:
        """打印实验对比结果"""
        self.console.print("\n")
        
        # 创建基本信息表格
        info_table = Table(box=SIMPLE, show_header=True, header_style="bold blue")
        info_table.add_column("指标", style="cyan")
        for exp in experiments:
            info_table.add_column(exp.experiment_id[:8], style="magenta")
        
        # 添加基本信息行
        info_table.add_row("名称", *[exp.name for exp in experiments])
        info_table.add_row("状态", *[self._format_status(exp.status) for exp in experiments])
        info_table.add_row("持续时间", *[self._format_duration(exp.duration_seconds) for exp in experiments])
        info_table.add_row("创建时间", *[exp.created_at.strftime("%Y-%m-%d") for exp in experiments])
        
        self.console.print(Panel(info_table, title="基本信息对比", border_style="blue"))
        self.console.print("\n")
        
        # 创建指标对比表格
        metrics_table = Table(box=SIMPLE, show_header=True, header_style="bold blue")
        metrics_table.add_column("指标", style="cyan")
        for exp in experiments:
            metrics_table.add_column(exp.experiment_id[:8], style="magenta")
        
        # 添加指标行
        metrics_table.add_row("Accuracy", *[self._format_metric(exp.metrics.accuracy) for exp in experiments])
        metrics_table.add_row("Precision", *[self._format_metric(exp.metrics.precision) for exp in experiments])
        metrics_table.add_row("Recall", *[self._format_metric(exp.metrics.recall) for exp in experiments])
        metrics_table.add_row("F1 Score", *[self._format_metric(exp.metrics.f1_score) for exp in experiments])
        if any(exp.metrics.auc_roc is not None for exp in experiments):
            metrics_table.add_row("AUC-ROC", *[self._format_metric(exp.metrics.auc_roc) for exp in experiments])
        
        self.console.print(Panel(metrics_table, title="性能指标对比", border_style="green"))
        self.console.print("\n")
        
        # 找出最佳实验
        best_exp = self._find_best_experiment(experiments)
        if best_exp:
            self.print_success(f"\n🏆 最佳实验: {best_exp.name} (ID: {best_exp.experiment_id[:8]})")
            self.print_info(f"   Accuracy: {best_exp.metrics.accuracy:.4f}")
            self.print_info(f"   F1 Score: {best_exp.metrics.f1_score:.4f}")
    
    def _format_status(self, status: str) -> str:
        """格式化状态显示"""
        if status == "completed":
            return "[green]✅ 完成[/green]"
        elif status == "running":
            return "[yellow]⏳ 运行中[/yellow]"
        elif status == "failed":
            return "[red]❌ 失败[/red]"
        else:
            return f"[dim]{status}[/dim]"
    
    def _format_duration(self, seconds: float) -> str:
        """格式化持续时间"""
        if seconds is None:
            return "-"
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def _format_metric(self, value: float) -> str:
        """格式化指标值"""
        if value is None:
            return "-"
        return f"{value:.4f}"
    
    def _find_best_experiment(self, experiments: List[Experiment]) -> Experiment:
        """找出最佳实验（按F1分数）"""
        valid_experiments = [exp for exp in experiments if exp.metrics.f1_score is not None]
        if not valid_experiments:
            return None
        return max(valid_experiments, key=lambda x: x.metrics.f1_score)
