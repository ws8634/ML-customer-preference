"""
列出实验命令模块
负责列出历史实验并支持按时间或指标排序
"""
from typing import Dict, Any, Optional, List
from rich.table import Table
from rich.box import SIMPLE
from experiment_cli.commands.base import BaseCommand
from experiment_cli.core.experiment_manager import ExperimentManager
from experiment_cli.core.models import Experiment


class ListCommand(BaseCommand):
    """列出实验命令类"""
    
    def execute(self, tracking_uri: str = "./experiments",
                status: Optional[str] = None,
                sort_by: str = "created_at",
                limit: Optional[int] = None) -> Dict[str, Any]:
        """
        执行列出实验命令
        
        Args:
            tracking_uri: 实验追踪路径
            status: 实验状态过滤器
            sort_by: 排序字段 (created_at, updated_at, accuracy, precision, recall, f1_score)
            limit: 限制显示的实验数量
            
        Returns:
            Dict[str, Any]: 实验列表信息
        """
        experiment_manager = ExperimentManager(tracking_uri)
        experiments = experiment_manager.list_experiments(status=status)
        
        # 排序
        experiments = self._sort_experiments(experiments, sort_by)
        
        # 限制数量
        if limit and limit > 0:
            experiments = experiments[:limit]
        
        # 打印结果
        self._print_experiments_table(experiments, sort_by)
        
        return {
            "success": True,
            "experiment_count": len(experiments),
            "experiments": [self._experiment_to_dict(exp) for exp in experiments]
        }
    
    def _sort_experiments(self, experiments: List[Experiment], sort_by: str) -> List[Experiment]:
        """根据指定字段排序实验"""
        if sort_by == "created_at":
            experiments.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "updated_at":
            experiments.sort(key=lambda x: x.updated_at, reverse=True)
        elif sort_by in ["accuracy", "precision", "recall", "f1_score", "auc_roc"]:
            # 按指标降序排序，没有指标的排最后
            def get_metric(exp: Experiment) -> float:
                value = getattr(exp.metrics, sort_by, None)
                return value if value is not None else -1.0
            
            experiments.sort(key=get_metric, reverse=True)
        
        return experiments
    
    def _print_experiments_table(self, experiments: List[Experiment], sort_by: str) -> None:
        """打印实验列表表格"""
        if not experiments:
            self.print_warning("没有找到实验记录")
            return
        
        table = Table(box=SIMPLE, show_header=True, header_style="bold blue")
        
        table.add_column("实验ID", style="dim", width=36)
        table.add_column("名称", style="cyan")
        table.add_column("状态", style="green")
        table.add_column("Accuracy", style="magenta")
        table.add_column("F1 Score", style="magenta")
        table.add_column("持续时间", style="yellow")
        table.add_column("创建时间", style="dim")
        
        for exp in experiments:
            # 格式化指标
            accuracy = f"{exp.metrics.accuracy:.4f}" if exp.metrics.accuracy is not None else "-"
            f1 = f"{exp.metrics.f1_score:.4f}" if exp.metrics.f1_score is not None else "-"
            
            # 格式化持续时间
            duration = self._format_duration(exp.duration_seconds)
            
            # 格式化时间
            created_at = exp.created_at.strftime("%Y-%m-%d %H:%M:%S")
            
            # 状态颜色
            status_style = "green" if exp.status == "completed" else "yellow" if exp.status == "running" else "red"
            
            table.add_row(
                exp.experiment_id,
                exp.name,
                f"[{status_style}]{exp.status}[/{status_style}]",
                accuracy,
                f1,
                duration,
                created_at
            )
        
        self.console.print("\n")
        self.print_highlight(f"实验列表 (按{sort_by}排序):")
        self.console.print(table)
        self.console.print("\n")
    
    def _format_duration(self, seconds: Optional[float]) -> str:
        """格式化持续时间"""
        if seconds is None:
            return "-"
        
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            return f"{seconds/60:.1f}分钟"
        else:
            return f"{seconds/3600:.1f}小时"
    
    def _experiment_to_dict(self, experiment: Experiment) -> Dict[str, Any]:
        """将实验对象转换为字典"""
        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status,
            "accuracy": experiment.metrics.accuracy,
            "precision": experiment.metrics.precision,
            "recall": experiment.metrics.recall,
            "f1_score": experiment.metrics.f1_score,
            "auc_roc": experiment.metrics.auc_roc,
            "duration_seconds": experiment.duration_seconds,
            "created_at": experiment.created_at.isoformat(),
            "tags": experiment.tags
        }
