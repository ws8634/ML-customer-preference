"""
清理实验命令模块
负责清理过期实验数据，保留头部模型
"""
from typing import Dict, Any, List
from rich.prompt import Confirm
from experiment_cli.commands.base import BaseCommand
from experiment_cli.core.experiment_manager import ExperimentManager


class CleanCommand(BaseCommand):
    """清理实验命令类"""
    
    def execute(self, keep_top_n: int = 10,
                metric: str = "accuracy",
                dry_run: bool = False,
                force: bool = False,
                tracking_uri: str = "./experiments") -> Dict[str, Any]:
        """
        执行清理命令
        
        Args:
            keep_top_n: 保留的最佳实验数量
            metric: 用于排序的指标
            dry_run: 是否仅模拟删除
            force: 是否强制删除（不确认）
            tracking_uri: 实验追踪路径
            
        Returns:
            Dict[str, Any]: 清理结果信息
        """
        experiment_manager = ExperimentManager(tracking_uri)
        
        # 获取所有实验
        all_experiments = experiment_manager.list_experiments()
        total_count = len(all_experiments)
        
        if total_count == 0:
            self.print_info("没有实验需要清理")
            return {"success": True, "deleted_count": 0, "deleted_ids": []}
        
        self.print_highlight(f"找到 {total_count} 个实验")
        self.print_info(f"将保留最佳的 {keep_top_n} 个实验 (按 {metric} 排序)")
        
        # 执行清理
        if not force and not dry_run:
            confirm = Confirm.ask(f"确定要删除 {total_count - keep_top_n} 个实验吗？")
            if not confirm:
                self.print_info("已取消清理")
                return {"success": True, "deleted_count": 0, "deleted_ids": []}
        
        deleted_count, deleted_ids = experiment_manager.clean_experiments(
            keep_top_n=keep_top_n,
            metric=metric,
            dry_run=dry_run
        )
        
        if dry_run:
            self.print_info(f"模拟删除: 将删除 {deleted_count} 个实验")
            for exp_id in deleted_ids:
                self.console.print(f"  - {exp_id}")
        else:
            self.print_success(f"已删除 {deleted_count} 个实验")
            if deleted_count > 0:
                self.print_info(f"保留了 {total_count - deleted_count} 个最佳实验")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "deleted_ids": deleted_ids,
            "dry_run": dry_run
        }
