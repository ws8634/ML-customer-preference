"""
CLI入口模块
使用Click定义命令行接口
"""
import click
from rich.console import Console
from experiment_cli.commands.init_command import InitCommand
from experiment_cli.commands.train_command import TrainCommand
from experiment_cli.commands.list_command import ListCommand
from experiment_cli.commands.compare_command import CompareCommand
from experiment_cli.commands.clean_command import CleanCommand


console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="experiment-cli")
def cli():
    """
    顾客偏好预测实验命令行工具
    
    提供实验初始化、训练、结果查看和对比等功能
    """
    pass


@cli.command()
@click.option("--name", default="customer_preference_experiment", help="实验名称")
@click.option("--description", default="顾客偏好预测实验", help="实验描述")
@click.option("--output-dir", default=".", help="输出目录")
def init(name, description, output_dir):
    """初始化实验目录结构，生成配置文件"""
    command = InitCommand()
    command.execute(name=name, description=description, output_dir=output_dir)


@cli.command()
@click.option("--config", "config_path", default="experiment_config.yaml", help="配置文件路径")
@click.option("--tracking-uri", default="./experiments", help="实验追踪路径")
def train(config_path, tracking_uri):
    """执行训练脚本，自动记录参数和指标"""
    command = TrainCommand()
    command.execute(config_path=config_path, tracking_uri=tracking_uri)


@cli.command()
@click.option("--tracking-uri", default="./experiments", help="实验追踪路径")
@click.option("--status", help="按状态过滤实验 (completed, running, failed)")
@click.option("--sort-by", default="created_at", 
              type=click.Choice(["created_at", "updated_at", "accuracy", "precision", "recall", "f1_score"]),
              help="排序字段")
@click.option("--limit", type=int, help="限制显示的实验数量")
def list(tracking_uri, status, sort_by, limit):
    """列出历史实验，支持按时间或指标排序"""
    command = ListCommand()
    command.execute(tracking_uri=tracking_uri, status=status, sort_by=sort_by, limit=limit)


@cli.command()
@click.argument("experiment_ids", nargs=-1, required=True)
@click.option("--tracking-uri", default="./experiments", help="实验追踪路径")
def compare(experiment_ids, tracking_uri):
    """对比多个实验的差异"""
    command = CompareCommand()
    command.execute(experiment_ids=list(experiment_ids), tracking_uri=tracking_uri)


@cli.command()
@click.option("--keep-top-n", default=10, type=int, help="保留的最佳实验数量")
@click.option("--metric", default="accuracy", 
              type=click.Choice(["accuracy", "precision", "recall", "f1_score", "auc_roc"]),
              help="用于排序的指标")
@click.option("--dry-run", is_flag=True, help="仅模拟删除，不实际执行")
@click.option("--force", is_flag=True, help="强制删除，不提示确认")
@click.option("--tracking-uri", default="./experiments", help="实验追踪路径")
def clean(keep_top_n, metric, dry_run, force, tracking_uri):
    """清理过期实验数据，保留头部模型"""
    command = CleanCommand()
    command.execute(keep_top_n=keep_top_n, metric=metric, dry_run=dry_run, force=force, tracking_uri=tracking_uri)


if __name__ == "__main__":
    cli()
