#!/usr/bin/env python3
import click
from rich.console import Console
from typing import Dict, Any
import json

from src.commands.init_command import InitCommand
from src.commands.train_command import TrainCommand
from src.commands.list_command import ListCommand
from src.commands.compare_command import CompareCommand
from src.commands.clean_command import CleanCommand


console = Console()


def parse_hyperparameters(ctx, param, value) -> Dict[str, Any]:
    """解析超参数JSON字符串"""
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise click.BadParameter("超参数必须是有效的JSON格式")


def parse_features(ctx, param, value):
    """解析特征列表"""
    if not value:
        return []
    return [f.strip() for f in value.split(",")]


@click.group()
@click.version_option(version="1.0.0", prog_name="customer-preference")
@click.pass_context
def cli(ctx):
    """顾客偏好预测实验管理工具"""
    ctx.ensure_object(dict)
    ctx.obj["console"] = console


@cli.command()
@click.option("--path", default=".", help="项目根目录路径", type=click.Path())
@click.pass_context
def init(ctx, path):
    """初始化实验目录结构"""
    command = InitCommand(ctx.obj["console"])
    command.execute(path=path)


@cli.command()
@click.option("--name", "experiment_name", help="实验名称", default=None)
@click.option("--model", "model_type", help="模型类型", type=click.Choice(["random_forest", "gradient_boosting", "logistic_regression"]))
@click.option("--dataset", "dataset_path", help="数据集路径", default="data/customers.csv")
@click.option("--hyperparams", "hyperparameters", help="超参数(JSON格式)", callback=parse_hyperparameters, default="{}")
@click.option("--features", help="特征列(逗号分隔)", callback=parse_features, default=None)
@click.option("--target", help="目标列名", default=None)
@click.option("--path", default=".", help="项目根目录路径", type=click.Path())
@click.pass_context
def train(ctx, experiment_name, model_type, dataset_path, hyperparameters, features, target, path):
    """执行模型训练并记录结果"""
    command = TrainCommand(ctx.obj["console"])
    command.execute(
        experiment_name=experiment_name,
        model_type=model_type,
        dataset_path=dataset_path,
        hyperparameters=hyperparameters,
        features=features,
        target=target,
        path=path
    )


@cli.command("list")
@click.option("--sort-by", "sort_by", help="排序方式", type=click.Choice(["time", "metric"]), default="time")
@click.option("--metric", help="排序指标", default="accuracy")
@click.option("--ascending", is_flag=True, help="升序排列")
@click.option("--path", default=".", help="项目根目录路径", type=click.Path())
@click.pass_context
def list_cmd(ctx, sort_by, metric, ascending, path):
    """列出历史实验记录"""
    command = ListCommand(ctx.obj["console"])
    command.execute(
        sort_by=sort_by,
        metric=metric,
        ascending=ascending,
        path=path
    )


@cli.command()
@click.argument("experiment_ids", nargs=-1, required=True)
@click.option("--path", default=".", help="项目根目录路径", type=click.Path())
@click.pass_context
def compare(ctx, experiment_ids, path):
    """对比多个实验的差异"""
    command = CompareCommand(ctx.obj["console"])
    command.execute(
        experiment_ids=list(experiment_ids),
        path=path
    )


@cli.command()
@click.option("--keep-top", "keep_top", help="保留顶部模型数量", type=int, default=5)
@click.option("--days", help="保留N天内的实验", type=int, default=None)
@click.option("--metric", help="排序指标", default="accuracy")
@click.option("--dry-run", "dry_run", is_flag=True, help="试运行模式")
@click.option("--force", is_flag=True, help="强制删除（不确认）")
@click.option("--path", default=".", help="项目根目录路径", type=click.Path())
@click.pass_context
def clean(ctx, keep_top, days, metric, dry_run, force, path):
    """清理过期实验数据"""
    command = CleanCommand(ctx.obj["console"])
    command.execute(
        keep_top=keep_top,
        days=days,
        metric=metric,
        dry_run=dry_run,
        force=force,
        path=path
    )


if __name__ == "__main__":
    cli()
