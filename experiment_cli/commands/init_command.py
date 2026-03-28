"""
初始化命令模块
负责创建实验目录结构和配置文件
"""
import os
from pathlib import Path
from typing import Dict, Any
from experiment_cli.commands.base import BaseCommand
from experiment_cli.core.config_manager import ConfigManager
from experiment_cli.core.models import ExperimentConfig


class InitCommand(BaseCommand):
    """初始化实验命令类"""
    
    def execute(self, name: str = "customer_preference_experiment",
                description: str = "顾客偏好预测实验",
                output_dir: str = ".") -> Dict[str, Any]:
        """
        执行初始化命令
        
        Args:
            name: 实验名称
            description: 实验描述
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 初始化结果信息
        """
        self.print_highlight(f"开始初始化实验: {name}")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建必要的子目录
        self._create_directory_structure(output_path)
        
        # 创建默认配置文件
        config_path = output_path / "experiment_config.yaml"
        config_manager = ConfigManager(config_path)
        
        default_config = config_manager.create_default_config(name, description)
        config_manager.save_config(default_config)
        
        # 创建示例数据目录和README
        self._create_example_files(output_path)
        
        self.print_success(f"实验初始化完成!")
        self.print_info(f"配置文件已创建: {config_path}")
        self.print_info(f"请修改配置文件后使用 'experiment-cli train' 开始训练")
        
        return {
            "success": True,
            "config_path": str(config_path),
            "experiment_name": name
        }
    
    def _create_directory_structure(self, root_path: Path) -> None:
        """创建实验目录结构"""
        directories = [
            "data",          # 数据目录
            "models",        # 模型保存目录
            "experiments",   # 实验追踪目录
            "logs",          # 日志目录
            "notebooks"      # 笔记本目录
        ]
        
        for directory in directories:
            dir_path = root_path / directory
            dir_path.mkdir(exist_ok=True)
            self.print_info(f"创建目录: {dir_path}")
    
    def _create_example_files(self, root_path: Path) -> None:
        """创建示例文件"""
        # 创建数据目录的README
        data_readme = root_path / "data" / "README.md"
        with open(data_readme, "w", encoding="utf-8") as f:
            f.write("# 数据目录\n\n")
            f.write("请将训练数据和测试数据放置在此目录中。\n")
            f.write("支持的格式: CSV, JSON等。\n")
        
        # 创建主目录的README
        main_readme = root_path / "README_EXPERIMENT.md"
        with open(main_readme, "w", encoding="utf-8") as f:
            f.write("# 顾客偏好预测实验\n\n")
            f.write("## 目录结构\n")
            f.write("- data/: 训练和测试数据\n")
            f.write("- models/: 训练好的模型\n")
            f.write("- experiments/: 实验追踪数据\n")
            f.write("- logs/: 日志文件\n")
            f.write("- notebooks/: 分析笔记本\n\n")
            f.write("## 使用方法\n")
            f.write("1. 修改 experiment_config.yaml 配置文件\n")
            f.write("2. 运行 `experiment-cli train` 开始训练\n")
            f.write("3. 运行 `experiment-cli list` 查看实验结果\n")
