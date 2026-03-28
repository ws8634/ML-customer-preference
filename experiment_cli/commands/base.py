"""
命令模式基类模块
定义所有命令的公共接口和基础功能
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from rich.console import Console
from rich.theme import Theme


# 自定义Rich主题
custom_theme = Theme({
    "info": "dim cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "highlight": "bold blue",
    "metric": "bold magenta",
    "command": "bold green"
})

console = Console(theme=custom_theme)


class BaseCommand(ABC):
    """命令基类，所有具体命令必须继承此类"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        """
        初始化命令
        
        Args:
            context: 命令执行上下文，包含配置、管理器等共享对象
        """
        self.context = context or {}
        self.console = console
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        执行命令的抽象方法
        
        Args:
            **kwargs: 命令参数
            
        Returns:
            Any: 命令执行结果
        """
        pass
    
    def print_info(self, message: str) -> None:
        """打印信息级别的消息"""
        self.console.print(f"[info]ℹ️  {message}[/info]")
    
    def print_success(self, message: str) -> None:
        """打印成功级别的消息"""
        self.console.print(f"[success]✅ {message}[/success]")
    
    def print_warning(self, message: str) -> None:
        """打印警告级别的消息"""
        self.console.print(f"[warning]⚠️  {message}[/warning]")
    
    def print_error(self, message: str) -> None:
        """打印错误级别的消息"""
        self.console.print(f"[error]❌ {message}[/error]")
    
    def print_highlight(self, message: str) -> None:
        """打印高亮消息"""
        self.console.print(f"[highlight]{message}[/highlight]")
