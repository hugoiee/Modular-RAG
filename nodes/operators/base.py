"""
基础 Operator 类：所有操作器的抽象基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseOperator(ABC):
    """
    Operator 基类：论文中提到的底层操作单元
    所有具体的 operator 都应该继承此类
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 operator

        Args:
            config: 配置字典，用于参数化 operator
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        执行 operator 的核心逻辑

        Args:
            input_data: 输入数据

        Returns:
            处理后的输出数据
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
