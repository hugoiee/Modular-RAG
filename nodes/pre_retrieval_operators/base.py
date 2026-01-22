"""
Pre-Retrieval Operator 基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union


class BasePreRetrievalOperator(ABC):
    """
    检索前操作器基类

    所有检索前优化技术都继承此类
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 operator

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def execute(self, query: str) -> Union[str, List[str]]:
        """
        执行查询优化

        Args:
            query: 原始查询

        Returns:
            优化后的查询（单个或多个）
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
