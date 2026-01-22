"""
Generation Operator 基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.documents import Document


class BaseGenerationOperator(ABC):
    """
    生成操作器基类

    所有生成相关操作都继承此类
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
    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        执行生成

        Args:
            query: 用户查询
            context: 检索到的上下文文档
            **kwargs: 额外参数

        Returns:
            生成的答案
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
