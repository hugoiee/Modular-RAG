"""
Post-Retrieval Operator 基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from langchain_core.documents import Document


class BasePostRetrievalOperator(ABC):
    """
    检索后操作器基类

    所有检索后优化技术都继承此类
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
    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        处理检索到的文档

        Args:
            documents: 检索到的文档列表
            query: 原始查询（某些操作需要）

        Returns:
            处理后的文档列表
        """
        pass

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
