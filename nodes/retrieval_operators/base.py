"""
Retrieval Operator 基类
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


class BaseRetrievalOperator(ABC):
    """
    检索操作器基类

    所有检索器都继承此类
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 operator

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        self.retriever: BaseRetriever = None

    @abstractmethod
    def build_retriever(self, **kwargs) -> BaseRetriever:
        """
        构建检索器

        Args:
            **kwargs: 构建检索器所需的参数

        Returns:
            检索器实例
        """
        pass

    def retrieve(self, query: Union[str, List[str]], **kwargs) -> List[Document]:
        """
        执行检索

        Args:
            query: 查询（单个或多个）
            **kwargs: 检索参数

        Returns:
            检索到的文档列表
        """
        if self.retriever is None:
            raise ValueError("检索器未初始化，请先调用 build_retriever()")

        # 处理单个查询
        if isinstance(query, str):
            return self.retriever.invoke(query, **kwargs)

        # 处理多个查询
        all_docs = []
        for q in query:
            docs = self.retriever.invoke(q, **kwargs)
            all_docs.extend(docs)

        # 去重（基于 page_content）
        unique_docs = self._deduplicate_documents(all_docs)
        return unique_docs

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        去除重复文档

        Args:
            documents: 文档列表

        Returns:
            去重后的文档列表
        """
        seen = set()
        unique_docs = []

        for doc in documents:
            # 使用内容的哈希作为唯一标识
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        return unique_docs

    def get_retriever(self) -> BaseRetriever:
        """获取检索器实例"""
        if self.retriever is None:
            raise ValueError("检索器未初始化")
        return self.retriever

    def __repr__(self) -> str:
        return f"{self.name}(config={self.config})"
