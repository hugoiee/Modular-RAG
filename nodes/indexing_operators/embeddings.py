"""
向量化 Operators
支持不同的 embedding 模型
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from .base import BaseOperator


class EmbeddingOperator(BaseOperator):
    """向量化操作器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.embedding_model = None

    def execute(self, documents: List[Document]) -> tuple[List[Document], Embeddings]:
        """
        返回文档列表和 embedding 模型
        （embedding 在存储时进行，这里只是初始化模型）

        Args:
            documents: Document 对象列表

        Returns:
            (文档列表, embedding 模型)
        """
        return documents, self.embedding_model

    def get_model(self) -> Embeddings:
        """获取 embedding 模型"""
        return self.embedding_model


class DashScopeEmbeddingOperator(EmbeddingOperator):
    """
    DashScope (通义千问) Embedding 操作器
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "text-embedding-v4")
        self.api_key = self.config.get("api_key", None)

        # 初始化 embedding 模型
        if self.api_key:
            self.embedding_model = DashScopeEmbeddings(
                model=self.model_name,
                dashscope_api_key=self.api_key,
            )
        else:
            # 如果没有传入 API key，使用环境变量
            self.embedding_model = DashScopeEmbeddings(
                model=self.model_name,
            )

    def execute(self, documents: List[Document]) -> tuple[List[Document], Embeddings]:
        """
        准备向量化

        Args:
            documents: Document 对象列表

        Returns:
            (文档列表, embedding 模型)
        """
        print(f"🔧 使用 DashScope Embedding 模型: {self.model_name}")
        return documents, self.embedding_model


class OpenAIEmbeddingOperator(EmbeddingOperator):
    """
    OpenAI Embedding 操作器
    （预留接口，可根据需要实现）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "text-embedding-3-small")
        # 这里可以添加 OpenAI embedding 的初始化逻辑
        print(f"⚠️  OpenAI Embedding 操作器尚未完全实现")


class HuggingFaceEmbeddingOperator(EmbeddingOperator):
    """
    HuggingFace Embedding 操作器
    （预留接口，可根据需要实现）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "BAAI/bge-small-zh-v1.5")
        # 这里可以添加 HuggingFace embedding 的初始化逻辑
        print(f"⚠️  HuggingFace Embedding 操作器尚未完全实现")
