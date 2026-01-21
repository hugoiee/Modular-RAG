"""
存储 Operators
支持不同的向量数据库
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.vectorstores import VectorStore
from .base import BaseOperator


class StoreOperator(BaseOperator):
    """存储操作器基类"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.vectorstore: Optional[VectorStore] = None

    def execute(
        self, documents: List[Document], embedding_model: Embeddings
    ) -> VectorStore:
        """
        将文档向量化并存储到向量数据库

        Args:
            documents: Document 对象列表
            embedding_model: Embedding 模型

        Returns:
            VectorStore 对象
        """
        raise NotImplementedError

    def get_vectorstore(self) -> VectorStore:
        """获取向量数据库实例"""
        return self.vectorstore


class ChromaStoreOperator(StoreOperator):
    """
    Chroma 向量数据库操作器
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.persist_directory = self.config.get("persist_directory", "./chroma_db")
        self.collection_name = self.config.get("collection_name", "default_collection")

    def execute(
        self, documents: List[Document], embedding_model: Embeddings
    ) -> VectorStore:
        """
        使用 Chroma 存储文档向量

        Args:
            documents: Document 对象列表
            embedding_model: Embedding 模型

        Returns:
            Chroma VectorStore 对象
        """
        print(f"💾 正在使用 Chroma 存储 {len(documents)} 个文档块...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )

        print(f"✅ 向量数据库创建成功！")
        print(f"   - 存储路径: {self.persist_directory}")
        print(f"   - 集合名称: {self.collection_name}")
        print(f"   - 文档数量: {len(documents)}")

        return self.vectorstore

    def load_existing(self, embedding_model: Embeddings) -> VectorStore:
        """
        加载已存在的 Chroma 向量数据库

        Args:
            embedding_model: Embedding 模型

        Returns:
            Chroma VectorStore 对象
        """
        print(f"📂 正在加载已存在的 Chroma 数据库...")

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding_model,
            collection_name=self.collection_name,
        )

        print(f"✅ 向量数据库加载成功！")
        return self.vectorstore


class FAISSStoreOperator(StoreOperator):
    """
    FAISS 向量数据库操作器
    （FAISS 适合大规模数据，性能更好）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.index_path = self.config.get("index_path", "./faiss_index")

    def execute(
        self, documents: List[Document], embedding_model: Embeddings
    ) -> VectorStore:
        """
        使用 FAISS 存储文档向量

        Args:
            documents: Document 对象列表
            embedding_model: Embedding 模型

        Returns:
            FAISS VectorStore 对象
        """
        print(f"💾 正在使用 FAISS 存储 {len(documents)} 个文档块...")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,
        )

        # 保存索引
        self.vectorstore.save_local(self.index_path)

        print(f"✅ 向量数据库创建成功！")
        print(f"   - 存储路径: {self.index_path}")
        print(f"   - 文档数量: {len(documents)}")

        return self.vectorstore

    def load_existing(self, embedding_model: Embeddings) -> VectorStore:
        """
        加载已存在的 FAISS 向量数据库

        Args:
            embedding_model: Embedding 模型

        Returns:
            FAISS VectorStore 对象
        """
        print(f"📂 正在加载已存在的 FAISS 索引...")

        self.vectorstore = FAISS.load_local(
            self.index_path,
            embedding_model,
            allow_dangerous_deserialization=True,  # FAISS 需要此参数
        )

        print(f"✅ 向量数据库加载成功！")
        return self.vectorstore


class InMemoryStoreOperator(StoreOperator):
    """
    内存存储操作器
    用于测试或小规模数据
    """

    def execute(
        self, documents: List[Document], embedding_model: Embeddings
    ) -> VectorStore:
        """
        使用内存存储文档向量（使用 FAISS，不持久化）

        Args:
            documents: Document 对象列表
            embedding_model: Embedding 模型

        Returns:
            FAISS VectorStore 对象（内存）
        """
        print(f"💾 正在使用内存存储 {len(documents)} 个文档块...")

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model,
        )

        print(f"✅ 内存向量数据库创建成功！（数据不会持久化）")
        print(f"   - 文档数量: {len(documents)}")

        return self.vectorstore
