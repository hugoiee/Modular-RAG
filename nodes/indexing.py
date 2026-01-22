"""
索引模块：文档加载、切分、向量化入库

基于论文《Modular RAG》的三层架构设计：
- 顶层：IndexModule（索引模块）
- 中层：不同的索引策略（Hierarchical, Small-to-Big, etc.）
- 底层：Operators（加载、分块、向量化、存储）

核心功能：
1. 文档加载（Document Loading）
2. 文本分块（Text Splitting）- 支持多种优化策略
3. 向量化（Vectorization）
4. 存储（Storage）- 支持多种向量数据库

优化策略：
- Sliding Window（滑动窗口）
- Metadata Attachment（元数据附加）
- Small-to-Big（小到大检索策略）
- Hierarchical Indexing（层次化索引）
"""

from typing import List, Dict, Any, Optional, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from .indexing_operators import (
    PDFLoaderOperator,
    WebLoaderOperator,
    TextLoaderOperator,
    DirectoryLoaderOperator,
    RecursiveSplitterOperator,
    SemanticSplitterOperator,
    SmallToBigSplitterOperator,
    StructureAwareSplitterOperator,
    DashScopeEmbeddingOperator,
    ChromaStoreOperator,
    FAISSStoreOperator,
    InMemoryStoreOperator,
)
from .strategies import HierarchicalIndexStrategy


class IndexModule:
    """
    索引模块（顶层）

    使用方式：
    1. 配置各个 operator
    2. 执行 pipeline
    3. 返回向量数据库

    Example:
        config = {
            "loader": {"type": "pdf"},
            "splitter": {"type": "small_to_big", "small_chunk_size": 400},
            "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
            "store": {"type": "chroma", "persist_directory": "./db"}
        }

        index_module = IndexModule(config)
        vectorstore = index_module.index_documents("path/to/docs")
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化索引模块

        Args:
            config: 配置字典，包含各个 operator 的配置
        """
        self.config = config or {}

        # 初始化各个 operator
        self.loader = self._init_loader()
        self.splitter = self._init_splitter()
        self.embedding = self._init_embedding()
        self.store = self._init_store()
        self.strategy = self._init_strategy()

        # 存储处理结果
        self.documents: List[Document] = []
        self.splits: List[Document] = []
        self.vectorstore: Optional[VectorStore] = None

    def _init_loader(self):
        """初始化文档加载器"""
        loader_config = self.config.get("loader", {})
        loader_type = loader_config.get("type", "pdf")

        if loader_type == "pdf":
            return PDFLoaderOperator(loader_config)
        elif loader_type == "text":
            return TextLoaderOperator(loader_config)
        elif loader_type == "directory":
            return DirectoryLoaderOperator(loader_config)
        elif loader_type == "web":
            return WebLoaderOperator(loader_config)
        else:
            print(f"⚠️  未知的 loader 类型: {loader_type}，使用默认 PDF loader")
            return PDFLoaderOperator()

    def _init_splitter(self):
        """初始化文本分块器"""
        splitter_config = self.config.get("splitter", {})
        splitter_type = splitter_config.get("type", "recursive")

        if splitter_type == "recursive":
            return RecursiveSplitterOperator(splitter_config)
        elif splitter_type == "semantic":
            return SemanticSplitterOperator(splitter_config)
        elif splitter_type == "small_to_big":
            return SmallToBigSplitterOperator(splitter_config)
        elif splitter_type == "structure_aware":
            return StructureAwareSplitterOperator(splitter_config)
        else:
            print(f"⚠️  未知的 splitter 类型: {splitter_type}，使用默认递归分块器")
            return RecursiveSplitterOperator()

    def _init_embedding(self):
        """初始化 embedding 模型"""
        embedding_config = self.config.get("embedding", {})
        embedding_type = embedding_config.get("type", "dashscope")

        if embedding_type == "dashscope":
            return DashScopeEmbeddingOperator(embedding_config)
        else:
            print(f"⚠️  未知的 embedding 类型: {embedding_type}，使用默认 DashScope")
            return DashScopeEmbeddingOperator()

    def _init_store(self):
        """初始化存储后端"""
        store_config = self.config.get("store", {})
        store_type = store_config.get("type", "chroma")

        if store_type == "chroma":
            return ChromaStoreOperator(store_config)
        elif store_type == "faiss":
            return FAISSStoreOperator(store_config)
        elif store_type == "memory":
            return InMemoryStoreOperator(store_config)
        else:
            print(f"⚠️  未知的 store 类型: {store_type}，使用默认 Chroma")
            return ChromaStoreOperator()

    def _init_strategy(self):
        """初始化索引策略（可选）"""
        strategy_config = self.config.get("strategy", {})
        strategy_type = strategy_config.get("type", None)

        if strategy_type == "hierarchical":
            return HierarchicalIndexStrategy(strategy_config)
        else:
            return None

    def index_documents(
        self,
        file_path: Union[str, List[str]],
        verbose: bool = True
    ) -> VectorStore:
        """
        执行完整的索引 pipeline

        Pipeline 流程：
        1. 文档加载
        2. 文本分块
        3. （可选）应用索引策略
        4. 向量化 + 存储

        Args:
            file_path: 文件路径或文件路径列表
            verbose: 是否打印详细信息

        Returns:
            VectorStore 对象
        """
        if verbose:
            print("=" * 60)
            print("🚀 开始索引 Pipeline")
            print("=" * 60)

        # 1. 文档加载
        if verbose:
            print("\n📂 步骤 1: 文档加载")
        self.documents = self.loader.execute(file_path)
        if verbose:
            print(f"   ✓ 加载了 {len(self.documents)} 个文档")

        # 2. 文本分块
        if verbose:
            print("\n✂️  步骤 2: 文本分块")
        self.splits = self.splitter.execute(self.documents)
        if verbose:
            print(f"   ✓ 生成了 {len(self.splits)} 个文档块")

        # 3. 应用索引策略（可选）
        if self.strategy:
            if verbose:
                print(f"\n🌲 步骤 3: 应用索引策略 ({self.strategy.__class__.__name__})")

            if isinstance(self.strategy, HierarchicalIndexStrategy):
                # 对于层次化策略，使用原始文档而不是分块后的文档
                self.splits = self.strategy.build_hierarchy(self.documents)

            if verbose:
                print(f"   ✓ 策略应用完成，文档数: {len(self.splits)}")

        # 4. 向量化 + 存储
        if verbose:
            print("\n🔧 步骤 4: 向量化 + 存储")

        # 获取 embedding 模型
        _, embedding_model = self.embedding.execute(self.splits)

        # 存储到向量数据库
        self.vectorstore = self.store.execute(self.splits, embedding_model)

        if verbose:
            print("\n" + "=" * 60)
            print("✅ 索引 Pipeline 完成！")
            print("=" * 60)

        return self.vectorstore

    def load_existing_index(self, verbose: bool = True) -> VectorStore:
        """
        加载已存在的向量数据库

        Args:
            verbose: 是否打印详细信息

        Returns:
            VectorStore 对象
        """
        if verbose:
            print("📂 正在加载已存在的索引...")

        # 获取 embedding 模型
        _, embedding_model = self.embedding.execute([])

        # 加载向量数据库
        if hasattr(self.store, 'load_existing'):
            self.vectorstore = self.store.load_existing(embedding_model)
        else:
            raise NotImplementedError(f"{self.store.__class__.__name__} 不支持加载已存在的索引")

        return self.vectorstore

    def get_vectorstore(self) -> VectorStore:
        """获取向量数据库实例"""
        if self.vectorstore is None:
            raise ValueError("向量数据库未初始化，请先调用 index_documents() 或 load_existing_index()")
        return self.vectorstore

    def get_retriever(self, **kwargs):
        """
        获取检索器

        Args:
            **kwargs: 传递给 retriever 的参数
                - search_type: "similarity" 或 "mmr"
                - search_kwargs: {"k": 3} 等

        Returns:
            Retriever 对象
        """
        vectorstore = self.get_vectorstore()
        return vectorstore.as_retriever(**kwargs)

    def summary(self) -> Dict[str, Any]:
        """
        返回索引模块的摘要信息

        Returns:
            摘要信息字典
        """
        return {
            "loader": self.loader.name,
            "splitter": self.splitter.name,
            "embedding": self.embedding.name,
            "store": self.store.name,
            "strategy": self.strategy.__class__.__name__ if self.strategy else None,
            "documents_count": len(self.documents),
            "splits_count": len(self.splits),
            "vectorstore_initialized": self.vectorstore is not None,
        }
