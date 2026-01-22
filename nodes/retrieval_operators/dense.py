"""
Dense Retrieval Operators（密集检索/向量检索）

论文核心技术：
- 使用预训练语言模型进行语义表示
- 通过向量相似度进行检索
- 更好地捕获复杂语义
"""

from typing import Dict, Any, List
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from .base import BaseRetrievalOperator


class DenseRetrieverOperator(BaseRetrievalOperator):
    """
    Dense Retriever 操作器（基于向量数据库）

    功能：
    - 使用向量相似度进行语义检索
    - 支持多种搜索策略（similarity, MMR）
    - 可配置返回文档数量

    优势：
    - 语义理解能力强
    - 能捕获深层语义关系
    - 适合语义搜索场景
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.search_type = self.config.get("search_type", "similarity")
        self.search_kwargs = self.config.get("search_kwargs", {"k": 5})
        self.vectorstore = None

    def build_retriever(self, vectorstore: VectorStore = None, **kwargs) -> BaseRetriever:
        """
        从向量数据库构建检索器

        Args:
            vectorstore: 向量数据库实例
            **kwargs: 额外参数

        Returns:
            检索器实例
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore 参数")

        self.vectorstore = vectorstore

        # 从配置或 kwargs 获取参数
        search_type = kwargs.get("search_type", self.search_type)
        search_kwargs = kwargs.get("search_kwargs", self.search_kwargs)

        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

        print(f"✅ Dense Retriever 已构建")
        print(f"   - 搜索类型: {search_type}")
        print(f"   - 搜索参数: {search_kwargs}")

        return self.retriever


class SemanticRetrieverOperator(DenseRetrieverOperator):
    """
    Semantic Retriever 操作器（语义检索）

    基于 Dense Retriever，增强语义理解

    特性：
    - 支持 MMR（Maximum Marginal Relevance）去重
    - 支持相似度阈值过滤
    - 支持 fetch_k 参数优化性能
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # 默认使用 MMR 策略以增加多样性
        self.search_type = self.config.get("search_type", "mmr")
        self.k = self.config.get("k", 5)
        self.fetch_k = self.config.get("fetch_k", 20)  # 先获取更多候选
        self.lambda_mult = self.config.get("lambda_mult", 0.5)  # MMR 多样性参数
        self.score_threshold = self.config.get("score_threshold", None)

    def build_retriever(self, vectorstore: VectorStore = None, **kwargs) -> BaseRetriever:
        """
        构建语义检索器

        Args:
            vectorstore: 向量数据库实例
            **kwargs: 额外参数

        Returns:
            检索器实例
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore 参数")

        self.vectorstore = vectorstore

        # 构建搜索参数
        search_kwargs = {
            "k": kwargs.get("k", self.k),
        }

        # MMR 特定参数
        if self.search_type == "mmr":
            search_kwargs["fetch_k"] = kwargs.get("fetch_k", self.fetch_k)
            search_kwargs["lambda_mult"] = kwargs.get("lambda_mult", self.lambda_mult)

        # 相似度阈值过滤
        if self.score_threshold is not None:
            search_kwargs["score_threshold"] = self.score_threshold

        self.retriever = self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )

        print(f"✅ Semantic Retriever 已构建")
        print(f"   - 搜索类型: {self.search_type}")
        print(f"   - 返回数量: {search_kwargs['k']}")
        if self.search_type == "mmr":
            print(f"   - 候选数量: {search_kwargs.get('fetch_k', 'N/A')}")
            print(f"   - 多样性参数: {search_kwargs.get('lambda_mult', 'N/A')}")
        if self.score_threshold:
            print(f"   - 相似度阈值: {self.score_threshold}")

        return self.retriever


class MultiVectorRetrieverOperator(BaseRetrievalOperator):
    """
    Multi-Vector Retriever 操作器

    功能：
    - 支持多向量检索（例如使用多个 embedding 模型）
    - 融合多个检索结果
    - 提高检索鲁棒性

    应用场景：
    - 需要从多个角度检索
    - 融合不同粒度的检索结果
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.vectorstores: List[VectorStore] = []
        self.weights = self.config.get("weights", None)  # 各向量库的权重
        self.k = self.config.get("k", 5)

    def build_retriever(self, vectorstores: List[VectorStore] = None, **kwargs) -> BaseRetriever:
        """
        从多个向量数据库构建检索器

        Args:
            vectorstores: 向量数据库列表
            **kwargs: 额外参数

        Returns:
            检索器实例（返回第一个，实际使用 retrieve 方法）
        """
        if not vectorstores or len(vectorstores) == 0:
            raise ValueError("需要提供至少一个 vectorstore")

        self.vectorstores = vectorstores

        # 设置权重（默认均等）
        if self.weights is None:
            self.weights = [1.0 / len(vectorstores)] * len(vectorstores)

        # 返回第一个作为默认
        self.retriever = vectorstores[0].as_retriever(
            search_kwargs={"k": self.k}
        )

        print(f"✅ Multi-Vector Retriever 已构建")
        print(f"   - 向量库数量: {len(self.vectorstores)}")
        print(f"   - 权重: {self.weights}")

        return self.retriever

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        从多个向量库检索并融合结果

        Args:
            query: 查询字符串
            **kwargs: 检索参数

        Returns:
            融合后的文档列表
        """
        all_results = []
        k = kwargs.get("k", self.k)

        # 从每个向量库检索
        for i, vectorstore in enumerate(self.vectorstores):
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)

            # 添加权重信息到 metadata
            for doc in docs:
                doc.metadata["retriever_index"] = i
                doc.metadata["retriever_weight"] = self.weights[i]

            all_results.extend(docs)

        # 去重并根据权重排序
        unique_docs = self._deduplicate_and_rank(all_results)

        return unique_docs[:k]

    def _deduplicate_and_rank(self, documents: List[Document]) -> List[Document]:
        """
        去重并根据权重排序

        Args:
            documents: 文档列表

        Returns:
            排序后的文档列表
        """
        # 使用字典记录每个文档及其累积权重
        doc_weights = {}

        for doc in documents:
            doc_hash = hash(doc.page_content)
            weight = doc.metadata.get("retriever_weight", 1.0)

            if doc_hash in doc_weights:
                # 累积权重
                doc_weights[doc_hash]["weight"] += weight
            else:
                doc_weights[doc_hash] = {
                    "doc": doc,
                    "weight": weight
                }

        # 按权重排序
        sorted_docs = sorted(
            doc_weights.values(),
            key=lambda x: x["weight"],
            reverse=True
        )

        return [item["doc"] for item in sorted_docs]
