"""
Hybrid Retrieval Operators（混合检索）

论文核心技术：
- 结合 Dense 和 Sparse 检索的优势
- 利用互补性提升检索效果
- 增强零样本检索能力
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from .base import BaseRetrievalOperator


class HybridRetrieverOperator(BaseRetrievalOperator):
    """
    Hybrid Retriever 操作器（混合检索）

    核心思想（论文重点）：
    - 结合 Dense Retrieval（语义理解）和 Sparse Retrieval（关键词匹配）
    - 使用加权融合策略
    - 综合两者优势

    优势：
    - Dense 捕获语义相似度
    - Sparse 确保关键词覆盖
    - 提高检索鲁棒性
    - 增强零样本能力

    应用场景：
    - 生产环境的推荐配置
    - 需要平衡语义和精确匹配
    - 提高检索召回率
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.dense_weight = self.config.get("dense_weight", 0.5)
        self.sparse_weight = self.config.get("sparse_weight", 0.5)
        self.k = self.config.get("k", 5)

        self.dense_retriever = None
        self.sparse_retriever = None

    def build_retriever(
        self,
        vectorstore: VectorStore = None,
        documents: List[Document] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        构建混合检索器

        Args:
            vectorstore: 向量数据库（用于 Dense 检索）
            documents: 文档列表（用于 Sparse 检索）
            **kwargs: 额外参数

        Returns:
            混合检索器实例
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore 用于 Dense 检索")

        if documents is None or len(documents) == 0:
            raise ValueError("需要提供 documents 用于 Sparse 检索")

        k = kwargs.get("k", self.k)
        dense_weight = kwargs.get("dense_weight", self.dense_weight)
        sparse_weight = kwargs.get("sparse_weight", self.sparse_weight)

        # 构建 Dense Retriever（向量检索）
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        # 构建 Sparse Retriever（BM25）
        self.sparse_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=k,
        )

        # 使用 EnsembleRetriever 融合两者
        self.retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[dense_weight, sparse_weight],
        )

        print(f"✅ Hybrid Retriever 已构建")
        print(f"   - Dense 权重: {dense_weight}")
        print(f"   - Sparse 权重: {sparse_weight}")
        print(f"   - 返回数量: {k}")

        return self.retriever


class EnsembleRetrieverOperator(BaseRetrievalOperator):
    """
    Ensemble Retriever 操作器（集成检索）

    功能：
    - 集成多个不同的检索器
    - 支持自定义权重分配
    - 融合多种检索策略

    应用场景：
    - 需要结合多种检索方法
    - 自定义检索流水线
    - 提升检索覆盖率
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.weights = self.config.get("weights", None)
        self.k = self.config.get("k", 5)
        self.retrievers: List[BaseRetriever] = []

    def build_retriever(
        self,
        retrievers: List[BaseRetriever] = None,
        weights: List[float] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        构建集成检索器

        Args:
            retrievers: 检索器列表
            weights: 权重列表
            **kwargs: 额外参数

        Returns:
            集成检索器实例
        """
        if retrievers is None or len(retrievers) == 0:
            raise ValueError("需要提供至少一个检索器")

        self.retrievers = retrievers

        # 设置权重
        if weights is not None:
            self.weights = weights
        elif self.weights is None:
            # 默认均等权重
            self.weights = [1.0 / len(retrievers)] * len(retrievers)

        # 创建 EnsembleRetriever
        self.retriever = EnsembleRetriever(
            retrievers=self.retrievers,
            weights=self.weights,
        )

        print(f"✅ Ensemble Retriever 已构建")
        print(f"   - 检索器数量: {len(self.retrievers)}")
        print(f"   - 权重: {self.weights}")

        return self.retriever


class AdaptiveHybridRetrieverOperator(BaseRetrievalOperator):
    """
    Adaptive Hybrid Retriever 操作器（自适应混合检索）

    核心思想：
    - 根据查询类型动态调整 Dense/Sparse 权重
    - 关键词查询偏向 Sparse
    - 语义查询偏向 Dense

    优势：
    - 智能适应不同查询
    - 提升检索效果
    - 减少手动调参
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.dense_retriever = None
        self.sparse_retriever = None

    def build_retriever(
        self,
        vectorstore: VectorStore = None,
        documents: List[Document] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        构建自适应混合检索器

        Args:
            vectorstore: 向量数据库
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            检索器实例（返回第一个，实际使用 retrieve 方法）
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore")

        if documents is None or len(documents) == 0:
            raise ValueError("需要提供 documents")

        k = kwargs.get("k", self.k)

        # 构建两个检索器
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        self.sparse_retriever = BM25Retriever.from_documents(
            documents=documents,
            k=k,
        )

        # 返回 dense retriever 作为默认
        self.retriever = self.dense_retriever

        print(f"✅ Adaptive Hybrid Retriever 已构建")
        print(f"   - 将根据查询类型自动调整权重")

        return self.retriever

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        自适应检索

        Args:
            query: 查询字符串
            **kwargs: 检索参数

        Returns:
            检索到的文档列表
        """
        # 分析查询特征
        dense_weight, sparse_weight = self._analyze_query(query)

        print(f"📊 查询分析:")
        print(f"   - Dense 权重: {dense_weight:.2f}")
        print(f"   - Sparse 权重: {sparse_weight:.2f}")

        k = kwargs.get("k", self.k)

        # 从两个检索器获取结果
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        # 加权融合
        merged_docs = self._weighted_merge(
            dense_docs, sparse_docs,
            dense_weight, sparse_weight
        )

        return merged_docs[:k]

    def _analyze_query(self, query: str) -> tuple[float, float]:
        """
        分析查询特征，决定权重分配

        Args:
            query: 查询字符串

        Returns:
            (dense_weight, sparse_weight)
        """
        # 简单的启发式规则
        query_lower = query.lower()

        # 关键词密度
        words = query.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

        # 包含疑问词 -> 更偏向语义检索
        question_words = ["什么", "为什么", "如何", "怎么", "哪些", "what", "why", "how"]
        has_question = any(qw in query_lower for qw in question_words)

        # 短查询且包含专有名词 -> 更偏向关键词检索
        is_short = len(query) < 20

        # 决定权重
        if has_question and not is_short:
            # 语义查询
            return 0.7, 0.3
        elif is_short or avg_word_length > 6:
            # 关键词查询（短查询或包含长词）
            return 0.3, 0.7
        else:
            # 平衡
            return 0.5, 0.5

    def _weighted_merge(
        self,
        dense_docs: List[Document],
        sparse_docs: List[Document],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Document]:
        """
        加权融合文档

        Args:
            dense_docs: Dense 检索结果
            sparse_docs: Sparse 检索结果
            dense_weight: Dense 权重
            sparse_weight: Sparse 权重

        Returns:
            融合后的文档列表
        """
        # 记录文档得分
        doc_scores = {}

        # 处理 Dense 结果
        for i, doc in enumerate(dense_docs):
            doc_hash = hash(doc.page_content)
            score = (len(dense_docs) - i) * dense_weight  # 排名越前分数越高
            doc_scores[doc_hash] = {
                "doc": doc,
                "score": score
            }

        # 处理 Sparse 结果
        for i, doc in enumerate(sparse_docs):
            doc_hash = hash(doc.page_content)
            score = (len(sparse_docs) - i) * sparse_weight

            if doc_hash in doc_scores:
                doc_scores[doc_hash]["score"] += score
            else:
                doc_scores[doc_hash] = {
                    "doc": doc,
                    "score": score
                }

        # 按分数排序
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [item["doc"] for item in sorted_docs]
