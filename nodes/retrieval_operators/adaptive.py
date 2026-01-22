"""
Adaptive Retrieval Operators（自适应检索）

实现智能化的检索策略：
- 根据查询特征动态调整
- 自适应确定检索数量
- 智能路由到合适的检索器
"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from .base import BaseRetrievalOperator


class AdaptiveKRetrieverOperator(BaseRetrievalOperator):
    """
    Adaptive-K Retriever 操作器

    功能：
    - 根据查询复杂度动态确定返回文档数量
    - 简单查询返回较少文档
    - 复杂查询返回更多文档

    优势：
    - 减少无关信息干扰
    - 优化 LLM 上下文使用
    - 提升生成质量
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_k = self.config.get("min_k", 3)
        self.max_k = self.config.get("max_k", 10)
        self.default_k = self.config.get("default_k", 5)
        self.vectorstore = None

    def build_retriever(self, vectorstore: VectorStore = None, **kwargs) -> BaseRetriever:
        """
        构建自适应 K 检索器

        Args:
            vectorstore: 向量数据库
            **kwargs: 额外参数

        Returns:
            检索器实例
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore")

        self.vectorstore = vectorstore

        # 使用默认 k 创建检索器
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.default_k}
        )

        print(f"✅ Adaptive-K Retriever 已构建")
        print(f"   - K 范围: [{self.min_k}, {self.max_k}]")
        print(f"   - 默认 K: {self.default_k}")

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
        # 分析查询复杂度
        complexity = self._analyze_complexity(query)

        # 根据复杂度确定 k
        k = self._determine_k(complexity)

        print(f"📊 查询复杂度: {complexity:.2f} -> K = {k}")

        # 执行检索
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

        return retriever.invoke(query)

    def _analyze_complexity(self, query: str) -> float:
        """
        分析查询复杂度

        Args:
            query: 查询字符串

        Returns:
            复杂度分数 (0-1)
        """
        complexity = 0.0

        # 长度因素
        if len(query) > 100:
            complexity += 0.3
        elif len(query) > 50:
            complexity += 0.2
        else:
            complexity += 0.1

        # 关键词数量
        words = query.split()
        if len(words) > 15:
            complexity += 0.3
        elif len(words) > 8:
            complexity += 0.2
        else:
            complexity += 0.1

        # 包含复杂词汇
        complex_indicators = ["比较", "分析", "评估", "综合", "对比", "详细", "全面"]
        if any(indicator in query for indicator in complex_indicators):
            complexity += 0.2

        # 包含多个问题
        question_marks = query.count("？") + query.count("?")
        if question_marks > 1:
            complexity += 0.2

        return min(complexity, 1.0)

    def _determine_k(self, complexity: float) -> int:
        """
        根据复杂度确定 k

        Args:
            complexity: 复杂度分数

        Returns:
            文档数量 k
        """
        k = int(self.min_k + (self.max_k - self.min_k) * complexity)
        return max(self.min_k, min(k, self.max_k))


class QueryRouterRetrieverOperator(BaseRetrievalOperator):
    """
    Query Router Retriever 操作器

    功能：
    - 根据查询类型路由到不同的检索器
    - 问题类查询 -> 语义检索
    - 关键词查询 -> BM25 检索
    - 混合查询 -> 混合检索

    优势：
    - 智能选择最佳检索策略
    - 提升检索效率和效果
    - 减少不必要的计算
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.retrievers: Dict[str, BaseRetriever] = {}

    def build_retriever(
        self,
        retrievers: Dict[str, BaseRetriever] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        构建路由检索器

        Args:
            retrievers: 检索器字典 {"dense": retriever1, "sparse": retriever2, ...}
            **kwargs: 额外参数

        Returns:
            检索器实例（返回第一个，实际使用 retrieve 方法）
        """
        if not retrievers or len(retrievers) == 0:
            raise ValueError("需要提供至少一个检索器")

        self.retrievers = retrievers

        # 返回第一个作为默认
        self.retriever = list(retrievers.values())[0]

        print(f"✅ Query Router Retriever 已构建")
        print(f"   - 可用检索器: {list(retrievers.keys())}")

        return self.retriever

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        根据查询类型路由检索

        Args:
            query: 查询字符串
            **kwargs: 检索参数

        Returns:
            检索到的文档列表
        """
        # 分析查询类型
        query_type = self._classify_query(query)

        print(f"🔀 查询类型: {query_type}")

        # 选择检索器
        retriever = self._select_retriever(query_type)

        # 执行检索
        return retriever.invoke(query)

    def _classify_query(self, query: str) -> str:
        """
        分类查询类型

        Args:
            query: 查询字符串

        Returns:
            查询类型 ("semantic", "keyword", "hybrid")
        """
        query_lower = query.lower()

        # 语义查询指标
        semantic_indicators = [
            "什么", "为什么", "如何", "怎么", "解释", "描述", "说明",
            "what", "why", "how", "explain", "describe"
        ]

        # 关键词查询指标
        keyword_indicators = [
            "查找", "搜索", "列出", "包含", "匹配",
            "find", "search", "list", "contain", "match"
        ]

        semantic_score = sum(1 for ind in semantic_indicators if ind in query_lower)
        keyword_score = sum(1 for ind in keyword_indicators if ind in query_lower)

        # 判断类型
        if semantic_score > keyword_score and semantic_score > 0:
            return "semantic"
        elif keyword_score > semantic_score and keyword_score > 0:
            return "keyword"
        else:
            # 默认混合
            return "hybrid"

    def _select_retriever(self, query_type: str) -> BaseRetriever:
        """
        根据查询类型选择检索器

        Args:
            query_type: 查询类型

        Returns:
            检索器实例
        """
        # 映射关系
        type_to_key = {
            "semantic": ["dense", "semantic", "vector"],
            "keyword": ["sparse", "bm25", "keyword"],
            "hybrid": ["hybrid", "ensemble"],
        }

        # 查找匹配的检索器
        for key_option in type_to_key.get(query_type, []):
            if key_option in self.retrievers:
                print(f"   -> 使用 {key_option} 检索器")
                return self.retrievers[key_option]

        # 如果没有找到，使用第一个
        first_key = list(self.retrievers.keys())[0]
        print(f"   -> 使用默认检索器: {first_key}")
        return self.retrievers[first_key]


class ThresholdRetrieverOperator(BaseRetrievalOperator):
    """
    Threshold Retriever 操作器

    功能：
    - 只返回相似度超过阈值的文档
    - 动态确定返回数量
    - 避免低质量文档干扰

    优势：
    - 保证检索质量
    - 减少噪音
    - 适应不同查询质量
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.score_threshold = self.config.get("score_threshold", 0.7)
        self.min_docs = self.config.get("min_docs", 1)
        self.max_docs = self.config.get("max_docs", 10)
        self.vectorstore = None

    def build_retriever(self, vectorstore: VectorStore = None, **kwargs) -> BaseRetriever:
        """
        构建阈值检索器

        Args:
            vectorstore: 向量数据库
            **kwargs: 额外参数

        Returns:
            检索器实例
        """
        if vectorstore is None:
            raise ValueError("需要提供 vectorstore")

        self.vectorstore = vectorstore

        # 使用 similarity_score_threshold 搜索类型
        self.retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": self.score_threshold,
                "k": self.max_docs,
            }
        )

        print(f"✅ Threshold Retriever 已构建")
        print(f"   - 相似度阈值: {self.score_threshold}")
        print(f"   - 文档数范围: [{self.min_docs}, {self.max_docs}]")

        return self.retriever

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        基于阈值检索

        Args:
            query: 查询字符串
            **kwargs: 检索参数

        Returns:
            检索到的文档列表
        """
        # 执行检索
        docs = self.retriever.invoke(query)

        # 确保至少返回 min_docs 个文档
        if len(docs) < self.min_docs:
            print(f"⚠️  检索结果少于最小值，降低阈值重试...")

            # 使用较低阈值重试
            fallback_retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.min_docs}
            )
            docs = fallback_retriever.invoke(query)

        print(f"📄 返回 {len(docs)} 个文档（阈值: {self.score_threshold}）")

        return docs[:self.max_docs]
