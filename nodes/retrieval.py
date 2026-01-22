"""
检索模块 (Retrieval Module)

基于论文《Modular RAG》的三层架构设计：
- 顶层：RetrievalModule（检索模块）
- 中层：四大类检索策略（Dense, Sparse, Hybrid, Adaptive）
- 底层：Operators（具体的检索技术）

核心功能：
高效访问和选择相关文档块，为 LLM 提供上下文信息

主要技术：
1. Dense Retrieval（密集检索）
   - 语义向量检索
   - MMR 多样性检索
   - 多向量融合

2. Sparse Retrieval（稀疏检索）
   - BM25 算法
   - TF-IDF 算法
   - 关键词匹配

3. Hybrid Retrieval（混合检索）
   - Dense + Sparse 融合
   - 加权集成
   - 自适应混合

4. Adaptive Retrieval（自适应检索）
   - 动态 K 值
   - 查询路由
   - 阈值过滤
"""

from typing import List, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from .retrieval_operators import (
    BaseRetrievalOperator,
    DenseRetrieverOperator,
    SemanticRetrieverOperator,
    MultiVectorRetrieverOperator,
    BM25RetrieverOperator,
    TFIDFRetrieverOperator,
    KeywordRetrieverOperator,
    RegexRetrieverOperator,
    HybridRetrieverOperator,
    EnsembleRetrieverOperator,
    AdaptiveHybridRetrieverOperator,
    AdaptiveKRetrieverOperator,
    QueryRouterRetrieverOperator,
    ThresholdRetrieverOperator,
)


class RetrievalModule:
    """
    检索模块（顶层）

    使用方式：
    1. 选择检索策略
    2. 构建检索器
    3. 执行检索

    Example:
        config = {
            "strategy": "hybrid",
            "dense_weight": 0.6,
            "sparse_weight": 0.4,
            "k": 5
        }

        retrieval = RetrievalModule(config)
        retrieval.build(vectorstore=vs, documents=docs)
        results = retrieval.retrieve(query)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化检索模块

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.strategy = self.config.get("strategy", "dense")
        self.operator = self._init_operator()

    def _init_operator(self) -> BaseRetrievalOperator:
        """根据策略初始化 operator"""
        strategy = self.strategy.lower()

        # Dense Retrieval
        if strategy == "dense":
            return DenseRetrieverOperator(self.config)
        elif strategy == "semantic":
            return SemanticRetrieverOperator(self.config)
        elif strategy == "multi_vector":
            return MultiVectorRetrieverOperator(self.config)

        # Sparse Retrieval
        elif strategy == "bm25":
            return BM25RetrieverOperator(self.config)
        elif strategy == "tfidf":
            return TFIDFRetrieverOperator(self.config)
        elif strategy == "keyword":
            return KeywordRetrieverOperator(self.config)
        elif strategy == "regex":
            return RegexRetrieverOperator(self.config)

        # Hybrid Retrieval
        elif strategy == "hybrid":
            return HybridRetrieverOperator(self.config)
        elif strategy == "ensemble":
            return EnsembleRetrieverOperator(self.config)
        elif strategy == "adaptive_hybrid":
            return AdaptiveHybridRetrieverOperator(self.config)

        # Adaptive Retrieval
        elif strategy == "adaptive_k":
            return AdaptiveKRetrieverOperator(self.config)
        elif strategy == "query_router":
            return QueryRouterRetrieverOperator(self.config)
        elif strategy == "threshold":
            return ThresholdRetrieverOperator(self.config)

        # 默认
        else:
            print(f"⚠️  未知的策略: {strategy}，使用默认的 Dense Retrieval")
            return DenseRetrieverOperator(self.config)

    def build(self, **kwargs) -> BaseRetriever:
        """
        构建检索器

        Args:
            **kwargs: 构建检索器所需的参数
                - vectorstore: 向量数据库（Dense/Hybrid 需要）
                - documents: 文档列表（Sparse/Hybrid 需要）
                - retrievers: 检索器列表（Ensemble 需要）
                - 其他策略特定参数

        Returns:
            检索器实例
        """
        print("\n" + "=" * 60)
        print(f"🔧 构建检索器: {self.strategy}")
        print("=" * 60)

        retriever = self.operator.build_retriever(**kwargs)

        print("=" * 60)

        return retriever

    def retrieve(
        self,
        query: Union[str, List[str]],
        verbose: bool = True,
        **kwargs
    ) -> List[Document]:
        """
        执行检索

        Args:
            query: 查询（单个或多个）
            verbose: 是否打印详细信息
            **kwargs: 检索参数

        Returns:
            检索到的文档列表
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔍 执行检索: {self.strategy}")
            print("=" * 60)

            if isinstance(query, str):
                print(f"查询: {query}")
            else:
                print(f"查询数量: {len(query)}")

        results = self.operator.retrieve(query, **kwargs)

        if verbose:
            print(f"\n✅ 检索完成，找到 {len(results)} 个文档")
            print("=" * 60)

        return results

    def change_strategy(self, new_strategy: str, new_config: Dict[str, Any] = None):
        """
        动态更换检索策略

        Args:
            new_strategy: 新策略名称
            new_config: 新配置（可选）
        """
        self.strategy = new_strategy
        if new_config:
            self.config.update(new_config)
        self.operator = self._init_operator()
        print(f"✅ 已切换到策略: {new_strategy}")

    def get_retriever(self) -> BaseRetriever:
        """获取底层检索器"""
        return self.operator.get_retriever()

    def summary(self) -> Dict[str, Any]:
        """
        返回模块摘要信息

        Returns:
            摘要字典
        """
        return {
            "module": "RetrievalModule",
            "strategy": self.strategy,
            "operator": self.operator.name,
            "config": self.config,
        }


class RetrievalPipeline:
    """
    检索流水线

    支持多阶段检索和结果融合

    Example:
        pipeline = RetrievalPipeline()
        pipeline.add_stage("bm25", documents=docs, k=10)  # 召回阶段
        pipeline.add_stage("semantic", vectorstore=vs, k=5)  # 精排阶段

        results = pipeline.retrieve(query)
    """

    def __init__(self):
        """初始化流水线"""
        self.stages: List[RetrievalModule] = []

    def add_stage(self, strategy: str, config: Dict[str, Any] = None, **build_kwargs):
        """
        添加检索阶段

        Args:
            strategy: 策略名称
            config: 配置字典
            **build_kwargs: 传递给 build() 的参数
        """
        stage_config = config or {}
        stage_config["strategy"] = strategy

        module = RetrievalModule(stage_config)
        module.build(**build_kwargs)

        self.stages.append(module)

        print(f"✅ 已添加检索阶段: {strategy}")

    def retrieve(self, query: str, verbose: bool = True) -> List[Document]:
        """
        通过流水线检索

        Args:
            query: 查询字符串
            verbose: 是否打印详细信息

        Returns:
            检索到的文档列表
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔄 检索流水线: {len(self.stages)} 个阶段")
            print("=" * 60)

        all_docs = []

        for i, stage in enumerate(self.stages, 1):
            if verbose:
                print(f"\n--- 阶段 {i}: {stage.strategy} ---")

            docs = stage.retrieve(query, verbose=False)
            all_docs.extend(docs)

            if verbose:
                print(f"   本阶段检索到 {len(docs)} 个文档")

        # 去重
        unique_docs = self._deduplicate_documents(all_docs)

        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ 流水线完成，总共 {len(unique_docs)} 个唯一文档")
            print("=" * 60)

        return unique_docs

    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """去除重复文档"""
        seen = set()
        unique_docs = []

        for doc in documents:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)

        return unique_docs

    def clear(self):
        """清空流水线"""
        self.stages = []
        print("✅ 流水线已清空")

    def summary(self) -> Dict[str, Any]:
        """返回流水线摘要"""
        return {
            "type": "RetrievalPipeline",
            "num_stages": len(self.stages),
            "stages": [stage.strategy for stage in self.stages],
        }
