"""
Retrieval Operators 包

实现论文中的检索技术：
1. Dense Retrieval（密集检索/向量检索）
2. Sparse Retrieval（稀疏检索/关键词检索）
3. Hybrid Retrieval（混合检索）
4. Adaptive Retrieval（自适应检索）
"""

from .base import BaseRetrievalOperator
from .dense import (
    DenseRetrieverOperator,
    SemanticRetrieverOperator,
    MultiVectorRetrieverOperator,
)
from .sparse import (
    BM25RetrieverOperator,
    TFIDFRetrieverOperator,
    KeywordRetrieverOperator,
    RegexRetrieverOperator,
)
from .hybrid import (
    HybridRetrieverOperator,
    EnsembleRetrieverOperator,
    AdaptiveHybridRetrieverOperator,
)
from .adaptive import (
    AdaptiveKRetrieverOperator,
    QueryRouterRetrieverOperator,
    ThresholdRetrieverOperator,
)

__all__ = [
    "BaseRetrievalOperator",
    "DenseRetrieverOperator",
    "SemanticRetrieverOperator",
    "MultiVectorRetrieverOperator",
    "BM25RetrieverOperator",
    "TFIDFRetrieverOperator",
    "KeywordRetrieverOperator",
    "RegexRetrieverOperator",
    "HybridRetrieverOperator",
    "EnsembleRetrieverOperator",
    "AdaptiveHybridRetrieverOperator",
    "AdaptiveKRetrieverOperator",
    "QueryRouterRetrieverOperator",
    "ThresholdRetrieverOperator",
]
