"""
Pre-Retrieval Operators 包

实现论文中的检索前优化技术：
1. Query Expansion（查询扩展）
2. Query Transformation（查询转换）
3. Query Construction（查询构建）
"""

from .base import BasePreRetrievalOperator
from .expansion import (
    MultiQueryOperator,
    SubQueryOperator,
    HybridExpansionOperator,
)
from .transformation import (
    QueryRewriteOperator,
    HyDEOperator,
    StepBackOperator,
    ChainOfThoughtRewriteOperator,
)
from .construction import (
    TextToSQLOperator,
    TextToCypherOperator,
    MetadataFilterOperator,
)

__all__ = [
    "BasePreRetrievalOperator",
    "MultiQueryOperator",
    "SubQueryOperator",
    "HybridExpansionOperator",
    "QueryRewriteOperator",
    "HyDEOperator",
    "StepBackOperator",
    "ChainOfThoughtRewriteOperator",
    "TextToSQLOperator",
    "TextToCypherOperator",
    "MetadataFilterOperator",
]
