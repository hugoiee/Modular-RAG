"""
检索前模块 (Pre-Retrieval Module)

三层架构设计：
- 顶层：PreRetrievalModule（检索前模块）
- 中层：三大类优化策略（Expansion, Transformation, Construction）
- 底层：Operators（具体的优化技术）

核心功能：
在检索之前优化查询，提高检索质量

主要技术：
1. Query Expansion（查询扩展）
   - Multi-Query: 生成多个查询变体
   - Sub-Query: 分解复杂查询

2. Query Transformation（查询转换）
   - Query Rewrite: 优化查询表述
   - HyDE: 生成假设性文档
   - Step-back: 抽象为高层概念

3. Query Construction（查询构建）
   - Text-to-SQL: 转换为SQL查询
   - Text-to-Cypher: 转换为图查询
   - Metadata Filter: 提取过滤条件
"""

from typing import List, Dict, Any, Union, Optional
from .pre_retrieval_operators import (
    BasePreRetrievalOperator,
    MultiQueryOperator,
    SubQueryOperator,
    HybridExpansionOperator,
    QueryRewriteOperator,
    HyDEOperator,
    StepBackOperator,
    ChainOfThoughtRewriteOperator,
    TextToSQLOperator,
    TextToCypherOperator,
    MetadataFilterOperator,
)


class PreRetrievalModule:
    """
    检索前模块（顶层）

    使用方式：
    1. 配置优化策略
    2. 处理查询
    3. 返回优化后的查询

    Example:
        config = {
            "strategy": "multi_query",
            "num_queries": 3,
            "model": "qwen-plus"
        }

        pre_retrieval = PreRetrievalModule(config)
        optimized_queries = pre_retrieval.process(query)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化检索前模块

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.strategy = self.config.get("strategy", "query_rewrite")
        self.operator = self._init_operator()

    def _init_operator(self) -> BasePreRetrievalOperator:
        """根据策略初始化 operator"""
        strategy = self.strategy.lower()

        # Query Expansion
        if strategy == "multi_query":
            return MultiQueryOperator(self.config)
        elif strategy == "sub_query":
            return SubQueryOperator(self.config)
        elif strategy == "hybrid_expansion":
            return HybridExpansionOperator(self.config)

        # Query Transformation
        elif strategy == "query_rewrite":
            return QueryRewriteOperator(self.config)
        elif strategy == "hyde":
            return HyDEOperator(self.config)
        elif strategy == "step_back":
            return StepBackOperator(self.config)
        elif strategy == "cot_rewrite":
            return ChainOfThoughtRewriteOperator(self.config)

        # Query Construction
        elif strategy == "text_to_sql":
            return TextToSQLOperator(self.config)
        elif strategy == "text_to_cypher":
            return TextToCypherOperator(self.config)
        elif strategy == "metadata_filter":
            return MetadataFilterOperator(self.config)

        # 默认
        else:
            print(f"⚠️  未知的策略: {strategy}，使用默认的 Query Rewrite")
            return QueryRewriteOperator(self.config)

    def process(self, query: str, verbose: bool = True) -> Union[str, List[str], Dict[str, Any]]:
        """
        处理查询

        Args:
            query: 原始查询
            verbose: 是否打印详细信息

        Returns:
            优化后的查询（可能是单个字符串、列表或字典）
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔧 检索前优化: {self.strategy}")
            print("=" * 60)

        result = self.operator.execute(query)

        if verbose:
            print("=" * 60)

        return result

    def process_batch(self, queries: List[str], verbose: bool = False) -> List[Union[str, List[str], Dict[str, Any]]]:
        """
        批量处理查询

        Args:
            queries: 查询列表
            verbose: 是否打印详细信息

        Returns:
            优化后的查询列表
        """
        results = []
        for i, query in enumerate(queries, 1):
            if verbose:
                print(f"\n处理查询 {i}/{len(queries)}")
            result = self.process(query, verbose=verbose)
            results.append(result)

        return results

    def change_strategy(self, new_strategy: str, new_config: Dict[str, Any] = None):
        """
        动态更换优化策略

        Args:
            new_strategy: 新策略名称
            new_config: 新配置（可选）
        """
        self.strategy = new_strategy
        if new_config:
            self.config.update(new_config)
        self.operator = self._init_operator()
        print(f"✅ 已切换到策略: {new_strategy}")

    def get_operator(self) -> BasePreRetrievalOperator:
        """获取当前 operator"""
        return self.operator

    def summary(self) -> Dict[str, Any]:
        """
        返回模块摘要信息

        Returns:
            摘要字典
        """
        return {
            "module": "PreRetrievalModule",
            "strategy": self.strategy,
            "operator": self.operator.name,
            "config": self.config,
        }


class PreRetrievalPipeline:
    """
    检索前处理流水线

    支持链式应用多个优化技术

    Example:
        pipeline = PreRetrievalPipeline()
        pipeline.add_step("query_rewrite")
        pipeline.add_step("multi_query", {"num_queries": 3})

        result = pipeline.process(query)
    """

    def __init__(self):
        """初始化流水线"""
        self.steps: List[PreRetrievalModule] = []

    def add_step(self, strategy: str, config: Dict[str, Any] = None):
        """
        添加处理步骤

        Args:
            strategy: 策略名称
            config: 配置字典
        """
        step_config = config or {}
        step_config["strategy"] = strategy

        module = PreRetrievalModule(step_config)
        self.steps.append(module)

        print(f"✅ 已添加步骤: {strategy}")

    def process(self, query: str, verbose: bool = True) -> Union[str, List[str]]:
        """
        通过流水线处理查询

        Args:
            query: 原始查询
            verbose: 是否打印详细信息

        Returns:
            处理后的查询
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔄 检索前流水线: {len(self.steps)} 个步骤")
            print("=" * 60)

        current_queries = [query]

        for i, step in enumerate(self.steps, 1):
            if verbose:
                print(f"\n--- 步骤 {i}: {step.strategy} ---")

            next_queries = []

            for q in current_queries:
                result = step.process(q, verbose=False)

                # 处理不同类型的返回值
                if isinstance(result, list):
                    next_queries.extend(result)
                elif isinstance(result, str):
                    next_queries.append(result)
                else:
                    # 对于字典等其他类型，转换为字符串
                    next_queries.append(str(result))

            current_queries = next_queries

            if verbose:
                print(f"   当前查询数量: {len(current_queries)}")

        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ 流水线处理完成，生成 {len(current_queries)} 个查询")
            print("=" * 60)

        # 如果只有一个查询，返回字符串；否则返回列表
        return current_queries[0] if len(current_queries) == 1 else current_queries

    def clear(self):
        """清空流水线"""
        self.steps = []
        print("✅ 流水线已清空")

    def summary(self) -> Dict[str, Any]:
        """返回流水线摘要"""
        return {
            "type": "PreRetrievalPipeline",
            "num_steps": len(self.steps),
            "steps": [step.strategy for step in self.steps],
        }
