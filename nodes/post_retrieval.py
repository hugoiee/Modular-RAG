"""
检索后模块 (Post-Retrieval Module)

基于论文《Modular RAG》的三层架构设计：
- 顶层：PostRetrievalModule（检索后模块）
- 中层：三大类优化策略（Rerank, Compression, Selection）
- 底层：Operators（具体的优化技术）

核心功能：
优化和精炼检索到的文档块，提高 LLM 的信息感知能力

解决的主要挑战：
1. "Lost in the middle"：LLM 倾向于记住开头和结尾
2. 噪音文档：不相关或矛盾的文档干扰
3. 上下文窗口限制：需要压缩和精选

主要技术：
1. Rerank（重排序）
   - Rule-based: 基于规则的重排序
   - Diversity: 多样性重排序
   - LLM-based: 使用 LLM 评分重排序
   - Lost-in-Middle Aware: 优化位置布局

2. Compression（压缩）
   - Context Compression: 上下文压缩
   - Summary: 摘要压缩
   - Token-level: Token 级压缩
   - Adaptive: 自适应压缩

3. Selection/Filtering（选择/过滤）
   - Relevance: 相关性过滤
   - Redundancy: 冗余过滤
   - Quality: 质量过滤
   - Contradiction: 矛盾过滤
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

from .post_retrieval_operators import (
    BasePostRetrievalOperator,
    RerankOperator,
    DiversityRerankOperator,
    LLMRerankOperator,
    ContextCompressionOperator,
    SummaryCompressionOperator,
    TokenCompressionOperator,
    SelectionOperator,
    RelevanceFilterOperator,
    RedundancyFilterOperator,
)


class PostRetrievalModule:
    """
    检索后模块（顶层）

    使用方式：
    1. 选择优化策略
    2. 处理检索结果
    3. 返回优化后的文档

    Example:
        config = {
            "strategy": "rerank",
            "top_n": 5
        }

        post_retrieval = PostRetrievalModule(config)
        optimized_docs = post_retrieval.process(documents, query)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化检索后模块

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.strategy = self.config.get("strategy", "rerank")
        self.operator = self._init_operator()

    def _init_operator(self) -> BasePostRetrievalOperator:
        """根据策略初始化 operator"""
        strategy = self.strategy.lower()

        # Rerank
        if strategy == "rerank":
            return RerankOperator(self.config)
        elif strategy == "diversity_rerank":
            return DiversityRerankOperator(self.config)
        elif strategy == "llm_rerank":
            return LLMRerankOperator(self.config)

        # Compression
        elif strategy == "context_compression":
            return ContextCompressionOperator(self.config)
        elif strategy == "summary_compression":
            return SummaryCompressionOperator(self.config)
        elif strategy == "token_compression":
            return TokenCompressionOperator(self.config)

        # Selection/Filtering
        elif strategy == "selection":
            return SelectionOperator(self.config)
        elif strategy == "relevance_filter":
            return RelevanceFilterOperator(self.config)
        elif strategy == "redundancy_filter":
            return RedundancyFilterOperator(self.config)

        # 默认
        else:
            print(f"⚠️  未知的策略: {strategy}，使用默认的 Rerank")
            return RerankOperator(self.config)

    def process(
        self,
        documents: List[Document],
        query: str = None,
        verbose: bool = True
    ) -> List[Document]:
        """
        处理检索结果

        Args:
            documents: 检索到的文档列表
            query: 原始查询
            verbose: 是否打印详细信息

        Returns:
            优化后的文档列表
        """
        if not documents:
            return []

        if verbose:
            print("\n" + "=" * 60)
            print(f"🔧 检索后优化: {self.strategy}")
            print("=" * 60)
            print(f"输入文档数: {len(documents)}")

        result = self.operator.process(documents, query)

        if verbose:
            print(f"输出文档数: {len(result)}")
            print("=" * 60)

        return result

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

    def get_operator(self) -> BasePostRetrievalOperator:
        """获取当前 operator"""
        return self.operator

    def summary(self) -> Dict[str, Any]:
        """
        返回模块摘要信息

        Returns:
            摘要字典
        """
        return {
            "module": "PostRetrievalModule",
            "strategy": self.strategy,
            "operator": self.operator.name,
            "config": self.config,
        }


class PostRetrievalPipeline:
    """
    检索后处理流水线

    支持链式应用多个优化技术

    Example:
        pipeline = PostRetrievalPipeline()
        pipeline.add_step("rerank", {"top_n": 10})
        pipeline.add_step("redundancy_filter")
        pipeline.add_step("context_compression", {"compression_ratio": 0.6})

        optimized = pipeline.process(documents, query)
    """

    def __init__(self):
        """初始化流水线"""
        self.steps: List[PostRetrievalModule] = []

    def add_step(self, strategy: str, config: Dict[str, Any] = None):
        """
        添加处理步骤

        Args:
            strategy: 策略名称
            config: 配置字典
        """
        step_config = config or {}
        step_config["strategy"] = strategy

        module = PostRetrievalModule(step_config)
        self.steps.append(module)

        print(f"✅ 已添加步骤: {strategy}")

    def process(
        self,
        documents: List[Document],
        query: str = None,
        verbose: bool = True
    ) -> List[Document]:
        """
        通过流水线处理文档

        Args:
            documents: 文档列表
            query: 原始查询
            verbose: 是否打印详细信息

        Returns:
            处理后的文档列表
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔄 检索后流水线: {len(self.steps)} 个步骤")
            print("=" * 60)

        current_docs = documents

        for i, step in enumerate(self.steps, 1):
            if verbose:
                print(f"\n--- 步骤 {i}: {step.strategy} ---")
                print(f"输入: {len(current_docs)} 个文档")

            current_docs = step.process(current_docs, query, verbose=False)

            if verbose:
                print(f"输出: {len(current_docs)} 个文档")

        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ 流水线完成")
            print(f"   原始文档数: {len(documents)}")
            print(f"   最终文档数: {len(current_docs)}")
            print("=" * 60)

        return current_docs

    def clear(self):
        """清空流水线"""
        self.steps = []
        print("✅ 流水线已清空")

    def summary(self) -> Dict[str, Any]:
        """返回流水线摘要"""
        return {
            "type": "PostRetrievalPipeline",
            "num_steps": len(self.steps),
            "steps": [step.strategy for step in self.steps],
        }
