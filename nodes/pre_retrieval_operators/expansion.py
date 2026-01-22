"""
Query Expansion Operators（查询扩展）

论文核心技术：
1. Multi-Query: 生成多个并行查询变体
2. Sub-Query: 将复杂问题分解为多个子问题
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BasePreRetrievalOperator


class MultiQueryOperator(BasePreRetrievalOperator):
    """
    Multi-Query 操作器

    功能：
    - 从不同角度生成多个查询变体
    - 提高检索召回率
    - 减少因查询表述不当导致的遗漏

    原理：
    使用 LLM 生成原始查询的多个语义相似但表述不同的版本
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.num_queries = self.config.get("num_queries", 3)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.7)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> List[str]:
        """
        生成多个查询变体

        Args:
            query: 原始查询

        Returns:
            查询列表（包含原始查询和变体）
        """
        print(f"🔄 Multi-Query: 正在生成 {self.num_queries} 个查询变体...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化助手。给定一个用户查询，生成 {num_queries} 个语义相似但表述不同的查询变体。

要求：
1. 从不同角度重新表述原始查询
2. 保持核心语义不变
3. 使用不同的关键词和表达方式
4. 每个查询一行，不需要编号

示例：
原始查询：什么是机器学习？
变体1：机器学习的定义是什么？
变体2：请解释一下机器学习的概念
变体3：机器学习指的是什么技术？"""),
            ("human", "原始查询：{query}\n\n请生成 {num_queries} 个查询变体："),
        ])

        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({
            "query": query,
            "num_queries": self.num_queries
        })

        # 解析生成的查询
        queries = [line.strip() for line in result.strip().split("\n") if line.strip()]

        # 过滤掉可能的编号
        queries = [q.split(":", 1)[-1].strip() if ":" in q else q for q in queries]

        # 确保包含原始查询
        if query not in queries:
            queries.insert(0, query)

        # 限制数量
        queries = queries[:self.num_queries + 1]

        print(f"   ✓ 生成了 {len(queries)} 个查询（包含原始查询）")
        for i, q in enumerate(queries, 1):
            print(f"      {i}. {q}")

        return queries


class SubQueryOperator(BasePreRetrievalOperator):
    """
    Sub-Query 操作器（查询分解）

    功能：
    - 将复杂问题分解为多个简单子问题
    - 适用于多步推理场景
    - 每个子问题独立检索，最后综合答案

    原理：
    使用 LLM 分析查询的复杂度，将其分解为可独立回答的子问题
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_sub_queries = self.config.get("max_sub_queries", 4)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> List[str]:
        """
        将查询分解为子查询

        Args:
            query: 原始复杂查询

        Returns:
            子查询列表
        """
        print(f"🔍 Sub-Query: 正在分解复杂查询...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询分解专家。给定一个复杂的用户查询，将其分解为多个简单的子查询。

分解原则：
1. 识别查询中的多个信息需求
2. 每个子查询应该是独立的、可以单独回答的
3. 子查询应该按逻辑顺序排列
4. 避免过度分解（2-4个子查询为佳）
5. 每个子查询一行，不需要编号

示例1：
原始查询：比较Python和Java在机器学习领域的应用，并分析各自的优缺点。
子查询1：Python在机器学习领域有哪些应用？
子查询2：Java在机器学习领域有哪些应用？
子查询3：Python用于机器学习的优缺点是什么？
子查询4：Java用于机器学习的优缺点是什么？

示例2：
原始查询：美国科技行业的投资风险如何？
子查询1：美国科技行业的当前状况如何？
子查询2：美国科技行业存在哪些投资风险？"""),
            ("human", "原始查询：{query}\n\n请分解为子查询："),
        ])

        chain = prompt | self.llm | StrOutputParser()

        result = chain.invoke({"query": query})

        # 解析子查询
        sub_queries = [line.strip() for line in result.strip().split("\n") if line.strip()]

        # 过滤掉可能的编号或前缀
        sub_queries = [q.split(":", 1)[-1].strip() if ":" in q else q for q in sub_queries]

        # 限制数量
        sub_queries = sub_queries[:self.max_sub_queries]

        # 如果没有成功分解（只有1个或没有），返回原始查询
        if len(sub_queries) <= 1:
            print(f"   ℹ️  查询无需分解，使用原始查询")
            return [query]

        print(f"   ✓ 分解为 {len(sub_queries)} 个子查询：")
        for i, sq in enumerate(sub_queries, 1):
            print(f"      {i}. {sq}")

        return sub_queries


class HybridExpansionOperator(BasePreRetrievalOperator):
    """
    混合扩展操作器

    结合 Multi-Query 和 Sub-Query 的优势：
    1. 先判断查询是否复杂
    2. 复杂查询进行分解
    3. 简单查询生成变体
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.multi_query = MultiQueryOperator(config)
        self.sub_query = SubQueryOperator(config)
        self.complexity_threshold = self.config.get("complexity_threshold", 0.6)

    def _estimate_complexity(self, query: str) -> float:
        """
        估计查询的复杂度

        简单规则：
        - 包含"比较"、"分析"、"并且"等词 -> 复杂
        - 包含多个问号 -> 复杂
        - 字数过长 -> 复杂
        """
        complexity_keywords = ["比较", "分析", "对比", "综合", "评估", "并且", "以及"]
        complexity_score = 0.0

        # 关键词检测
        for keyword in complexity_keywords:
            if keyword in query:
                complexity_score += 0.3

        # 多个问号
        if query.count("？") > 1 or query.count("?") > 1:
            complexity_score += 0.3

        # 长度检测
        if len(query) > 50:
            complexity_score += 0.2

        return min(complexity_score, 1.0)

    def execute(self, query: str) -> List[str]:
        """
        智能选择扩展策略

        Args:
            query: 原始查询

        Returns:
            扩展后的查询列表
        """
        complexity = self._estimate_complexity(query)
        print(f"📊 查询复杂度评估: {complexity:.2f}")

        if complexity >= self.complexity_threshold:
            print(f"   → 使用 Sub-Query 策略（复杂查询分解）")
            return self.sub_query.execute(query)
        else:
            print(f"   → 使用 Multi-Query 策略（查询变体生成）")
            return self.multi_query.execute(query)
