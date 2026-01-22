"""
Rerank Operators（重排序）

论文核心技术：
- 重新排序检索到的文档块
- 不改变内容，只调整顺序
- 解决"Lost in the middle"问题
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BasePostRetrievalOperator


class RerankOperator(BasePostRetrievalOperator):
    """
    基础 Rerank 操作器（基于规则）

    功能：
    - 基于相似度分数重排序
    - 将最相关的文档放在前面和后面（避免 Lost in the middle）
    - 简单高效

    应用场景：
    - 快速重排序
    - 优化 LLM 上下文
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", None)  # 只保留前 N 个
        self.reverse_order = self.config.get("reverse_order", False)  # 是否反转顺序

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        基于规则重排序文档

        Args:
            documents: 文档列表
            query: 原始查询（本方法不使用）

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        print(f"🔄 Rerank: 重排序 {len(documents)} 个文档...")

        # 假设文档已经按相关性排序（来自检索器）
        # 策略：将最相关的放在首尾，避免 "Lost in the middle"
        reranked = self._reorder_for_llm(documents)

        # 如果指定了 top_n，只保留前 N 个
        if self.top_n and self.top_n < len(reranked):
            reranked = reranked[:self.top_n]
            print(f"   保留前 {self.top_n} 个文档")

        # 是否反转顺序
        if self.reverse_order:
            reranked = list(reversed(reranked))
            print(f"   顺序已反转")

        print(f"   ✓ 重排序完成")

        return reranked

    def _reorder_for_llm(self, documents: List[Document]) -> List[Document]:
        """
        重新排序以优化 LLM 感知

        策略：最相关的放在开头和结尾

        Args:
            documents: 文档列表

        Returns:
            重排序后的文档列表
        """
        if len(documents) <= 2:
            return documents

        # 将文档分成三组
        # 高相关（前1/3）-> 放在开头和结尾
        # 中相关（中间1/3）-> 放在次要位置
        # 低相关（后1/3）-> 放在中间

        n = len(documents)
        high = documents[:n//3]
        mid = documents[n//3:2*n//3]
        low = documents[2*n//3:]

        # 重新组合：高相关的一半在开头，一半在结尾
        half_high = len(high) // 2

        reordered = []
        reordered.extend(high[:half_high])  # 最相关的一部分在开头
        reordered.extend(mid)               # 中等相关的
        reordered.extend(low)               # 低相关的在中间
        reordered.extend(high[half_high:])  # 最相关的另一部分在结尾

        return reordered


class DiversityRerankOperator(BasePostRetrievalOperator):
    """
    Diversity Rerank 操作器（多样性重排序）

    功能：
    - 在保持相关性的同时增加多样性
    - 避免内容过度重复
    - 使用 MMR 类似的策略

    应用场景：
    - 需要多角度信息
    - 避免信息冗余
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.diversity_weight = self.config.get("diversity_weight", 0.5)
        self.top_n = self.config.get("top_n", None)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        基于多样性重排序

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        print(f"🌈 Diversity Rerank: 重排序 {len(documents)} 个文档（多样性权重: {self.diversity_weight}）...")

        selected = []
        remaining = documents.copy()

        # 先选择第一个（最相关的）
        selected.append(remaining.pop(0))

        # 迭代选择，平衡相关性和多样性
        while remaining:
            best_idx = self._select_next(selected, remaining)
            selected.append(remaining.pop(best_idx))

        # 限制数量
        if self.top_n and self.top_n < len(selected):
            selected = selected[:self.top_n]

        print(f"   ✓ 多样性重排序完成")

        return selected

    def _select_next(self, selected: List[Document], remaining: List[Document]) -> int:
        """
        选择下一个文档（平衡相关性和多样性）

        Args:
            selected: 已选择的文档
            remaining: 剩余的文档

        Returns:
            下一个应选择的文档索引
        """
        best_idx = 0
        best_score = -float('inf')

        for idx, doc in enumerate(remaining):
            # 相关性分数（假设按顺序递减）
            relevance_score = 1.0 - (idx / len(remaining))

            # 多样性分数（与已选择文档的差异）
            diversity_score = self._calculate_diversity(doc, selected)

            # 综合分数
            combined_score = (
                (1 - self.diversity_weight) * relevance_score +
                self.diversity_weight * diversity_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        return best_idx

    def _calculate_diversity(self, doc: Document, selected: List[Document]) -> float:
        """
        计算文档与已选择文档的多样性

        Args:
            doc: 候选文档
            selected: 已选择的文档列表

        Returns:
            多样性分数（0-1）
        """
        if not selected:
            return 1.0

        # 简单的多样性度量：内容相似度的倒数
        min_similarity = 1.0

        for sel_doc in selected:
            similarity = self._simple_similarity(doc.page_content, sel_doc.page_content)
            min_similarity = min(min_similarity, similarity)

        # 多样性 = 1 - 最大相似度
        return 1.0 - min_similarity

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """
        简单的文本相似度计算（基于词重叠）

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            相似度分数（0-1）
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class LLMRerankOperator(BasePostRetrievalOperator):
    """
    LLM Rerank 操作器（基于 LLM 的重排序）

    功能：
    - 使用 LLM 评估每个文档与查询的相关性
    - 更准确的相关性判断
    - 适合复杂的语义理解

    应用场景：
    - 需要精确重排序
    - 复杂查询场景
    - 对质量要求高的应用
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)
        self.top_n = self.config.get("top_n", None)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        使用 LLM 重排序文档

        Args:
            documents: 文档列表
            query: 原始查询（必需）

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        if not query:
            print("⚠️  LLM Rerank 需要查询，返回原始顺序")
            return documents

        print(f"🤖 LLM Rerank: 使用 LLM 重排序 {len(documents)} 个文档...")

        # 为每个文档评分
        scored_docs = []
        for i, doc in enumerate(documents):
            score = self._score_document(doc, query)
            scored_docs.append((doc, score))
            print(f"   文档 {i+1}: 相关性分数 = {score:.2f}")

        # 按分数排序（降序）
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        reranked = [doc for doc, score in scored_docs]

        # 限制数量
        if self.top_n and self.top_n < len(reranked):
            reranked = reranked[:self.top_n]

        print(f"   ✓ LLM 重排序完成")

        return reranked

    def _score_document(self, doc: Document, query: str) -> float:
        """
        使用 LLM 评估文档相关性

        Args:
            doc: 文档
            query: 查询

        Returns:
            相关性分数（0-10）
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个文档相关性评估专家。给定一个查询和一个文档，评估文档与查询的相关性。

评分标准（0-10分）：
- 10分：完美匹配，直接回答查询
- 7-9分：高度相关，包含关键信息
- 4-6分：部分相关，有一些相关内容
- 1-3分：弱相关，只有少量关联
- 0分：完全不相关

只输出一个数字分数（0-10），不需要解释。"""),
            ("human", "查询：{query}\n\n文档：{document}\n\n相关性分数："),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "document": doc.page_content[:500]  # 只使用前500字符
            }).strip()

            # 提取数字
            score = float(result.split()[0])
            return max(0.0, min(10.0, score))  # 限制在 0-10 范围
        except Exception as e:
            print(f"   ⚠️  评分失败: {e}，使用默认分数 5.0")
            return 5.0


class LostInMiddleRerankOperator(BasePostRetrievalOperator):
    """
    Lost-in-Middle Aware Rerank 操作器

    专门解决"Lost in the middle"问题：
    - LLM 倾向于记住开头和结尾的信息
    - 将最重要的文档放在这些位置

    策略：
    最相关 -> 开头
    次相关 -> 结尾
    其他 -> 中间
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.top_n = self.config.get("top_n", None)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        优化文档顺序以应对 Lost in the middle

        Args:
            documents: 文档列表（假设已按相关性排序）
            query: 原始查询

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        if len(documents) <= 2:
            return documents

        print(f"📍 Lost-in-Middle Rerank: 优化 {len(documents)} 个文档的位置...")

        # 策略：奇数索引放开头，偶数索引放结尾
        reordered = []
        left = []
        right = []

        for i, doc in enumerate(documents):
            if i % 2 == 0:
                left.append(doc)  # 偶数索引 -> 开头
            else:
                right.append(doc)  # 奇数索引 -> 结尾

        # 组合：开头 + 结尾（反转）
        reordered = left + right[::-1]

        # 限制数量
        if self.top_n and self.top_n < len(reordered):
            reordered = reordered[:self.top_n]

        print(f"   ✓ 位置优化完成（最相关的在首尾）")

        return reordered
