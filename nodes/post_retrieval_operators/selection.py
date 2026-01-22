"""
Selection/Filtering Operators（选择/过滤）

论文核心技术：
- 直接移除不相关的文档块
- 过滤噪音和冗余信息
- 确保只有高质量文档被传递给 LLM
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BasePostRetrievalOperator


class SelectionOperator(BasePostRetrievalOperator):
    """
    基础 Selection 操作器

    功能：
    - 只保留前 N 个文档
    - 简单快速的过滤策略

    应用场景：
    - 快速限制文档数量
    - 减少 LLM 上下文负担
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.top_k = self.config.get("top_k", 5)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        选择前 K 个文档

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            选择后的文档列表
        """
        if not documents:
            return []

        print(f"✂️  Selection: 选择前 {self.top_k} 个文档...")

        selected = documents[:self.top_k]

        print(f"   ✓ 从 {len(documents)} 个文档中选择了 {len(selected)} 个")

        return selected


class RelevanceFilterOperator(BasePostRetrievalOperator):
    """
    Relevance Filter 操作器（相关性过滤）

    功能：
    - 使用 LLM 判断文档与查询的相关性
    - 移除不相关的文档
    - 提高文档质量

    应用场景：
    - 需要精确过滤
    - 对质量要求高
    - 避免误导信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)
        self.relevance_threshold = self.config.get("relevance_threshold", 0.5)
        self.min_docs = self.config.get("min_docs", 1)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        基于相关性过滤文档

        Args:
            documents: 文档列表
            query: 原始查询（必需）

        Returns:
            过滤后的文档列表
        """
        if not documents:
            return []

        if not query:
            print("⚠️  Relevance Filter 需要查询，返回所有文档")
            return documents

        print(f"🎯 Relevance Filter: 过滤 {len(documents)} 个文档（阈值: {self.relevance_threshold}）...")

        filtered_docs = []

        for i, doc in enumerate(documents, 1):
            is_relevant = self._check_relevance(doc, query)

            if is_relevant:
                filtered_docs.append(doc)
                print(f"   ✓ 文档 {i}: 相关")
            else:
                print(f"   ✗ 文档 {i}: 不相关（已过滤）")

        # 确保至少保留 min_docs 个文档
        if len(filtered_docs) < self.min_docs and len(documents) > 0:
            print(f"   ⚠️  过滤结果少于 {self.min_docs} 个，保留前 {self.min_docs} 个")
            filtered_docs = documents[:self.min_docs]

        print(f"   ✓ 过滤完成，保留 {len(filtered_docs)}/{len(documents)} 个文档")

        return filtered_docs

    def _check_relevance(self, doc: Document, query: str) -> bool:
        """
        检查文档是否与查询相关

        Args:
            doc: 文档
            query: 查询

        Returns:
            是否相关
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个文档相关性判断专家。判断文档是否与查询相关。

判断标准：
- 相关：文档包含查询所需的信息，能帮助回答问题
- 不相关：文档内容与查询主题无关，不能提供有价值的信息

只输出"相关"或"不相关"，不需要解释。"""),
            ("human", "查询：{query}\n\n文档：{document}\n\n判断："),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "query": query,
                "document": doc.page_content[:500]  # 只使用前500字符
            }).strip()

            return "相关" in result or "relevant" in result.lower()
        except Exception as e:
            print(f"   ⚠️  判断失败: {e}，默认保留")
            return True


class RedundancyFilterOperator(BasePostRetrievalOperator):
    """
    Redundancy Filter 操作器（冗余过滤）

    功能：
    - 检测并移除重复或高度相似的文档
    - 减少冗余信息
    - 提高信息多样性

    应用场景：
    - 去除重复内容
    - 优化上下文利用率
    - 提供多样化信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.85)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        过滤冗余文档

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            去重后的文档列表
        """
        if not documents:
            return []

        print(f"🔍 Redundancy Filter: 检测冗余文档（阈值: {self.similarity_threshold}）...")

        filtered_docs = []
        filtered_docs.append(documents[0])  # 保留第一个

        for i, doc in enumerate(documents[1:], 2):
            # 检查与已选择文档的相似度
            is_redundant = False

            for selected_doc in filtered_docs:
                similarity = self._calculate_similarity(
                    doc.page_content,
                    selected_doc.page_content
                )

                if similarity >= self.similarity_threshold:
                    is_redundant = True
                    print(f"   ✗ 文档 {i}: 冗余（相似度: {similarity:.2f}）")
                    break

            if not is_redundant:
                filtered_docs.append(doc)
                print(f"   ✓ 文档 {i}: 保留")

        print(f"   ✓ 冗余过滤完成，保留 {len(filtered_docs)}/{len(documents)} 个文档")

        return filtered_docs

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（简单的词重叠）

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


class QualityFilterOperator(BasePostRetrievalOperator):
    """
    Quality Filter 操作器（质量过滤）

    功能：
    - 评估文档质量
    - 移除低质量文档
    - 确保信息可靠性

    质量标准：
    - 文档长度合理
    - 包含完整信息
    - 语言规范
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_length = self.config.get("min_length", 50)
        self.max_length = self.config.get("max_length", 5000)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        基于质量过滤文档

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            高质量文档列表
        """
        if not documents:
            return []

        print(f"⭐ Quality Filter: 过滤低质量文档...")

        filtered_docs = []

        for i, doc in enumerate(documents, 1):
            quality_score = self._assess_quality(doc)

            if quality_score >= 0.5:  # 质量阈值
                filtered_docs.append(doc)
                print(f"   ✓ 文档 {i}: 质量良好（{quality_score:.2f}）")
            else:
                print(f"   ✗ 文档 {i}: 质量不佳（{quality_score:.2f}）")

        print(f"   ✓ 质量过滤完成，保留 {len(filtered_docs)}/{len(documents)} 个文档")

        return filtered_docs

    def _assess_quality(self, doc: Document) -> float:
        """
        评估文档质量

        Args:
            doc: 文档

        Returns:
            质量分数（0-1）
        """
        content = doc.page_content
        score = 0.0

        # 1. 长度检查（30分）
        length = len(content)
        if self.min_length <= length <= self.max_length:
            score += 0.3
        elif length < self.min_length:
            score += 0.1  # 太短扣分
        else:
            score += 0.2  # 太长也扣分

        # 2. 完整性检查（30分）
        # 简单启发式：包含标点符号和完整句子
        has_periods = '。' in content or '.' in content
        has_multiple_sentences = content.count('。') > 1 or content.count('.') > 1

        if has_periods and has_multiple_sentences:
            score += 0.3
        elif has_periods:
            score += 0.15

        # 3. 信息密度检查（20分）
        # 词汇多样性
        words = content.split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0

        if diversity > 0.5:
            score += 0.2
        elif diversity > 0.3:
            score += 0.1

        # 4. 格式检查（20分）
        # 避免过多特殊字符或重复
        special_char_ratio = sum(1 for c in content if not c.isalnum() and c not in '。，！？、；：""''（）《》\n ') / len(content) if content else 0

        if special_char_ratio < 0.1:
            score += 0.2
        elif special_char_ratio < 0.2:
            score += 0.1

        return min(score, 1.0)


class ContradictionFilterOperator(BasePostRetrievalOperator):
    """
    Contradiction Filter 操作器（矛盾过滤）

    功能：
    - 检测文档间的矛盾信息
    - 移除矛盾或冲突的文档
    - 提高信息一致性

    应用场景：
    - 事实性查询
    - 需要一致性的场景
    - 避免误导信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.0)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        过滤矛盾文档

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            无矛盾的文档列表
        """
        if not documents or len(documents) < 2:
            return documents

        print(f"⚖️  Contradiction Filter: 检测矛盾文档...")

        # 保留第一个文档作为基准
        filtered_docs = [documents[0]]

        for i, doc in enumerate(documents[1:], 2):
            # 检查与已选择文档是否矛盾
            has_contradiction = self._check_contradiction(doc, filtered_docs)

            if not has_contradiction:
                filtered_docs.append(doc)
                print(f"   ✓ 文档 {i}: 无矛盾")
            else:
                print(f"   ✗ 文档 {i}: 存在矛盾（已过滤）")

        print(f"   ✓ 矛盾检测完成，保留 {len(filtered_docs)}/{len(documents)} 个文档")

        return filtered_docs

    def _check_contradiction(self, doc: Document, reference_docs: List[Document]) -> bool:
        """
        检查文档是否与参考文档矛盾

        Args:
            doc: 待检查文档
            reference_docs: 参考文档列表

        Returns:
            是否存在矛盾
        """
        # 简化版：只检查与第一个文档的矛盾
        if not reference_docs:
            return False

        reference_content = reference_docs[0].page_content[:300]
        doc_content = doc.page_content[:300]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个事实一致性检查专家。判断两段文本是否存在事实矛盾。

矛盾的定义：
- 两段文本对同一事实给出了不同的描述
- 数字、日期、人名等关键信息冲突
- 结论或观点完全相反

只输出"矛盾"或"无矛盾"，不需要解释。"""),
            ("human", "文本1：{text1}\n\n文本2：{text2}\n\n判断："),
        ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({
                "text1": reference_content,
                "text2": doc_content
            }).strip()

            return "矛盾" in result or "contradiction" in result.lower()
        except Exception as e:
            print(f"   ⚠️  矛盾检测失败: {e}，默认保留")
            return False
