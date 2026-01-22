"""
Sparse Retrieval Operators（稀疏检索/关键词检索）

论文核心技术：
- 使用统计方法（TF-IDF, BM25）
- 基于关键词匹配
- 计算效率高，适合大规模数据集
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever, TFIDFRetriever
from .base import BaseRetrievalOperator


class BM25RetrieverOperator(BaseRetrievalOperator):
    """
    BM25 Retriever 操作器

    功能：
    - 使用 BM25 算法进行关键词检索
    - 基于词频和逆文档频率
    - 对长度归一化

    优势：
    - 计算效率高
    - 适合大规模数据集
    - 对精确关键词匹配效果好
    - 零样本检索能力强

    应用场景：
    - 需要精确关键词匹配
    - 大规模文档检索
    - 计算资源受限
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.documents: List[Document] = []

    def build_retriever(self, documents: List[Document] = None, **kwargs) -> BaseRetriever:
        """
        从文档列表构建 BM25 检索器

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            BM25 检索器实例
        """
        if documents is None or len(documents) == 0:
            raise ValueError("需要提供文档列表")

        self.documents = documents
        k = kwargs.get("k", self.k)

        # 创建 BM25 检索器
        self.retriever = BM25Retriever.from_documents(
            documents=self.documents,
            k=k,
        )

        print(f"✅ BM25 Retriever 已构建")
        print(f"   - 文档数量: {len(self.documents)}")
        print(f"   - 返回数量: {k}")

        return self.retriever

    def update_documents(self, new_documents: List[Document]):
        """
        更新文档集合并重建检索器

        Args:
            new_documents: 新的文档列表
        """
        self.documents.extend(new_documents)
        self.build_retriever(documents=self.documents, k=self.k)
        print(f"✅ 文档已更新，当前文档数量: {len(self.documents)}")


class TFIDFRetrieverOperator(BaseRetrievalOperator):
    """
    TF-IDF Retriever 操作器

    功能：
    - 使用 TF-IDF 算法进行关键词检索
    - 基于词频-逆文档频率统计
    - 经典的信息检索方法

    优势：
    - 简单高效
    - 适合关键词搜索
    - 可解释性强

    应用场景：
    - 传统关键词检索
    - 需要快速原型验证
    - 基线方法对比
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.documents: List[Document] = []

    def build_retriever(self, documents: List[Document] = None, **kwargs) -> BaseRetriever:
        """
        从文档列表构建 TF-IDF 检索器

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            TF-IDF 检索器实例
        """
        if documents is None or len(documents) == 0:
            raise ValueError("需要提供文档列表")

        self.documents = documents
        k = kwargs.get("k", self.k)

        # 创建 TF-IDF 检索器
        self.retriever = TFIDFRetriever.from_documents(
            documents=self.documents,
            k=k,
        )

        print(f"✅ TF-IDF Retriever 已构建")
        print(f"   - 文档数量: {len(self.documents)}")
        print(f"   - 返回数量: {k}")

        return self.retriever


class KeywordRetrieverOperator(BaseRetrievalOperator):
    """
    Keyword Retriever 操作器（简单关键词匹配）

    功能：
    - 基于简单的关键词包含判断
    - 支持多个关键词的 AND/OR 逻辑
    - 最基础的检索方法

    优势：
    - 实现简单
    - 速度极快
    - 完全可控

    应用场景：
    - 规则化检索
    - 快速过滤
    - 辅助检索
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.case_sensitive = self.config.get("case_sensitive", False)
        self.match_mode = self.config.get("match_mode", "any")  # any 或 all
        self.documents: List[Document] = []

    def build_retriever(self, documents: List[Document] = None, **kwargs) -> BaseRetriever:
        """
        构建关键词检索器（实际上不需要 build，直接使用 retrieve）

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            None（使用自定义 retrieve 方法）
        """
        if documents is None or len(documents) == 0:
            raise ValueError("需要提供文档列表")

        self.documents = documents

        print(f"✅ Keyword Retriever 已构建")
        print(f"   - 文档数量: {len(self.documents)}")
        print(f"   - 匹配模式: {self.match_mode}")
        print(f"   - 大小写敏感: {self.case_sensitive}")

        # 返回 None，使用自定义 retrieve 方法
        self.retriever = None
        return None

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        基于关键词检索文档

        Args:
            query: 查询字符串
            **kwargs: 检索参数

        Returns:
            匹配的文档列表
        """
        if not self.documents:
            return []

        k = kwargs.get("k", self.k)
        match_mode = kwargs.get("match_mode", self.match_mode)

        # 提取关键词（简单分词）
        keywords = query.split()

        # 处理大小写
        if not self.case_sensitive:
            keywords = [kw.lower() for kw in keywords]

        matched_docs = []

        for doc in self.documents:
            content = doc.page_content
            if not self.case_sensitive:
                content = content.lower()

            # 检查匹配
            if match_mode == "all":
                # 所有关键词都要匹配
                if all(kw in content for kw in keywords):
                    matched_docs.append(doc)
            else:
                # 任意关键词匹配
                if any(kw in content for kw in keywords):
                    matched_docs.append(doc)

            # 达到数量限制就停止
            if len(matched_docs) >= k:
                break

        return matched_docs[:k]


class RegexRetrieverOperator(BaseRetrievalOperator):
    """
    Regex Retriever 操作器（正则表达式检索）

    功能：
    - 使用正则表达式进行模式匹配
    - 支持复杂的匹配规则
    - 适合结构化文本检索

    应用场景：
    - 需要精确模式匹配
    - 结构化信息提取
    - 特定格式的文档检索
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.k = self.config.get("k", 5)
        self.documents: List[Document] = []

    def build_retriever(self, documents: List[Document] = None, **kwargs) -> BaseRetriever:
        """
        构建正则表达式检索器

        Args:
            documents: 文档列表
            **kwargs: 额外参数

        Returns:
            None（使用自定义 retrieve 方法）
        """
        if documents is None or len(documents) == 0:
            raise ValueError("需要提供文档列表")

        self.documents = documents

        print(f"✅ Regex Retriever 已构建")
        print(f"   - 文档数量: {len(self.documents)}")

        self.retriever = None
        return None

    def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        使用正则表达式检索文档

        Args:
            query: 正则表达式模式
            **kwargs: 检索参数

        Returns:
            匹配的文档列表
        """
        import re

        if not self.documents:
            return []

        k = kwargs.get("k", self.k)

        try:
            pattern = re.compile(query)
        except re.error as e:
            print(f"❌ 正则表达式错误: {e}")
            return []

        matched_docs = []

        for doc in self.documents:
            if pattern.search(doc.page_content):
                matched_docs.append(doc)

            if len(matched_docs) >= k:
                break

        return matched_docs[:k]
