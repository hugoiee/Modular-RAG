"""
Compression Operators（压缩）

论文核心技术：
- 减少检索内容以最小化噪音
- 压缩上下文以适应 LLM 窗口限制
- 保留关键信息，去除冗余
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BasePostRetrievalOperator


class ContextCompressionOperator(BasePostRetrievalOperator):
    """
    Context Compression 操作器（上下文压缩）

    功能：
    - 压缩文档内容
    - 只保留与查询最相关的部分
    - 减少 token 使用

    应用场景：
    - 长文档需要精简
    - Token 预算有限
    - 需要提取关键信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_tokens = self.config.get("max_tokens", 200)  # 每个文档最大 token 数
        self.compression_ratio = self.config.get("compression_ratio", 0.5)  # 压缩比例

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        压缩文档内容

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            压缩后的文档列表
        """
        if not documents:
            return []

        print(f"🗜️  Context Compression: 压缩 {len(documents)} 个文档...")

        compressed_docs = []
        total_original_tokens = 0
        total_compressed_tokens = 0

        for doc in documents:
            original_length = len(doc.page_content)
            total_original_tokens += original_length

            # 压缩文档
            compressed_content = self._compress_content(doc.page_content, query)

            # 创建新文档
            compressed_doc = Document(
                page_content=compressed_content,
                metadata=doc.metadata.copy()
            )
            compressed_doc.metadata["original_length"] = original_length
            compressed_doc.metadata["compressed_length"] = len(compressed_content)
            compressed_doc.metadata["compression_ratio"] = len(compressed_content) / original_length if original_length > 0 else 0

            compressed_docs.append(compressed_doc)
            total_compressed_tokens += len(compressed_content)

        actual_ratio = total_compressed_tokens / total_original_tokens if total_original_tokens > 0 else 0
        print(f"   ✓ 压缩完成")
        print(f"   原始: {total_original_tokens} tokens")
        print(f"   压缩后: {total_compressed_tokens} tokens")
        print(f"   压缩率: {actual_ratio:.2%}")

        return compressed_docs

    def _compress_content(self, content: str, query: str = None) -> str:
        """
        压缩文档内容

        简单策略：
        1. 如果有查询，提取与查询相关的句子
        2. 否则，保留前 N 个字符

        Args:
            content: 原始内容
            query: 查询

        Returns:
            压缩后的内容
        """
        # 计算目标长度
        target_length = int(len(content) * self.compression_ratio)
        target_length = max(target_length, self.max_tokens)

        if len(content) <= target_length:
            return content

        if query:
            # 提取与查询相关的句子
            sentences = content.split('。')
            query_words = set(query.lower().split())

            # 计算每个句子的相关性
            scored_sentences = []
            for sent in sentences:
                if not sent.strip():
                    continue
                sent_words = set(sent.lower().split())
                overlap = len(query_words & sent_words)
                scored_sentences.append((sent, overlap))

            # 按相关性排序
            scored_sentences.sort(key=lambda x: x[1], reverse=True)

            # 选择最相关的句子直到达到目标长度
            compressed = []
            current_length = 0

            for sent, score in scored_sentences:
                if current_length + len(sent) <= target_length:
                    compressed.append(sent)
                    current_length += len(sent)
                else:
                    break

            return '。'.join(compressed) + '。' if compressed else content[:target_length]
        else:
            # 简单截断
            return content[:target_length] + "..."


class SummaryCompressionOperator(BasePostRetrievalOperator):
    """
    Summary Compression 操作器（摘要压缩）

    功能：
    - 使用 LLM 生成文档摘要
    - 保留关键信息
    - 更智能的压缩方式

    应用场景：
    - 需要高质量压缩
    - 保持语义完整性
    - 对质量要求高
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_summary_length = self.config.get("max_summary_length", 200)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        通过摘要压缩文档

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            摘要后的文档列表
        """
        if not documents:
            return []

        print(f"📝 Summary Compression: 生成 {len(documents)} 个文档的摘要...")

        summarized_docs = []

        for i, doc in enumerate(documents, 1):
            print(f"   处理文档 {i}/{len(documents)}...")

            # 生成摘要
            summary = self._generate_summary(doc.page_content, query)

            # 创建新文档
            summarized_doc = Document(
                page_content=summary,
                metadata=doc.metadata.copy()
            )
            summarized_doc.metadata["original_length"] = len(doc.page_content)
            summarized_doc.metadata["summary_length"] = len(summary)
            summarized_doc.metadata["is_summary"] = True

            summarized_docs.append(summarized_doc)

        print(f"   ✓ 摘要生成完成")

        return summarized_docs

    def _generate_summary(self, content: str, query: str = None) -> str:
        """
        生成文档摘要

        Args:
            content: 原始内容
            query: 查询（用于引导摘要）

        Returns:
            摘要文本
        """
        # 如果内容已经很短，直接返回
        if len(content) <= self.max_summary_length:
            return content

        if query:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的文档摘要助手。请根据查询生成文档的简洁摘要。

要求：
1. 重点关注与查询相关的信息
2. 保留关键事实和数据
3. 摘要长度不超过 {max_length} 字
4. 保持客观准确
5. 直接输出摘要，不需要前缀

示例：
查询：美国科技行业现状
文档：[长文档内容]
摘要：美国科技行业近期出现大规模裁员潮，多家巨头公司削减员工。主要原因包括疫情期间过度扩张、AI投资巨大但盈利不及预期等。"""),
                ("human", "查询：{query}\n\n文档：{content}\n\n摘要："),
            ])
        else:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个专业的文档摘要助手。请生成文档的简洁摘要。

要求：
1. 提取核心信息和关键要点
2. 摘要长度不超过 {max_length} 字
3. 保持客观准确
4. 直接输出摘要，不需要前缀"""),
                ("human", "文档：{content}\n\n摘要："),
            ])

        chain = prompt | self.llm | StrOutputParser()

        try:
            summary = chain.invoke({
                "query": query,
                "content": content[:2000],  # 限制输入长度
                "max_length": self.max_summary_length
            }).strip()

            return summary
        except Exception as e:
            print(f"   ⚠️  摘要生成失败: {e}，使用截断")
            return content[:self.max_summary_length] + "..."


class TokenCompressionOperator(BasePostRetrievalOperator):
    """
    Token Compression 操作器（Token 级压缩）

    功能：
    - 移除不重要的 token（冠词、介词等）
    - 保留关键词和实体
    - 类似 LLMLingua 的思想

    应用场景：
    - 极致压缩需求
    - Token 预算非常有限
    - 关键词检索场景
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.compression_ratio = self.config.get("compression_ratio", 0.6)

        # 中文停用词（简化版）
        self.stopwords = set([
            '的', '了', '是', '在', '和', '与', '等', '及', '也', '都',
            '就', '而', '将', '被', '把', '给', '从', '向', '到', '为',
            '以', '于', '对', '着', '之', '这', '那', '有', '个', '和',
        ])

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        Token 级压缩

        Args:
            documents: 文档列表
            query: 原始查询

        Returns:
            压缩后的文档列表
        """
        if not documents:
            return []

        print(f"🔤 Token Compression: Token 级压缩 {len(documents)} 个文档...")

        compressed_docs = []
        total_original = 0
        total_compressed = 0

        for doc in documents:
            original_length = len(doc.page_content)
            total_original += original_length

            # 压缩
            compressed_content = self._compress_tokens(doc.page_content, query)

            # 创建新文档
            compressed_doc = Document(
                page_content=compressed_content,
                metadata=doc.metadata.copy()
            )
            compressed_doc.metadata["original_length"] = original_length
            compressed_doc.metadata["compressed_length"] = len(compressed_content)

            compressed_docs.append(compressed_doc)
            total_compressed += len(compressed_content)

        actual_ratio = total_compressed / total_original if total_original > 0 else 0
        print(f"   ✓ Token 压缩完成")
        print(f"   压缩率: {actual_ratio:.2%}")

        return compressed_docs

    def _compress_tokens(self, content: str, query: str = None) -> str:
        """
        移除不重要的 token

        Args:
            content: 原始内容
            query: 查询

        Returns:
            压缩后的内容
        """
        # 简单分词（按空格和标点）
        import re
        tokens = re.findall(r'[\w]+|[^\w\s]', content)

        # 提取查询关键词
        query_keywords = set(query.split()) if query else set()

        # 评估每个 token 的重要性
        important_tokens = []

        for token in tokens:
            # 保留条件：
            # 1. 不是停用词
            # 2. 长度 > 1
            # 3. 或者在查询中出现
            if (
                token not in self.stopwords and
                len(token) > 1
            ) or token in query_keywords:
                important_tokens.append(token)

        # 重组文本
        compressed = ''.join(important_tokens)

        # 如果压缩率不够，进一步压缩
        target_length = int(len(content) * self.compression_ratio)
        if len(compressed) > target_length:
            compressed = compressed[:target_length]

        return compressed


class AdaptiveCompressionOperator(BasePostRetrievalOperator):
    """
    Adaptive Compression 操作器（自适应压缩）

    功能：
    - 根据文档长度和相关性动态选择压缩策略
    - 高相关 + 短文档 -> 不压缩
    - 高相关 + 长文档 -> 摘要压缩
    - 低相关文档 -> 强力压缩或移除

    应用场景：
    - 智能压缩
    - 平衡质量和效率
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.short_threshold = self.config.get("short_threshold", 200)
        self.long_threshold = self.config.get("long_threshold", 1000)

    def process(self, documents: List[Document], query: str = None) -> List[Document]:
        """
        自适应压缩文档

        Args:
            documents: 文档列表（假设按相关性排序）
            query: 原始查询

        Returns:
            压缩后的文档列表
        """
        if not documents:
            return []

        print(f"🎯 Adaptive Compression: 自适应压缩 {len(documents)} 个文档...")

        processed_docs = []

        for i, doc in enumerate(documents):
            doc_length = len(doc.page_content)

            # 根据位置判断相关性（前面的更相关）
            is_highly_relevant = i < len(documents) * 0.3

            # 决定压缩策略
            if is_highly_relevant and doc_length <= self.short_threshold:
                # 高相关 + 短文档 -> 不压缩
                print(f"   文档 {i+1}: 保持原样（高相关且简短）")
                processed_docs.append(doc)

            elif is_highly_relevant and doc_length > self.long_threshold:
                # 高相关 + 长文档 -> 中等压缩
                print(f"   文档 {i+1}: 中等压缩（高相关但较长）")
                compressed_content = doc.page_content[:int(doc_length * 0.7)]
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata=doc.metadata.copy()
                )
                processed_docs.append(compressed_doc)

            elif not is_highly_relevant and doc_length > self.short_threshold:
                # 低相关 + 长文档 -> 强力压缩
                print(f"   文档 {i+1}: 强力压缩（相关性较低）")
                compressed_content = doc.page_content[:int(doc_length * 0.4)]
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata=doc.metadata.copy()
                )
                processed_docs.append(compressed_doc)

            else:
                # 其他情况 -> 适度压缩
                print(f"   文档 {i+1}: 适度压缩")
                compressed_content = doc.page_content[:int(doc_length * 0.6)]
                compressed_doc = Document(
                    page_content=compressed_content,
                    metadata=doc.metadata.copy()
                )
                processed_docs.append(compressed_doc)

        print(f"   ✓ 自适应压缩完成")

        return processed_docs
