"""
文本分块 Operators
实现论文中提到的多种分块优化策略：
1. Sliding Window（滑动窗口）
2. Metadata Attachment（元数据附加）
3. Small-to-Big（小到大策略）
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from .base import BaseOperator


class SplitterOperator(BaseOperator):
    """文本分块器基类"""

    def execute(self, documents: List[Document]) -> List[Document]:
        """
        对文档列表进行分块

        Args:
            documents: Document 对象列表

        Returns:
            分块后的 Document 对象列表
        """
        raise NotImplementedError


class RecursiveSplitterOperator(SplitterOperator):
    """
    递归字符分块器
    实现滑动窗口（Sliding Window）策略
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 200)  # 滑动窗口重叠
        self.add_start_index = self.config.get("add_start_index", True)
        self.separators = self.config.get("separators", None)

    def execute(self, documents: List[Document]) -> List[Document]:
        """
        使用递归字符分割器对文档进行分块

        Args:
            documents: Document 对象列表

        Returns:
            分块后的 Document 对象列表
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=self.add_start_index,
            separators=self.separators,
        )

        splits = splitter.split_documents(documents)

        # 元数据增强：添加分块信息
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["chunk_size"] = len(split.page_content)
            split.metadata["splitter_type"] = "recursive"

        return splits


class SemanticSplitterOperator(SplitterOperator):
    """
    语义分块器
    基于语义边界进行分块（段落、句子）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        # 使用段落和句子作为分隔符
        self.separators = ["\n\n", "\n", "。", "!", "?", ";", "；", ":", "："]

    def execute(self, documents: List[Document]) -> List[Document]:
        """
        基于语义边界进行分块

        Args:
            documents: Document 对象列表

        Returns:
            分块后的 Document 对象列表
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=self.separators,
        )

        splits = splitter.split_documents(documents)

        # 元数据增强
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["chunk_size"] = len(split.page_content)
            split.metadata["splitter_type"] = "semantic"

        return splits


class SmallToBigSplitterOperator(SplitterOperator):
    """
    Small-to-Big 分块策略（论文核心优化技术）

    核心思想：
    1. 创建小块用于检索（提高检索精度）
    2. 保留大块（父块）用于上下文生成
    3. 检索时使用小块，生成时引用对应的大块

    实现方式：
    - 小块：用于向量化和检索
    - 大块：作为父文档，提供完整上下文
    - 通过 metadata 维护父子关系
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # 小块配置（用于检索）
        self.small_chunk_size = self.config.get("small_chunk_size", 400)
        self.small_chunk_overlap = self.config.get("small_chunk_overlap", 50)

        # 大块配置（用于生成）
        self.big_chunk_size = self.config.get("big_chunk_size", 2000)
        self.big_chunk_overlap = self.config.get("big_chunk_overlap", 200)

    def execute(self, documents: List[Document]) -> List[Document]:
        """
        执行 Small-to-Big 分块策略

        Args:
            documents: Document 对象列表

        Returns:
            小块列表，每个小块的 metadata 中包含父块信息
        """
        # 1. 创建大块（父块）
        big_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.big_chunk_size,
            chunk_overlap=self.big_chunk_overlap,
            add_start_index=True,
        )
        big_chunks = big_splitter.split_documents(documents)

        # 2. 对每个大块再分割成小块
        small_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.small_chunk_size,
            chunk_overlap=self.small_chunk_overlap,
            add_start_index=True,
        )

        all_small_chunks = []
        for big_chunk_id, big_chunk in enumerate(big_chunks):
            # 分割成小块
            small_chunks = small_splitter.split_documents([big_chunk])

            # 为每个小块添加父块信息
            for small_chunk_id, small_chunk in enumerate(small_chunks):
                small_chunk.metadata.update({
                    "chunk_id": f"{big_chunk_id}_{small_chunk_id}",
                    "parent_chunk_id": big_chunk_id,
                    "parent_chunk_content": big_chunk.page_content,  # 保存父块内容
                    "chunk_size": len(small_chunk.page_content),
                    "parent_chunk_size": len(big_chunk.page_content),
                    "splitter_type": "small_to_big",
                    "is_small_chunk": True,  # 标记这是小块
                })
                all_small_chunks.append(small_chunk)

        print(f"📊 Small-to-Big 策略：生成 {len(big_chunks)} 个父块，{len(all_small_chunks)} 个子块")
        return all_small_chunks


class StructureAwareSplitterOperator(SplitterOperator):
    """
    结构感知分块器
    根据文档结构（标题、段落等）进行分块
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        # 优先根据文档结构分割
        self.separators = [
            "\n# ",      # Markdown 一级标题
            "\n## ",     # Markdown 二级标题
            "\n### ",    # Markdown 三级标题
            "\n\n",      # 段落
            "\n",        # 行
            "。",        # 中文句子
            ". ",        # 英文句子
        ]

    def execute(self, documents: List[Document]) -> List[Document]:
        """
        基于文档结构进行分块

        Args:
            documents: Document 对象列表

        Returns:
            分块后的 Document 对象列表
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
            separators=self.separators,
        )

        splits = splitter.split_documents(documents)

        # 元数据增强：尝试识别块的类型
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["chunk_size"] = len(split.page_content)
            split.metadata["splitter_type"] = "structure_aware"

            # 简单的结构识别
            content = split.page_content.strip()
            if content.startswith("# "):
                split.metadata["chunk_type"] = "heading_1"
            elif content.startswith("## "):
                split.metadata["chunk_type"] = "heading_2"
            elif content.startswith("### "):
                split.metadata["chunk_type"] = "heading_3"
            else:
                split.metadata["chunk_type"] = "paragraph"

        return splits
