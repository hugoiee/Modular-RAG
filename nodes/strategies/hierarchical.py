"""
层次化索引策略 (Hierarchical Indexing)

论文核心思想：
- 将文档组织成父子关系的树形结构
- 每个节点存储数据摘要
- 支持快速数据遍历
- 缓解块提取问题

实现方式：
1. 文档级：整个文档作为根节点
2. 章节级：按标题/段落分割的中间节点
3. 块级：最细粒度的叶子节点

查询时可以：
- 先在文档级摘要中搜索
- 再深入到相关章节
- 最后定位到具体块
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStore


class HierarchicalNode:
    """层次化索引的节点类"""

    def __init__(
        self,
        content: str,
        level: int,
        node_id: str,
        parent_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        self.content = content
        self.level = level  # 0=文档级, 1=章节级, 2=块级
        self.node_id = node_id
        self.parent_id = parent_id
        self.children_ids: List[str] = []
        self.metadata = metadata or {}
        self.summary = ""  # 节点摘要

    def add_child(self, child_id: str):
        """添加子节点"""
        self.children_ids.append(child_id)

    def set_summary(self, summary: str):
        """设置节点摘要"""
        self.summary = summary

    def to_document(self) -> Document:
        """转换为 LangChain Document"""
        metadata = self.metadata.copy()
        metadata.update(
            {
                "node_id": self.node_id,
                "parent_id": self.parent_id,
                "level": self.level,
                "children_ids": self.children_ids,
                "summary": self.summary,
                "hierarchical": True,
            }
        )

        # 如果有摘要，使用摘要作为检索内容，完整内容作为上下文
        if self.summary:
            content = f"摘要: {self.summary}\n\n完整内容: {self.content}"
        else:
            content = self.content

        return Document(page_content=content, metadata=metadata)


class HierarchicalIndexStrategy:
    """
    层次化索引策略

    使用场景：
    - 长文档需要结构化组织
    - 需要多层级检索（粗到细）
    - 需要保留文档结构信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.nodes: Dict[str, HierarchicalNode] = {}  # node_id -> node
        self.root_nodes: List[str] = []  # 根节点 ID 列表

    def build_hierarchy(
        self, documents: List[Document], chunk_size: int = 1000
    ) -> List[Document]:
        """
        构建层次化索引结构

        Args:
            documents: 原始文档列表
            chunk_size: 叶子节点的块大小

        Returns:
            包含层次化元数据的文档列表
        """
        hierarchical_docs = []
        node_counter = 0

        for doc_idx, doc in enumerate(documents):
            # 1. 创建文档级节点（根节点）
            doc_node_id = f"doc_{doc_idx}"
            doc_summary = self._generate_summary(doc.page_content, max_length=200)

            doc_node = HierarchicalNode(
                content=doc.page_content,
                level=0,
                node_id=doc_node_id,
                parent_id=None,
                metadata=doc.metadata.copy(),
            )
            doc_node.set_summary(doc_summary)
            self.nodes[doc_node_id] = doc_node
            self.root_nodes.append(doc_node_id)

            # 2. 将文档分割成章节级节点（中间节点）
            sections = self._split_into_sections(doc.page_content)

            for section_idx, section_content in enumerate(sections):
                section_node_id = f"doc_{doc_idx}_sec_{section_idx}"
                section_summary = self._generate_summary(
                    section_content, max_length=100
                )

                section_node = HierarchicalNode(
                    content=section_content,
                    level=1,
                    node_id=section_node_id,
                    parent_id=doc_node_id,
                    metadata=doc.metadata.copy(),
                )
                section_node.set_summary(section_summary)
                self.nodes[section_node_id] = section_node
                doc_node.add_child(section_node_id)

                # 3. 将章节分割成块级节点（叶子节点）
                chunks = self._split_into_chunks(section_content, chunk_size)

                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk_node_id = f"doc_{doc_idx}_sec_{section_idx}_chunk_{chunk_idx}"

                    chunk_node = HierarchicalNode(
                        content=chunk_content,
                        level=2,
                        node_id=chunk_node_id,
                        parent_id=section_node_id,
                        metadata=doc.metadata.copy(),
                    )
                    self.nodes[chunk_node_id] = chunk_node
                    section_node.add_child(chunk_node_id)

                    # 将叶子节点转换为文档
                    hierarchical_docs.append(chunk_node.to_document())

                # 也添加章节级节点（用于中间层检索）
                hierarchical_docs.append(section_node.to_document())

            # 也添加文档级节点（用于顶层检索）
            hierarchical_docs.append(doc_node.to_document())

        print(f"🌲 层次化索引构建完成：")
        print(f"   - 文档级节点: {len(self.root_nodes)}")
        print(f"   - 总节点数: {len(self.nodes)}")
        print(f"   - 可检索文档数: {len(hierarchical_docs)}")

        return hierarchical_docs

    def _split_into_sections(self, text: str) -> List[str]:
        """
        将文本分割成章节（基于段落）

        Args:
            text: 输入文本

        Returns:
            章节列表
        """
        # 简单实现：按双换行符分割
        sections = text.split("\n\n")
        # 过滤空章节
        sections = [s.strip() for s in sections if s.strip()]

        # 如果没有段落，返回整个文本
        if not sections:
            return [text]

        # 合并过小的段落（少于100字符）
        merged_sections = []
        current_section = ""

        for section in sections:
            if len(current_section) + len(section) < 500:  # 章节最小长度
                current_section += "\n\n" + section if current_section else section
            else:
                if current_section:
                    merged_sections.append(current_section)
                current_section = section

        if current_section:
            merged_sections.append(current_section)

        return merged_sections if merged_sections else [text]

    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        将文本分割成固定大小的块

        Args:
            text: 输入文本
            chunk_size: 块大小

        Returns:
            块列表
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        return chunks if chunks else [text]

    def _generate_summary(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要（简单版本：取前 N 个字符）

        在实际应用中，可以使用 LLM 生成更好的摘要

        Args:
            text: 输入文本
            max_length: 摘要最大长度

        Returns:
            摘要文本
        """
        text = text.strip()
        if len(text) <= max_length:
            return text

        # 简单截断，尽量在句子边界
        summary = text[:max_length]
        # 找到最后一个句号
        last_period = max(
            summary.rfind("。"), summary.rfind(". "), summary.rfind("! "), summary.rfind("? ")
        )

        if last_period > max_length * 0.5:  # 如果句号位置合理
            summary = summary[: last_period + 1]
        else:
            summary += "..."

        return summary

    def get_node(self, node_id: str) -> Optional[HierarchicalNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_parent_context(self, node_id: str) -> str:
        """
        获取节点的父上下文

        Args:
            node_id: 节点 ID

        Returns:
            父节点的内容
        """
        node = self.get_node(node_id)
        if not node or not node.parent_id:
            return ""

        parent = self.get_node(node.parent_id)
        return parent.content if parent else ""

    def get_full_context(self, node_id: str) -> str:
        """
        获取节点的完整上下文（从根到当前节点）

        Args:
            node_id: 节点 ID

        Returns:
            完整上下文
        """
        contexts = []
        current_node = self.get_node(node_id)

        while current_node:
            contexts.insert(0, f"[Level {current_node.level}]\n{current_node.content}")
            if current_node.parent_id:
                current_node = self.get_node(current_node.parent_id)
            else:
                break

        return "\n\n---\n\n".join(contexts)
