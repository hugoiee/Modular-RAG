"""
文档加载 Operators
支持多种文档格式的加载
"""

import os
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .base import BaseOperator


class LoaderOperator(BaseOperator):
    """文档加载器基类"""

    def execute(self, file_path: Union[str, List[str]]) -> List[Document]:
        """
        加载文档

        Args:
            file_path: 文件路径或文件路径列表

        Returns:
            Document 对象列表
        """
        if isinstance(file_path, str):
            return self._load_single_file(file_path)
        else:
            all_docs = []
            for path in file_path:
                all_docs.extend(self._load_single_file(path))
            return all_docs

    def _load_single_file(self, file_path: str) -> List[Document]:
        """加载单个文件，子类需实现"""
        raise NotImplementedError


class PDFLoaderOperator(LoaderOperator):
    """
    PDF 文档加载器
    使用 PyPDFLoader 加载 PDF 文件
    """

    def _load_single_file(self, file_path: str) -> List[Document]:
        """
        加载单个 PDF 文件

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象列表
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 元数据增强：添加文件名
        for doc in docs:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["file_type"] = "pdf"

        return docs


class TextLoaderOperator(LoaderOperator):
    """
    文本文档加载器
    支持 .txt, .md 等文本文件
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.encoding = config.get("encoding", "utf-8") if config else "utf-8"

    def _load_single_file(self, file_path: str) -> List[Document]:
        """
        加载单个文本文件

        Args:
            file_path: 文本文件路径

        Returns:
            Document 对象列表
        """
        loader = TextLoader(file_path, encoding=self.encoding)
        docs = loader.load()

        # 元数据增强
        for doc in docs:
            doc.metadata["filename"] = os.path.basename(file_path)
            doc.metadata["file_type"] = "text"

        return docs


class DirectoryLoaderOperator(LoaderOperator):
    """
    目录批量加载器
    自动识别目录中的文件类型并加载
    """

    def __init__(self, config: dict = None):
        super().__init__(config)
        self.file_extensions = config.get("file_extensions", [".pdf", ".txt", ".md"]) if config else [".pdf", ".txt", ".md"]

    def execute(self, directory_path: str) -> List[Document]:
        """
        加载目录中的所有文档

        Args:
            directory_path: 目录路径

        Returns:
            Document 对象列表
        """
        all_docs = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                if ext in self.file_extensions:
                    try:
                        if ext == ".pdf":
                            loader = PDFLoaderOperator()
                        elif ext in [".txt", ".md"]:
                            loader = TextLoaderOperator({"encoding": "utf-8"})
                        else:
                            continue

                        docs = loader.execute(file_path)
                        all_docs.extend(docs)
                        print(f"✅ 已加载: {file_path} ({len(docs)} 个文档)")
                    except Exception as e:
                        print(f"❌ 加载失败: {file_path}, 错误: {e}")

        return all_docs

    def _load_single_file(self, file_path: str) -> List[Document]:
        """该方法在 DirectoryLoader 中不使用"""
        pass
