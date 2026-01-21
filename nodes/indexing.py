from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

def load_documents(pdf_directory: str) -> list[Document]:
    """
    加载指定目录下的所有PDF文件。
    """
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    all_docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        all_docs.extend(docs)

    print(f"加载的文档数量为: {len(all_docs)}")
    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """
    将文档分割成较小的段落。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)
    print(f"分割的块有：{len(all_splits)}个")
    return all_splits


def create_embeddings() -> DashScopeEmbeddings:
    """
    创建嵌入模型。
    """
    embeddings_model = DashScopeEmbeddings(
        model="text-embedding-v4",
    )
    return embeddings_model


def store_vectors(documents: list[Document], embeddings: DashScopeEmbeddings, persist_directory: str) -> None:
    """
    将向量存储到向量数据库中。
    """
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print(f"✅ 向量数据库创建成功！")
    print(f"存储了 {len(documents)} 个文档块")
    print(f"持久化路径: {persist_directory}")


def indexing():
    """
    索引模块
    """
    pdf_directory = "../doc/金融新闻pdf/"
    persist_directory = "../chroma_db"

    # --- 加载文档、切片、向量、存储 ---
    documents = load_documents(pdf_directory)
    splits = split_documents(documents)
    embeddings = create_embeddings()
    store_vectors(splits, embeddings, persist_directory)


if __name__ == "__main__":
    indexing()