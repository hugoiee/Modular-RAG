"""
索引模块使用示例

演示如何使用 IndexModule 的不同配置和策略
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.indexing import IndexModule

# 加载环境变量
load_dotenv()


def example_1_basic_indexing():
    """示例 1: 基础索引（递归分块 + Chroma）"""
    print("\n" + "=" * 70)
    print("示例 1: 基础索引（递归分块）")
    print("=" * 70)

    config = {
        "loader": {"type": "web", "file_extensions": []},
        "splitter": {
            "type": "recursive",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/chroma_basic",
            "collection_name": "basic_index"
        },
    }

    index_module = IndexModule(config)
    # vectorstore = index_module.index_documents("./doc/",False)
    vectorstore = index_module.index_documents("https://lilianweng.github.io/posts/2023-06-23-agent/", True)

    # 测试检索
    retriever = index_module.get_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    results = retriever.invoke("What is Agent")
    print(f"\n🔍 检索结果: 找到 {len(results)} 个相关文档")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"内容: {doc.page_content[:300]}...")

    # 打印摘要
    print("\n📊 索引摘要:")
    summary = index_module.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


def example_2_small_to_big():
    """示例 2: Small-to-Big 策略"""
    print("\n" + "=" * 70)
    print("示例 2: Small-to-Big 索引策略")
    print("=" * 70)

    config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {
            "type": "small_to_big",
            "small_chunk_size": 400,
            "small_chunk_overlap": 50,
            "big_chunk_size": 2000,
            "big_chunk_overlap": 200,
        },
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/chroma_small_to_big",
            "collection_name": "small_to_big_index"
        },
    }

    index_module = IndexModule(config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/")

    # 测试检索（检索小块，但可以获取父块上下文）
    retriever = index_module.get_retriever(search_kwargs={"k": 2})
    results = retriever.invoke("科技公司裁员的原因是什么？")

    print(f"\n🔍 检索结果: 找到 {len(results)} 个相关文档")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"小块内容: {doc.page_content[:100]}...")
        if "parent_chunk_content" in doc.metadata:
            print(f"父块大小: {doc.metadata['parent_chunk_size']} 字符")
            print(f"父块内容预览: {doc.metadata['parent_chunk_content'][:150]}...")


def example_3_hierarchical():
    """示例 3: 层次化索引"""
    print("\n" + "=" * 70)
    print("示例 3: 层次化索引策略")
    print("=" * 70)

    config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {
            "type": "recursive",  # 层次化策略会忽略这个设置
            "chunk_size": 1000,
        },
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/chroma_hierarchical",
            "collection_name": "hierarchical_index"
        },
        "strategy": {
            "type": "hierarchical",
        },
    }

    index_module = IndexModule(config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/")

    # 测试检索
    retriever = index_module.get_retriever(search_kwargs={"k": 3})
    results = retriever.invoke("裁员潮的背景是什么？")

    print(f"\n🔍 检索结果: 找到 {len(results)} 个相关文档")
    for i, doc in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"层级: Level {doc.metadata.get('level', 'N/A')}")
        print(f"节点 ID: {doc.metadata.get('node_id', 'N/A')}")
        print(f"内容: {doc.page_content[:200]}...")


def example_4_semantic_splitter():
    """示例 4: 语义分块"""
    print("\n" + "=" * 70)
    print("示例 4: 语义分块策略")
    print("=" * 70)

    config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {
            "type": "semantic",
            "chunk_size": 800,
            "chunk_overlap": 100,
        },
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/chroma_semantic",
            "collection_name": "semantic_index"
        },
    }

    index_module = IndexModule(config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/")

    print("\n📊 索引摘要:")
    summary = index_module.summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")


def example_5_load_existing():
    """示例 5: 加载已存在的索引"""
    print("\n" + "=" * 70)
    print("示例 5: 加载已存在的索引")
    print("=" * 70)

    config = {
        "loader": {"type": "directory"},
        "splitter": {"type": "recursive"},
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/chroma_basic",
            "collection_name": "basic_index"
        },
    }

    index_module = IndexModule(config)

    try:
        vectorstore = index_module.load_existing_index()

        # 测试检索
        retriever = index_module.get_retriever(search_kwargs={"k": 2})
        results = retriever.invoke("科技股泡沫")

        print(f"\n🔍 检索结果: 找到 {len(results)} 个相关文档")
        for i, doc in enumerate(results, 1):
            print(f"\n结果 {i}: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("提示: 请先运行 example_1_basic_indexing() 创建索引")


if __name__ == "__main__":
    # 确保数据目录存在
    os.makedirs("./data", exist_ok=True)

    # 运行示例
    # 注意：根据需要取消注释相应的示例

    # 示例 1: 基础索引
    example_1_basic_indexing()

    # 示例 2: Small-to-Big 策略
    # example_2_small_to_big()

    # 示例 3: 层次化索引
    # example_3_hierarchical()

    # 示例 4: 语义分块
    # example_4_semantic_splitter()

    # 示例 5: 加载已存在的索引
    # example_5_load_existing()

    print("\n" + "=" * 70)
    print("✅ 所有示例运行完成！")
    print("=" * 70)
