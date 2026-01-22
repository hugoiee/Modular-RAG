"""
检索模块使用示例

演示如何使用 RetrievalModule 的各种检索策略
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.indexing import IndexModule
from nodes.retrieval import RetrievalModule, RetrievalPipeline

# 加载环境变量
load_dotenv()


def setup_test_data():
    """准备测试数据：索引文档"""
    print("\n" + "=" * 70)
    print("准备测试数据")
    print("=" * 70)

    # 使用索引模块创建向量数据库
    index_config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/retrieval_demo_db",
            "collection_name": "retrieval_test"
        },
    }

    index_module = IndexModule(index_config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/", verbose=False)

    # 获取文档列表（用于 Sparse 检索）
    documents = index_module.splits

    print(f"✅ 数据准备完成")
    print(f"   - 向量数据库: {len(documents)} 个文档块")

    return vectorstore, documents


def example_1_dense_retrieval(vectorstore, documents):
    """示例 1: Dense Retrieval（语义检索）"""
    print("\n" + "=" * 70)
    print("示例 1: Dense Retrieval（语义向量检索）")
    print("=" * 70)

    config = {
        "strategy": "dense",
        "search_type": "similarity",
        "k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore)

    query = "美国科技公司的投资风险"
    results = retrieval.retrieve(query, verbose=False)

    print(f"\n查询: {query}")
    print(f"检索到 {len(results)} 个文档:\n")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_2_sparse_retrieval(vectorstore, documents):
    """示例 2: Sparse Retrieval（BM25 关键词检索）"""
    print("\n" + "=" * 70)
    print("示例 2: Sparse Retrieval (BM25)")
    print("=" * 70)

    config = {
        "strategy": "bm25",
        "k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(documents=documents)

    query = "裁员 科技行业"
    results = retrieval.retrieve(query, verbose=False)

    print(f"\n查询: {query}")
    print(f"检索到 {len(results)} 个文档:\n")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_3_hybrid_retrieval(vectorstore, documents):
    """示例 3: Hybrid Retrieval（混合检索）"""
    print("\n" + "=" * 70)
    print("示例 3: Hybrid Retrieval（Dense + Sparse 融合）")
    print("=" * 70)

    config = {
        "strategy": "hybrid",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore, documents=documents)

    query = "科技股泡沫的主要原因"
    results = retrieval.retrieve(query, verbose=False)

    print(f"\n查询: {query}")
    print(f"检索到 {len(results)} 个文档:\n")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_4_semantic_mmr(vectorstore, documents):
    """示例 4: Semantic Retrieval with MMR（多样性检索）"""
    print("\n" + "=" * 70)
    print("示例 4: Semantic Retrieval with MMR（保证多样性）")
    print("=" * 70)

    config = {
        "strategy": "semantic",
        "search_type": "mmr",
        "k": 5,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore)

    query = "人工智能对经济的影响"
    results = retrieval.retrieve(query, verbose=False)

    print(f"\n查询: {query}")
    print(f"检索到 {len(results)} 个文档（使用 MMR 确保多样性）:\n")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_5_adaptive_hybrid(vectorstore, documents):
    """示例 5: Adaptive Hybrid Retrieval（自适应混合检索）"""
    print("\n" + "=" * 70)
    print("示例 5: Adaptive Hybrid Retrieval（智能调整权重）")
    print("=" * 70)

    config = {
        "strategy": "adaptive_hybrid",
        "k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore, documents=documents)

    # 测试不同类型的查询
    queries = [
        "什么是科技股泡沫？",  # 语义查询
        "裁员 英特尔",  # 关键词查询
    ]

    for query in queries:
        print(f"\n--- 查询: {query} ---")
        results = retrieval.retrieve(query, verbose=False)
        print(f"检索到 {len(results)} 个文档")


def example_6_adaptive_k(vectorstore, documents):
    """示例 6: Adaptive-K Retrieval（动态 K 值）"""
    print("\n" + "=" * 70)
    print("示例 6: Adaptive-K Retrieval（根据复杂度调整返回数量）")
    print("=" * 70)

    config = {
        "strategy": "adaptive_k",
        "min_k": 3,
        "max_k": 10,
        "default_k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore)

    # 简单查询 vs 复杂查询
    queries = [
        "AI",  # 简单
        "请详细分析并比较美国科技行业在2024年的投资风险和市场泡沫现象，包括各个细分领域的具体情况",  # 复杂
    ]

    for query in queries:
        print(f"\n--- 查询: {query[:50]}... ---")
        results = retrieval.retrieve(query, verbose=False)


def example_7_query_router(vectorstore, documents):
    """示例 7: Query Router（查询路由）"""
    print("\n" + "=" * 70)
    print("示例 7: Query Router（智能路由到合适的检索器）")
    print("=" * 70)

    # 创建多个检索器
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    from langchain_community.retrievers import BM25Retriever
    sparse_retriever = BM25Retriever.from_documents(documents, k=5)
    from langchain_core.retrievers import EnsembleRetriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )

    config = {
        "strategy": "query_router",
        "k": 5
    }

    retrieval = RetrievalModule(config)
    retrieval.build(retrievers={
        "dense": dense_retriever,
        "sparse": sparse_retriever,
        "hybrid": hybrid_retriever
    })

    # 测试不同类型的查询
    queries = [
        "什么是量子计算？",  # 语义查询 -> dense
        "查找 裁员 新闻",  # 关键词查询 -> sparse
    ]

    for query in queries:
        print(f"\n--- 查询: {query} ---")
        results = retrieval.retrieve(query, verbose=False)


def example_8_retrieval_pipeline(vectorstore, documents):
    """示例 8: Retrieval Pipeline（多阶段检索）"""
    print("\n" + "=" * 70)
    print("示例 8: Retrieval Pipeline（召回 + 精排）")
    print("=" * 70)

    # 创建流水线
    pipeline = RetrievalPipeline()

    # 阶段 1: BM25 快速召回（返回更多候选）
    pipeline.add_stage("bm25", config={"k": 10}, documents=documents)

    # 阶段 2: 语义精排（从候选中选择最相关的）
    # 注意：实际应用中精排阶段需要特殊处理，这里简化演示
    # pipeline.add_stage("semantic", config={"k": 5}, vectorstore=vectorstore)

    query = "科技公司裁员的原因"
    print(f"\n查询: {query}")

    results = pipeline.retrieve(query, verbose=False)
    print(f"\n最终检索到 {len(results)} 个文档")


def example_9_multi_query_retrieval(vectorstore, documents):
    """示例 9: Multi-Query Retrieval（多查询检索）"""
    print("\n" + "=" * 70)
    print("示例 9: Multi-Query Retrieval（使用多个查询变体）")
    print("=" * 70)

    config = {
        "strategy": "dense",
        "k": 3
    }

    retrieval = RetrievalModule(config)
    retrieval.build(vectorstore=vectorstore)

    # 使用多个查询（模拟 pre-retrieval 生成的查询变体）
    queries = [
        "美国科技行业投资风险",
        "科技股是否存在泡沫",
        "科技公司的市场表现"
    ]

    print(f"使用 {len(queries)} 个查询变体:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    results = retrieval.retrieve(queries, verbose=False)

    print(f"\n融合检索结果，共 {len(results)} 个唯一文档:")
    for i, doc in enumerate(results[:3], 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_10_dynamic_strategy(vectorstore, documents):
    """示例 10: 动态切换检索策略"""
    print("\n" + "=" * 70)
    print("示例 10: 动态切换检索策略")
    print("=" * 70)

    # 创建检索模块，初始使用 Dense
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)

    query = "科技行业现状"

    # 策略 1: Dense
    print(f"\n查询: {query}")
    print("\n--- 策略 1: Dense Retrieval ---")
    results1 = retrieval.retrieve(query, verbose=False)
    print(f"找到 {len(results1)} 个文档")

    # 切换到策略 2: BM25
    print("\n--- 切换策略 ---")
    retrieval.change_strategy("bm25", {"k": 3})
    retrieval.build(documents=documents)

    print("\n--- 策略 2: BM25 Retrieval ---")
    results2 = retrieval.retrieve(query, verbose=False)
    print(f"找到 {len(results2)} 个文档")

    # 切换到策略 3: Hybrid
    print("\n--- 切换策略 ---")
    retrieval.change_strategy("hybrid", {"dense_weight": 0.5, "sparse_weight": 0.5, "k": 3})
    retrieval.build(vectorstore=vectorstore, documents=documents)

    print("\n--- 策略 3: Hybrid Retrieval ---")
    results3 = retrieval.retrieve(query, verbose=False)
    print(f"找到 {len(results3)} 个文档")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 检索模块示例演示")
    print("=" * 70)

    # 准备数据
    vectorstore, documents = setup_test_data()

    # 运行示例（取消注释想要运行的示例）

    # 基础检索策略
    example_1_dense_retrieval(vectorstore, documents)
    example_2_sparse_retrieval(vectorstore, documents)
    example_3_hybrid_retrieval(vectorstore, documents)

    # 高级检索策略
    # example_4_semantic_mmr(vectorstore, documents)
    # example_5_adaptive_hybrid(vectorstore, documents)
    # example_6_adaptive_k(vectorstore, documents)

    # 智能路由和流水线
    # example_7_query_router(vectorstore, documents)
    # example_8_retrieval_pipeline(vectorstore, documents)

    # 多查询和动态策略
    # example_9_multi_query_retrieval(vectorstore, documents)
    # example_10_dynamic_strategy(vectorstore, documents)

    print("\n" + "=" * 70)
    print("✅ 示例演示完成！")
    print("=" * 70)
    print("\n💡 提示：可以取消注释其他示例来测试更多功能")
