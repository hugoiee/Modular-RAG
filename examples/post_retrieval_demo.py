"""
检索后模块使用示例

演示如何使用 PostRetrievalModule 的各种优化策略
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.indexing import IndexModule
from nodes.retrieval import RetrievalModule
from nodes.post_retrieval import PostRetrievalModule, PostRetrievalPipeline

# 加载环境变量
load_dotenv()


def setup_test_data():
    """准备测试数据"""
    print("\n" + "=" * 70)
    print("准备测试数据")
    print("=" * 70)

    # 索引文档
    index_config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
        "embedding": {"type": "dashscope", "model": "text-embedding-v4"},
        "store": {
            "type": "chroma",
            "persist_directory": "./data/post_retrieval_demo_db",
            "collection_name": "post_retrieval_test"
        },
    }

    index_module = IndexModule(index_config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/", verbose=False)

    print(f"✅ 数据准备完成")

    return vectorstore


def example_1_rerank(vectorstore):
    """示例 1: 基础重排序"""
    print("\n" + "=" * 70)
    print("示例 1: Rerank（基础重排序）")
    print("=" * 70)

    # 先检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 10})
    retrieval.build(vectorstore=vectorstore)
    query = "美国科技公司的裁员情况"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n检索到 {len(docs)} 个文档")
    print("\n重排序前（前3个）:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"{i}. {doc.page_content[:80]}...")

    # 重排序
    post_retrieval = PostRetrievalModule({"strategy": "rerank", "top_n": 5})
    reranked = post_retrieval.process(docs, query, verbose=False)

    print(f"\n重排序后（前3个）:")
    for i, doc in enumerate(reranked[:3], 1):
        print(f"{i}. {doc.page_content[:80]}...")


def example_2_diversity_rerank(vectorstore):
    """示例 2: 多样性重排序"""
    print("\n" + "=" * 70)
    print("示例 2: Diversity Rerank（多样性重排序）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 10})
    retrieval.build(vectorstore=vectorstore)
    query = "科技行业投资风险"
    docs = retrieval.retrieve(query, verbose=False)

    # 多样性重排序
    config = {
        "strategy": "diversity_rerank",
        "diversity_weight": 0.6,
        "top_n": 5
    }
    post_retrieval = PostRetrievalModule(config)
    reranked = post_retrieval.process(docs, query, verbose=False)

    print(f"\n多样性重排序结果（{len(reranked)} 个文档）:")
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc.page_content[:80]}...")


def example_3_llm_rerank(vectorstore):
    """示例 3: LLM 重排序"""
    print("\n" + "=" * 70)
    print("示例 3: LLM Rerank（使用 LLM 评分重排序）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 5})
    retrieval.build(vectorstore=vectorstore)
    query = "人工智能投资的主要风险是什么？"
    docs = retrieval.retrieve(query, verbose=False)

    # LLM 重排序
    config = {
        "strategy": "llm_rerank",
        "model": "qwen-plus",
        "top_n": 3
    }
    post_retrieval = PostRetrievalModule(config)
    reranked = post_retrieval.process(docs, query, verbose=False)

    print(f"\nLLM 重排序结果（{len(reranked)} 个文档）:")
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc.page_content[:100]}...")


def example_4_context_compression(vectorstore):
    """示例 4: 上下文压缩"""
    print("\n" + "=" * 70)
    print("示例 4: Context Compression（上下文压缩）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 5})
    retrieval.build(vectorstore=vectorstore)
    query = "裁员原因"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n压缩前:")
    for i, doc in enumerate(docs[:2], 1):
        print(f"文档 {i} 长度: {len(doc.page_content)} 字符")
        print(f"内容: {doc.page_content[:150]}...\n")

    # 压缩
    config = {
        "strategy": "context_compression",
        "compression_ratio": 0.5,
        "max_tokens": 200
    }
    post_retrieval = PostRetrievalModule(config)
    compressed = post_retrieval.process(docs, query, verbose=False)

    print(f"\n压缩后:")
    for i, doc in enumerate(compressed[:2], 1):
        print(f"文档 {i} 长度: {len(doc.page_content)} 字符")
        print(f"内容: {doc.page_content[:150]}...\n")


def example_5_summary_compression(vectorstore):
    """示例 5: 摘要压缩"""
    print("\n" + "=" * 70)
    print("示例 5: Summary Compression（摘要压缩）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    query = "科技股泡沫"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n原始文档（第1个）:")
    print(f"长度: {len(docs[0].page_content)} 字符")
    print(f"内容: {docs[0].page_content[:200]}...")

    # 摘要压缩
    config = {
        "strategy": "summary_compression",
        "model": "qwen-plus",
        "max_summary_length": 150
    }
    post_retrieval = PostRetrievalModule(config)
    summarized = post_retrieval.process(docs, query, verbose=False)

    print(f"\n摘要后（第1个）:")
    print(f"长度: {len(summarized[0].page_content)} 字符")
    print(f"摘要: {summarized[0].page_content}")


def example_6_relevance_filter(vectorstore):
    """示例 6: 相关性过滤"""
    print("\n" + "=" * 70)
    print("示例 6: Relevance Filter（相关性过滤）")
    print("=" * 70)

    # 检索（获取更多文档，其中可能有不相关的）
    retrieval = RetrievalModule({"strategy": "dense", "k": 8})
    retrieval.build(vectorstore=vectorstore)
    query = "英特尔公司的裁员数量"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n过滤前: {len(docs)} 个文档")

    # 相关性过滤
    config = {
        "strategy": "relevance_filter",
        "model": "qwen-plus",
        "relevance_threshold": 0.5,
        "min_docs": 2
    }
    post_retrieval = PostRetrievalModule(config)
    filtered = post_retrieval.process(docs, query, verbose=False)

    print(f"\n过滤后: {len(filtered)} 个相关文档")


def example_7_redundancy_filter(vectorstore):
    """示例 7: 冗余过滤"""
    print("\n" + "=" * 70)
    print("示例 7: Redundancy Filter（冗余过滤）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 10})
    retrieval.build(vectorstore=vectorstore)
    query = "科技公司裁员"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n去重前: {len(docs)} 个文档")

    # 冗余过滤
    config = {
        "strategy": "redundancy_filter",
        "similarity_threshold": 0.8
    }
    post_retrieval = PostRetrievalModule(config)
    filtered = post_retrieval.process(docs, query, verbose=False)

    print(f"\n去重后: {len(filtered)} 个唯一文档")


def example_8_pipeline(vectorstore):
    """示例 8: 检索后流水线"""
    print("\n" + "=" * 70)
    print("示例 8: Post-Retrieval Pipeline（多步骤优化）")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 15})
    retrieval.build(vectorstore=vectorstore)
    query = "美国科技行业的主要问题"
    docs = retrieval.retrieve(query, verbose=False)

    print(f"\n原始检索结果: {len(docs)} 个文档")

    # 创建流水线
    pipeline = PostRetrievalPipeline()
    pipeline.add_step("rerank", {"top_n": 10})               # 步骤1: 重排序
    pipeline.add_step("redundancy_filter", {"similarity_threshold": 0.85})  # 步骤2: 去重
    pipeline.add_step("context_compression", {"compression_ratio": 0.6})    # 步骤3: 压缩

    # 执行流水线
    optimized = pipeline.process(docs, query, verbose=True)

    print(f"\n最终结果（前2个）:")
    for i, doc in enumerate(optimized[:2], 1):
        print(f"\n文档 {i}:")
        print(f"长度: {len(doc.page_content)} 字符")
        print(f"内容: {doc.page_content[:150]}...")


def example_9_dynamic_strategy(vectorstore):
    """示例 9: 动态切换策略"""
    print("\n" + "=" * 70)
    print("示例 9: 动态切换优化策略")
    print("=" * 70)

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 8})
    retrieval.build(vectorstore=vectorstore)
    query = "投资风险"
    docs = retrieval.retrieve(query, verbose=False)

    # 创建模块，初始使用 Rerank
    post_retrieval = PostRetrievalModule({"strategy": "rerank", "top_n": 5})

    print(f"\n原始文档: {len(docs)} 个")

    # 策略 1: Rerank
    print("\n--- 策略 1: Rerank ---")
    result1 = post_retrieval.process(docs, query, verbose=False)
    print(f"结果: {len(result1)} 个文档")

    # 切换到策略 2: Redundancy Filter
    print("\n--- 切换策略 ---")
    post_retrieval.change_strategy("redundancy_filter", {"similarity_threshold": 0.8})

    print("\n--- 策略 2: Redundancy Filter ---")
    result2 = post_retrieval.process(docs, query, verbose=False)
    print(f"结果: {len(result2)} 个文档")

    # 切换到策略 3: Context Compression
    print("\n--- 切换策略 ---")
    post_retrieval.change_strategy("context_compression", {"compression_ratio": 0.5})

    print("\n--- 策略 3: Context Compression ---")
    result3 = post_retrieval.process(docs, query, verbose=False)
    print(f"结果: {len(result3)} 个文档（已压缩）")


def example_10_complete_workflow(vectorstore):
    """示例 10: 完整工作流（检索 + 后处理）"""
    print("\n" + "=" * 70)
    print("示例 10: 完整工作流（Retrieval + Post-Retrieval）")
    print("=" * 70)

    query = "科技公司大规模裁员的深层原因"

    # 步骤1: 混合检索
    print("\n📍 步骤 1: 混合检索")
    retrieval = RetrievalModule({
        "strategy": "hybrid",
        "dense_weight": 0.6,
        "sparse_weight": 0.4,
        "k": 10
    })
    # 需要 documents 用于 BM25
    from nodes.indexing import IndexModule
    index_module = IndexModule()
    # 假设已有 vectorstore 和 documents
    # 这里简化，只用 dense
    retrieval = RetrievalModule({"strategy": "dense", "k": 10})
    retrieval.build(vectorstore=vectorstore)

    docs = retrieval.retrieve(query, verbose=False)
    print(f"检索到 {len(docs)} 个文档")

    # 步骤2: 后处理流水线
    print("\n📍 步骤 2: 后处理优化")
    pipeline = PostRetrievalPipeline()
    pipeline.add_step("llm_rerank", {"top_n": 5})  # LLM 重排序
    pipeline.add_step("redundancy_filter")          # 去重
    pipeline.add_step("context_compression", {"compression_ratio": 0.7})  # 压缩

    optimized = pipeline.process(docs, query, verbose=False)

    print(f"\n最终优化结果: {len(optimized)} 个文档")
    print("\n最终文档内容（前2个）:")
    for i, doc in enumerate(optimized[:2], 1):
        print(f"\n文档 {i}:")
        print(f"长度: {len(doc.page_content)} 字符")
        print(f"内容: {doc.page_content[:200]}...")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 检索后模块示例演示")
    print("=" * 70)

    # 准备数据
    vectorstore = setup_test_data()

    # 运行示例（取消注释想要运行的示例）

    # 重排序示例
    example_1_rerank(vectorstore)
    # example_2_diversity_rerank(vectorstore)
    # example_3_llm_rerank(vectorstore)

    # 压缩示例
    # example_4_context_compression(vectorstore)
    # example_5_summary_compression(vectorstore)

    # 过滤示例
    # example_6_relevance_filter(vectorstore)
    # example_7_redundancy_filter(vectorstore)

    # 流水线和完整工作流
    # example_8_pipeline(vectorstore)
    # example_9_dynamic_strategy(vectorstore)
    # example_10_complete_workflow(vectorstore)

    print("\n" + "=" * 70)
    print("✅ 示例演示完成！")
    print("=" * 70)
    print("\n💡 提示：可以取消注释其他示例来测试更多功能")
