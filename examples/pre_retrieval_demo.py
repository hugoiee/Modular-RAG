"""
检索前模块使用示例

演示如何使用 PreRetrievalModule 的各种优化策略
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.pre_retrieval import PreRetrievalModule, PreRetrievalPipeline

# 加载环境变量
load_dotenv()


def example_1_query_rewrite():
    """示例 1: 查询重写"""
    print("\n" + "=" * 70)
    print("示例 1: Query Rewrite（查询重写）")
    print("=" * 70)

    config = {
        "strategy": "query_rewrite",
        "model": "qwen-plus",
        "temperature": 0.3
    }

    pre_retrieval = PreRetrievalModule(config)

    # 测试多个查询
    queries = [
        "AI是啥？",
        "Python好还是Java好？",
        "美国科技股咋样了？",
    ]

    for query in queries:
        print(f"\n原始查询: {query}")
        rewritten = pre_retrieval.process(query, verbose=False)
        print(f"重写查询: {rewritten}")


def example_2_multi_query():
    """示例 2: 多查询生成"""
    print("\n" + "=" * 70)
    print("示例 2: Multi-Query（多查询生成）")
    print("=" * 70)

    config = {
        "strategy": "multi_query",
        "num_queries": 3,
        "model": "qwen-plus",
        "temperature": 0.7
    }

    pre_retrieval = PreRetrievalModule(config)

    query = "美国科技行业的投资风险如何？"
    print(f"\n原始查询: {query}\n")

    queries = pre_retrieval.process(query)

    print(f"\n生成的查询变体：")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")


def example_3_sub_query():
    """示例 3: 查询分解"""
    print("\n" + "=" * 70)
    print("示例 3: Sub-Query（查询分解）")
    print("=" * 70)

    config = {
        "strategy": "sub_query",
        "max_sub_queries": 4,
        "model": "qwen-plus",
    }

    pre_retrieval = PreRetrievalModule(config)

    query = "比较Python和Java在机器学习领域的应用，并分析各自的优缺点。"
    print(f"\n原始复杂查询: {query}\n")

    sub_queries = pre_retrieval.process(query)

    print(f"\n分解后的子查询：")
    for i, sq in enumerate(sub_queries, 1):
        print(f"  {i}. {sq}")


def example_4_hyde():
    """示例 4: HyDE（假设性文档生成）"""
    print("\n" + "=" * 70)
    print("示例 4: HyDE（假设性文档生成）")
    print("=" * 70)

    config = {
        "strategy": "hyde",
        "doc_length": "medium",
        "model": "qwen-plus",
        "temperature": 0.7
    }

    pre_retrieval = PreRetrievalModule(config)

    query = "什么是量子计算？"
    print(f"\n原始查询: {query}\n")

    hypothetical_doc = pre_retrieval.process(query, verbose=False)

    print(f"\n生成的假设性文档：")
    print(f"{hypothetical_doc}")
    print(f"\n💡 提示：使用这个假设文档去检索，而不是原始问题")


def example_5_step_back():
    """示例 5: Step-back Prompting"""
    print("\n" + "=" * 70)
    print("示例 5: Step-back Prompting（后退提示）")
    print("=" * 70)

    config = {
        "strategy": "step_back",
        "return_both": True,
        "model": "qwen-plus",
    }

    pre_retrieval = PreRetrievalModule(config)

    query = "Transformer模型中的self-attention机制是如何工作的？"
    print(f"\n原始具体查询: {query}\n")

    step_back_result = pre_retrieval.process(query, verbose=False)

    print(f"\nStep-back 结果：")
    print(f"{step_back_result}")


def example_6_hybrid_expansion():
    """示例 6: 混合扩展策略"""
    print("\n" + "=" * 70)
    print("示例 6: Hybrid Expansion（智能选择扩展策略）")
    print("=" * 70)

    config = {
        "strategy": "hybrid_expansion",
        "num_queries": 3,
        "max_sub_queries": 4,
        "complexity_threshold": 0.6,
        "model": "qwen-plus",
    }

    pre_retrieval = PreRetrievalModule(config)

    # 测试简单查询（应该使用 Multi-Query）
    simple_query = "什么是机器学习？"
    print(f"\n测试 1 - 简单查询: {simple_query}")
    result1 = pre_retrieval.process(simple_query)
    print(f"生成查询数: {len(result1)}")

    print("\n" + "-" * 70)

    # 测试复杂查询（应该使用 Sub-Query）
    complex_query = "比较并分析深度学习和传统机器学习在图像识别领域的应用效果和局限性。"
    print(f"\n测试 2 - 复杂查询: {complex_query}")
    result2 = pre_retrieval.process(complex_query)
    print(f"生成查询数: {len(result2)}")


def example_7_text_to_sql():
    """示例 7: Text-to-SQL"""
    print("\n" + "=" * 70)
    print("示例 7: Text-to-SQL（自然语言转SQL）")
    print("=" * 70)

    # 定义数据库schema
    schema = {
        "orders": ["order_id", "customer_id", "order_date", "total_amount"],
        "customers": ["customer_id", "name", "email", "city"],
        "products": ["product_id", "name", "price", "category"]
    }

    config = {
        "strategy": "text_to_sql",
        "schema": schema,
        "model": "qwen-plus",
    }

    pre_retrieval = PreRetrievalModule(config)

    # 测试多个查询
    queries = [
        "查询所有订单总额超过1000的订单",
        "统计每个城市的客户数量",
        "找出销售额最高的5个产品",
    ]

    for query in queries:
        print(f"\n自然语言: {query}")
        sql = pre_retrieval.process(query, verbose=False)
        print(f"SQL查询: {sql}")


def example_8_pipeline():
    """示例 8: 检索前流水线"""
    print("\n" + "=" * 70)
    print("示例 8: Pre-Retrieval Pipeline（流水线处理）")
    print("=" * 70)

    # 创建流水线
    pipeline = PreRetrievalPipeline()

    # 添加多个处理步骤
    pipeline.add_step("query_rewrite")  # 步骤1：重写查询
    pipeline.add_step("multi_query", {"num_queries": 2})  # 步骤2：生成变体

    # 处理查询
    query = "Python在数据科学中的应用"
    print(f"\n原始查询: {query}\n")

    result = pipeline.process(query)

    print(f"\n最终结果：")
    if isinstance(result, list):
        for i, q in enumerate(result, 1):
            print(f"  {i}. {q}")
    else:
        print(f"  {result}")


def example_9_dynamic_strategy():
    """示例 9: 动态切换策略"""
    print("\n" + "=" * 70)
    print("示例 9: 动态切换优化策略")
    print("=" * 70)

    # 创建模块，初始使用 Query Rewrite
    pre_retrieval = PreRetrievalModule({"strategy": "query_rewrite"})

    query = "深度学习和机器学习有什么区别？"

    # 策略 1: Query Rewrite
    print(f"\n原始查询: {query}\n")
    print("--- 策略 1: Query Rewrite ---")
    result1 = pre_retrieval.process(query, verbose=False)
    print(f"结果: {result1}")

    # 切换到策略 2: Multi-Query
    print("\n--- 切换策略 ---")
    pre_retrieval.change_strategy("multi_query", {"num_queries": 2})

    print("\n--- 策略 2: Multi-Query ---")
    result2 = pre_retrieval.process(query, verbose=False)
    print(f"结果: {result2}")

    # 切换到策略 3: HyDE
    print("\n--- 切换策略 ---")
    pre_retrieval.change_strategy("hyde", {"doc_length": "short"})

    print("\n--- 策略 3: HyDE ---")
    result3 = pre_retrieval.process(query, verbose=False)
    print(f"结果: {result3}")


def example_10_cot_rewrite():
    """示例 10: Chain-of-Thought 改写"""
    print("\n" + "=" * 70)
    print("示例 10: Chain-of-Thought Rewrite（思维链改写）")
    print("=" * 70)

    config = {
        "strategy": "cot_rewrite",
        "model": "qwen-plus",
    }

    pre_retrieval = PreRetrievalModule(config)

    query = "美国科技行业是否存在投资泡沫？"
    print(f"\n原始查询: {query}\n")

    cot_query = pre_retrieval.process(query, verbose=False)

    print(f"CoT改写查询：")
    print(f"{cot_query}")


if __name__ == "__main__":
    # 运行示例
    # 注意：这些示例需要 API key，确保 .env 文件中配置了 DASHSCOPE_API_KEY

    print("\n" + "=" * 70)
    print("🚀 检索前模块示例演示")
    print("=" * 70)

    # 选择要运行的示例（取消注释）

    # 基础示例
    # example_1_query_rewrite()
    # example_2_multi_query()
    # example_3_sub_query()

    # 高级示例
    # example_4_hyde()
    # example_5_step_back()
    # example_6_hybrid_expansion()

    # 结构化查询
    # example_7_text_to_sql()

    # 流水线和动态策略
    # example_8_pipeline()
    # example_9_dynamic_strategy()
    # example_10_cot_rewrite()

    # 运行所有示例（需要较长时间）
    example_1_query_rewrite()
    example_2_multi_query()
    example_3_sub_query()

    print("\n" + "=" * 70)
    print("✅ 示例演示完成！")
    print("=" * 70)
    print("\n💡 提示：可以取消注释其他示例来测试更多功能")
