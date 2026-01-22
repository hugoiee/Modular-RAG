"""
生成模块使用示例
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes.indexing import IndexModule
from nodes.retrieval import RetrievalModule
from nodes.generation import GenerationModule

load_dotenv()


def setup_test_data():
    """准备测试数据"""
    print("准备测试数据...")

    index_config = {
        "loader": {"type": "directory", "file_extensions": [".pdf"]},
        "splitter": {"type": "recursive", "chunk_size": 1000, "chunk_overlap": 200},
        "embedding": {"type": "dashscope"},
        "store": {"type": "chroma", "persist_directory": "./data/gen_demo_db"},
    }

    index_module = IndexModule(index_config)
    vectorstore = index_module.index_documents("./doc/金融新闻pdf/", verbose=False)
    print("✅ 数据准备完成\n")

    return vectorstore


def example_1_basic_generation(vectorstore):
    """示例 1: 基础生成"""
    print("=" * 70)
    print("示例 1: 基础生成")
    print("=" * 70)

    query = "美国科技公司裁员的主要原因是什么？"

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    # 生成
    generation = GenerationModule({
        "prompt_strategy": "template",
        "generator": "llm",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs)
    print(f"\n答案：\n{answer}")


def example_2_stream_generation(vectorstore):
    """示例 2: 流式生成"""
    print("\n" + "=" * 70)
    print("示例 2: 流式生成")
    print("=" * 70)

    query = "分析科技股是否存在泡沫"

    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    generation = GenerationModule({
        "prompt_strategy": "contextual",
        "generator": "stream",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs, verbose=False)


def example_3_cot_generation(vectorstore):
    """示例 3: 思维链生成"""
    print("\n" + "=" * 70)
    print("示例 3: Chain-of-Thought 生成")
    print("=" * 70)

    query = "为什么科技公司会出现大规模裁员？背后的深层原因是什么？"

    retrieval = RetrievalModule({"strategy": "dense", "k": 4})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    generation = GenerationModule({
        "prompt_strategy": "cot",
        "generator": "llm",
        "model": "qwen-plus",
        "steps": ["理解问题", "分析上下文信息", "识别关键因素", "逻辑推理", "得出结论"]
    })

    answer = generation.generate(query, docs, verbose=False)
    print(f"\n答案：\n{answer}")


def example_4_verification(vectorstore):
    """示例 4: 答案验证"""
    print("\n" + "=" * 70)
    print("示例 4: 答案验证（事实核查）")
    print("=" * 70)

    from nodes.generation_operators import FactCheckOperator, VerificationOperator

    query = "美国科技公司裁员规模有多大？"

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    # 生成答案
    generation = GenerationModule({
        "prompt_strategy": "template",
        "generator": "llm",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs, verbose=False)
    print(f"\n生成的答案：\n{answer}")

    # 基础验证
    print("\n" + "-" * 70)
    print("基础验证：")
    verifier = VerificationOperator({"threshold": 0.5})
    result = verifier.execute(query, docs, answer=answer)
    print(f"  有效性: {result['is_valid']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  原因: {result['reason']}")

    # 事实核查
    print("\n" + "-" * 70)
    print("事实核查：")
    fact_checker = FactCheckOperator({"model": "qwen-plus"})
    fact_result = fact_checker.execute(query, docs, answer=answer)
    print(f"  事实准确: {fact_result.get('is_factual', True)}")
    print(f"  置信度: {fact_result.get('confidence', 0.0):.2f}")
    if fact_result.get('violations'):
        print(f"  问题陈述: {fact_result['violations']}")


def example_5_citation(vectorstore):
    """示例 5: 添加引用标注"""
    print("\n" + "=" * 70)
    print("示例 5: 添加引用标注")
    print("=" * 70)

    from nodes.generation_operators import CitationOperator

    query = "分析科技股的投资风险"

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    # 生成答案
    generation = GenerationModule({
        "prompt_strategy": "contextual",
        "generator": "llm",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs, verbose=False)

    # 添加引用（脚注样式）
    citation_op = CitationOperator({"style": "footnote", "model": "qwen-plus"})
    cited_answer = citation_op.execute(query, docs, answer=answer)

    print(f"\n添加引用后的答案：\n{cited_answer}")


def example_6_formatting(vectorstore):
    """示例 6: 答案格式化"""
    print("\n" + "=" * 70)
    print("示例 6: 答案格式化")
    print("=" * 70)

    from nodes.generation_operators import OutputFormatterOperator

    query = "科技公司裁员的主要原因"

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    # 生成答案
    generation = GenerationModule({
        "prompt_strategy": "template",
        "generator": "llm",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs, verbose=False)

    # Markdown 格式
    print("\n" + "-" * 70)
    print("Markdown 格式：")
    formatter = OutputFormatterOperator({"format": "markdown", "add_metadata": True})
    markdown_output = formatter.execute(query, docs, answer=answer)
    print(markdown_output)

    # JSON 格式
    print("\n" + "-" * 70)
    print("JSON 格式：")
    json_formatter = OutputFormatterOperator({"format": "json", "add_metadata": True})
    json_output = json_formatter.execute(query, docs, answer=answer)
    print(json_output)


def example_7_refinement(vectorstore):
    """示例 7: 答案精炼"""
    print("\n" + "=" * 70)
    print("示例 7: 答案精炼")
    print("=" * 70)

    from nodes.generation_operators import AnswerRefinementOperator

    query = "评估当前科技行业的发展趋势"

    # 检索
    retrieval = RetrievalModule({"strategy": "dense", "k": 4})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)

    # 生成答案
    generation = GenerationModule({
        "prompt_strategy": "contextual",
        "generator": "llm",
        "model": "qwen-plus"
    })

    answer = generation.generate(query, docs, verbose=False)
    print(f"\n原始答案：\n{answer}")

    # 精炼答案
    print("\n" + "-" * 70)
    print("精炼后的答案：")
    refiner = AnswerRefinementOperator({
        "goals": ["clarity", "conciseness", "professional"],
        "model": "qwen-plus"
    })
    refined_answer = refiner.execute(query, docs, answer=answer)
    print(refined_answer)


def example_8_complete_pipeline(vectorstore):
    """示例 8: 完整的生成流水线"""
    print("\n" + "=" * 70)
    print("示例 8: 完整的生成流水线（生成 → 验证 → 引用 → 格式化）")
    print("=" * 70)

    from nodes.generation_operators import (
        VerificationOperator,
        CitationOperator,
        OutputFormatterOperator
    )

    query = "总结美国科技公司裁员的情况和影响"

    # 步骤 1: 检索
    print("\n📥 步骤 1: 检索相关文档")
    retrieval = RetrievalModule({"strategy": "dense", "k": 3})
    retrieval.build(vectorstore=vectorstore)
    docs = retrieval.retrieve(query, verbose=False)
    print(f"✅ 检索到 {len(docs)} 个相关文档")

    # 步骤 2: 生成
    print("\n🤖 步骤 2: 生成答案")
    generation = GenerationModule({
        "prompt_strategy": "cot",
        "generator": "llm",
        "model": "qwen-plus",
        "steps": ["理解问题", "分析信息", "归纳总结", "得出结论"]
    })
    answer = generation.generate(query, docs, verbose=False)
    print(f"✅ 答案已生成（{len(answer)} 字符）")

    # 步骤 3: 验证
    print("\n🔍 步骤 3: 验证答案")
    verifier = VerificationOperator({"threshold": 0.6})
    verification = verifier.execute(query, docs, answer=answer)
    print(f"  验证结果: {'✅ 通过' if verification['is_valid'] else '❌ 未通过'}")
    print(f"  置信度: {verification['confidence']:.2f}")

    # 步骤 4: 添加引用
    print("\n📎 步骤 4: 添加引用")
    citation_op = CitationOperator({"style": "numbered"})
    cited_answer = citation_op.execute(query, docs, answer=answer)
    print("✅ 引用已添加")

    # 步骤 5: 格式化
    print("\n📝 步骤 5: 格式化输出")
    formatter = OutputFormatterOperator({"format": "structured", "add_metadata": True})
    final_output = formatter.execute(query, docs, answer=cited_answer)

    print("\n" + "=" * 70)
    print("最终输出：")
    print("=" * 70)
    print(final_output)


if __name__ == "__main__":
    print("🚀 生成模块示例演示\n")

    vectorstore = setup_test_data()

    # 基础生成示例
    example_1_basic_generation(vectorstore)
    # example_2_stream_generation(vectorstore)
    # example_3_cot_generation(vectorstore)

    # 验证和后处理示例
    # example_4_verification(vectorstore)
    # example_5_citation(vectorstore)
    # example_6_formatting(vectorstore)
    # example_7_refinement(vectorstore)

    # 完整流水线示例
    # example_8_complete_pipeline(vectorstore)

    print("\n✅ 示例演示完成！")
