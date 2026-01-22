"""
Query Transformation Operators（查询转换）

论文核心技术：
1. Query Rewrite: 重写查询以提高检索准确性
2. HyDE (Hypothetical Document Embeddings): 生成假设性文档
3. Step-back Prompting: 抽象为高层概念问题
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwen import ChatQwen
from .base import BasePreRetrievalOperator


class QueryRewriteOperator(BasePreRetrievalOperator):
    """
    Query Rewrite 操作器（查询重写）

    功能：
    - 优化措辞不当的查询
    - 消除歧义
    - 添加必要的上下文
    - 使查询更加精确和专业

    应用场景：
    - 用户查询表述不清
    - 包含口语化表达
    - 缺少关键信息
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.3)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询
        """
        print(f"✏️  Query Rewrite: 正在优化查询...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询优化专家。你的任务是将用户的原始查询重写为更加清晰、精确、专业的版本。

优化原则：
1. 消除歧义和模糊表达
2. 使用更专业和准确的术语
3. 保持原始意图不变
4. 添加必要的上下文（如果原查询过于简短）
5. 去除冗余和口语化表达
6. 确保查询是完整的问句

示例：
原始：AI是啥？
重写：什么是人工智能（AI）？请解释其定义和核心概念。

原始：Python好还是Java好？
重写：在软件开发领域，Python和Java各有什么优缺点？

原始：股票会涨吗？
重写：根据当前市场情况，股票市场的未来走势预测如何？

直接输出重写后的查询，不需要任何解释或前缀。"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        rewritten_query = chain.invoke({"query": query}).strip()

        print(f"   原始查询: {query}")
        print(f"   重写查询: {rewritten_query}")

        return rewritten_query


class HyDEOperator(BasePreRetrievalOperator):
    """
    HyDE (Hypothetical Document Embeddings) 操作器

    核心思想（论文创新技术）：
    - 不直接用查询去检索
    - 而是生成一个"假设性的答案文档"
    - 用这个假设文档去检索相似的真实文档

    优势：
    - 假设文档的语义空间更接近目标文档
    - 提高检索准确率
    - 适用于文档和查询的语义差异较大的场景

    原理：
    问题："什么是机器学习？"
    假设文档："机器学习是人工智能的一个分支，它使用算法和统计模型..."
    用假设文档的 embedding 去检索实际文档
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.7)
        self.doc_length = self.config.get("doc_length", "medium")  # short, medium, long

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        生成假设性文档

        Args:
            query: 原始查询

        Returns:
            假设性文档（用于检索）
        """
        print(f"📄 HyDE: 正在生成假设性文档...")

        # 根据长度配置调整 prompt
        length_guide = {
            "short": "1-2句话",
            "medium": "1-2段话（100-200字）",
            "long": "3-4段话（200-400字）"
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个知识助手。给定一个问题，请生成一个假设性的答案文档。

重要：
1. 假装你已经知道这个问题的答案
2. 生成一个详细、专业的答案文档
3. 使用与真实文档相似的语言风格和术语
4. 长度要求：{length_guide}
5. 直接输出答案内容，不需要前缀如"答案："或"假设文档："

这个假设文档将用于检索相似的真实文档，所以要尽可能专业和详细。

示例：
问题：什么是量子计算？
假设文档：量子计算是一种基于量子力学原理的新型计算模式。与传统计算机使用比特（bit）作为信息的基本单位不同，量子计算机使用量子比特（qubit）。量子比特可以同时处于0和1的叠加态，这种特性使得量子计算机在处理某些特定问题时具有指数级的速度优势。量子计算在密码学、药物研发、优化问题等领域有广泛的应用前景。"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        hypothetical_doc = chain.invoke({
            "query": query,
            "length_guide": length_guide.get(self.doc_length, length_guide["medium"])
        }).strip()

        print(f"   原始查询: {query}")
        print(f"   假设文档: {hypothetical_doc[:150]}...")

        return hypothetical_doc


class StepBackOperator(BasePreRetrievalOperator):
    """
    Step-back Prompting 操作器（后退提示）

    核心思想（论文创新技术）：
    - 将具体的查询抽象为更高层次的概念性问题
    - 先回答高层问题，建立概念框架
    - 再用框架指导具体问题的回答

    优势：
    - 避免陷入具体细节
    - 建立更好的概念基础
    - 提高复杂问题的回答质量

    示例：
    原始查询："Transformer模型中的self-attention机制是如何工作的？"
    Step-back查询："Transformer模型的基本架构和核心组件是什么？"
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.3)
        self.return_both = self.config.get("return_both", True)  # 是否返回原始+抽象

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        生成 step-back 查询

        Args:
            query: 原始具体查询

        Returns:
            抽象的高层次查询（或原始+抽象）
        """
        print(f"🔙 Step-back: 正在抽象查询...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个查询抽象专家。给定一个具体的查询，将其抽象为更高层次的概念性问题。

抽象原则：
1. 识别查询中的具体细节
2. 提升到更通用的概念层面
3. 问题应该更宏观、更基础
4. 有助于建立概念框架
5. 直接输出抽象后的查询，不需要解释

示例1：
原始查询：GPT-4的token限制是多少？
抽象查询：大语言模型的上下文长度限制及其影响是什么？

示例2：
原始查询：Python中如何使用装饰器？
抽象查询：Python中的装饰器是什么？有什么作用和应用场景？

示例3：
原始查询：2024年美国科技股是否存在泡沫？
抽象查询：如何判断股票市场是否存在泡沫？有哪些标志和指标？"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        step_back_query = chain.invoke({"query": query}).strip()

        print(f"   原始查询: {query}")
        print(f"   抽象查询: {step_back_query}")

        if self.return_both:
            # 返回两个查询的组合
            combined = f"首先回答这个基础问题：{step_back_query}\n\n然后具体回答：{query}"
            return combined
        else:
            return step_back_query


class ChainOfThoughtRewriteOperator(BasePreRetrievalOperator):
    """
    Chain-of-Thought Query Rewrite 操作器

    结合思维链技术改进查询：
    - 将查询转换为推理步骤
    - 适合需要多步推理的问题
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.3)

        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
        )

    def execute(self, query: str) -> str:
        """
        将查询转换为带推理步骤的版本

        Args:
            query: 原始查询

        Returns:
            带推理步骤的查询
        """
        print(f"🔗 CoT Rewrite: 正在添加推理步骤...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个推理专家。将用户的查询改写为带有明确推理步骤的版本。

改写要求：
1. 识别回答问题需要的关键步骤
2. 将查询扩展为"需要先...然后...最后..."的形式
3. 使推理过程清晰明确
4. 直接输出改写后的查询

示例：
原始：美国科技行业投资风险如何？
改写：请先分析美国科技行业的当前状况，然后识别存在的主要风险因素，最后评估整体投资风险水平。"""),
            ("human", "{query}"),
        ])

        chain = prompt | self.llm | StrOutputParser()

        cot_query = chain.invoke({"query": query}).strip()

        print(f"   原始查询: {query}")
        print(f"   CoT查询: {cot_query}")

        return cot_query
