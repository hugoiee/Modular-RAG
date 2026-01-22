"""
LLM Generator Operators（LLM 生成器）

支持不同的 LLM 和生成策略
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qwq import ChatQwen
from .base import BaseGenerationOperator


class LLMGeneratorOperator(BaseGenerationOperator):
    """
    基础 LLM Generator 操作器

    功能：
    - 调用 LLM 生成答案
    - 支持多种模型配置
    - 处理生成参数

    应用场景：
    - 标准文本生成
    - RAG 问答
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)
        self.top_p = self.config.get("top_p", 0.9)

        # 初始化 LLM
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        生成答案

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数（如 prompt_template）

        Returns:
            生成的答案
        """
        # 获取提示（可能从 kwargs 传入）
        prompt_text = kwargs.get("prompt", None)

        if prompt_text is None:
            # 使用默认提示格式
            prompt_text = self._build_default_prompt(query, context)

        # 创建 prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的AI助手。"),
            ("human", prompt_text)
        ])

        # 生成
        chain = prompt_template | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({})
            return answer.strip()
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return f"抱歉，生成答案时出现错误：{str(e)}"

    def _build_default_prompt(self, query: str, context: List[Document]) -> str:
        """构建默认提示"""
        if context:
            context_text = "\n\n".join([f"[文档 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context)])
            return f"""基于以下上下文信息回答问题：

上下文：
{context_text}

问题：{query}

答案："""
        else:
            return f"问题：{query}\n\n答案："


class StreamGeneratorOperator(BaseGenerationOperator):
    """
    Stream Generator 操作器（流式生成）

    功能：
    - 流式输出答案
    - 实时显示生成过程
    - 更好的用户体验

    应用场景：
    - 需要实时反馈
    - 长文本生成
    - 交互式应用
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen-plus")
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 2000)

        # 初始化 LLM（启用流式）
        self.llm = ChatQwen(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            streaming=True,
        )

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        流式生成答案

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            完整的生成答案
        """
        prompt_text = kwargs.get("prompt", self._build_default_prompt(query, context))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的AI助手。"),
            ("human", prompt_text)
        ])

        print("\n🔄 开始流式生成...")
        print("-" * 60)

        # 收集流式输出
        full_response = []

        try:
            for chunk in self.llm.stream(prompt_template.format_messages()):
                content = chunk.content
                print(content, end="", flush=True)
                full_response.append(content)

            print("\n" + "-" * 60)
            print("✅ 生成完成\n")

            return "".join(full_response).strip()

        except Exception as e:
            print(f"\n❌ 流式生成失败: {e}")
            return f"抱歉，生成答案时出现错误：{str(e)}"

    def _build_default_prompt(self, query: str, context: List[Document]) -> str:
        """构建默认提示"""
        if context:
            context_text = "\n\n".join([f"[文档 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context)])
            return f"""基于以下上下文信息回答问题：

上下文：
{context_text}

问题：{query}

答案："""
        else:
            return f"问题：{query}\n\n答案："


class EnsembleGeneratorOperator(BaseGenerationOperator):
    """
    Ensemble Generator 操作器（集成生成）

    功能：
    - 使用多个 LLM 生成答案
    - 融合多个回答
    - 提高答案质量和鲁棒性

    应用场景：
    - 高质量要求
    - 需要多样性
    - 关键任务
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.models = self.config.get("models", ["qwen-plus", "qwen-max"])
        self.temperature = self.config.get("temperature", 0.7)
        self.fusion_strategy = self.config.get("fusion_strategy", "voting")  # voting 或 concatenate

        # 初始化多个 LLM
        self.llms = [
            ChatQwen(model=model, temperature=self.temperature)
            for model in self.models
        ]

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        集成生成答案

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            融合后的答案
        """
        prompt_text = kwargs.get("prompt", self._build_default_prompt(query, context))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的AI助手。"),
            ("human", prompt_text)
        ])

        print(f"🔄 使用 {len(self.llms)} 个模型生成答案...")

        # 从每个模型生成答案
        answers = []
        for i, llm in enumerate(self.llms, 1):
            try:
                chain = prompt_template | llm | StrOutputParser()
                answer = chain.invoke({})
                answers.append(answer.strip())
                print(f"   ✓ 模型 {i} 完成")
            except Exception as e:
                print(f"   ✗ 模型 {i} 失败: {e}")

        if not answers:
            return "抱歉，所有模型都生成失败。"

        # 融合答案
        if self.fusion_strategy == "voting":
            # 简单的投票：返回最长的答案（假设更详细）
            final_answer = max(answers, key=len)
        elif self.fusion_strategy == "concatenate":
            # 连接所有答案
            final_answer = self._concatenate_answers(answers)
        else:
            # 默认返回第一个
            final_answer = answers[0]

        print(f"✅ 集成完成")

        return final_answer

    def _build_default_prompt(self, query: str, context: List[Document]) -> str:
        """构建默认提示"""
        if context:
            context_text = "\n\n".join([f"[文档 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context)])
            return f"""基于以下上下文信息回答问题：

上下文：
{context_text}

问题：{query}

答案："""
        else:
            return f"问题：{query}\n\n答案："

    def _concatenate_answers(self, answers: List[str]) -> str:
        """连接多个答案"""
        combined = "综合多个模型的回答：\n\n"

        for i, answer in enumerate(answers, 1):
            combined += f"模型 {i}：\n{answer}\n\n"

        combined += "综合结论：\n"
        # 简单地返回最长的答案作为综合结论
        combined += max(answers, key=len)

        return combined


class AdaptiveGeneratorOperator(BaseGenerationOperator):
    """
    Adaptive Generator 操作器（自适应生成）

    功能：
    - 根据查询复杂度选择模型
    - 动态调整生成参数
    - 优化成本和质量

    应用场景：
    - 需要平衡成本和质量
    - 查询复杂度不一
    - 资源优化
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.simple_model = self.config.get("simple_model", "qwen-turbo")
        self.complex_model = self.config.get("complex_model", "qwen-max")
        self.complexity_threshold = self.config.get("complexity_threshold", 0.6)

        # 初始化两个模型
        self.simple_llm = ChatQwen(model=self.simple_model, temperature=0.7)
        self.complex_llm = ChatQwen(model=self.complex_model, temperature=0.7)

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        自适应生成答案

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            生成的答案
        """
        # 评估查询复杂度
        complexity = self._assess_complexity(query, context)

        # 选择模型
        if complexity >= self.complexity_threshold:
            llm = self.complex_llm
            model_type = "复杂模型"
        else:
            llm = self.simple_llm
            model_type = "简单模型"

        print(f"📊 查询复杂度: {complexity:.2f}")
        print(f"🤖 选择模型: {model_type}")

        # 生成答案
        prompt_text = kwargs.get("prompt", self._build_default_prompt(query, context))

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的AI助手。"),
            ("human", prompt_text)
        ])

        try:
            chain = prompt_template | llm | StrOutputParser()
            answer = chain.invoke({})
            return answer.strip()
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return f"抱歉，生成答案时出现错误：{str(e)}"

    def _assess_complexity(self, query: str, context: List[Document]) -> float:
        """
        评估查询复杂度

        Args:
            query: 查询
            context: 上下文

        Returns:
            复杂度分数（0-1）
        """
        complexity = 0.0

        # 查询长度
        if len(query) > 100:
            complexity += 0.3
        elif len(query) > 50:
            complexity += 0.2
        else:
            complexity += 0.1

        # 上下文数量
        if context and len(context) > 5:
            complexity += 0.3
        elif context and len(context) > 2:
            complexity += 0.2
        else:
            complexity += 0.1

        # 复杂关键词
        complex_keywords = ["比较", "分析", "评估", "综合", "详细", "解释", "为什么"]
        if any(kw in query for kw in complex_keywords):
            complexity += 0.3

        return min(complexity, 1.0)

    def _build_default_prompt(self, query: str, context: List[Document]) -> str:
        """构建默认提示"""
        if context:
            context_text = "\n\n".join([f"[文档 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context)])
            return f"""基于以下上下文信息回答问题：

上下文：
{context_text}

问题：{query}

答案："""
        else:
            return f"问题：{query}\n\n答案："
