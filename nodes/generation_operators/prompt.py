"""
Prompt Engineering Operators（提示工程）

提供多种提示模板和策略
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from .base import BaseGenerationOperator


class PromptTemplateOperator(BaseGenerationOperator):
    """
    基础 Prompt Template 操作器

    功能：
    - 管理和格式化提示模板
    - 整合查询和上下文
    - 支持自定义模板

    应用场景：
    - 标准 RAG 问答
    - 需要一致的提示格式
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.template = self.config.get("template", self._default_template())
        self.include_sources = self.config.get("include_sources", False)

    def _default_template(self) -> str:
        """默认提示模板"""
        return """你是一个专业的AI助手。请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请提供准确、详细的答案。如果上下文中没有相关信息，请明确说明。

答案："""

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        构建提示

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            格式化的提示文本
        """
        # 格式化上下文
        context_text = self._format_context(context) if context else "暂无相关上下文信息。"

        # 构建提示
        prompt_template = PromptTemplate.from_template(self.template)
        prompt = prompt_template.format(
            query=query,
            context=context_text
        )

        return prompt

    def _format_context(self, documents: List[Document]) -> str:
        """
        格式化上下文文档

        Args:
            documents: 文档列表

        Returns:
            格式化的上下文文本
        """
        formatted_parts = []

        for i, doc in enumerate(documents, 1):
            content = doc.page_content

            if self.include_sources:
                source = doc.metadata.get("source", "未知来源")
                formatted_parts.append(f"[文档 {i}] (来源: {source})\n{content}")
            else:
                formatted_parts.append(f"[文档 {i}]\n{content}")

        return "\n\n".join(formatted_parts)


class ContextualPromptOperator(BaseGenerationOperator):
    """
    Contextual Prompt 操作器（上下文感知提示）

    功能：
    - 根据上下文数量和质量调整提示
    - 动态生成适合的提示策略
    - 优化上下文利用

    应用场景：
    - 上下文数量不确定
    - 需要自适应提示
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.max_context_length = self.config.get("max_context_length", 3000)
        self.prioritize_recent = self.config.get("prioritize_recent", True)

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        构建上下文感知的提示

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            格式化的提示文本
        """
        if not context or len(context) == 0:
            # 无上下文情况
            template = """你是一个AI助手。用户提出了以下问题：

{query}

注意：没有找到相关的上下文信息，请基于你的知识回答，并说明这是基于通用知识的回答。

答案："""
            return template.format(query=query)

        # 有上下文的情况
        context_quality = self._assess_context_quality(context)

        if context_quality == "high":
            # 高质量上下文：强调使用上下文
            system_msg = "你是一个专业的AI助手。请严格基于提供的上下文信息回答问题，不要使用上下文之外的信息。"
        elif context_quality == "medium":
            # 中等质量：平衡上下文和知识
            system_msg = "你是一个AI助手。请主要基于提供的上下文信息回答，必要时可以补充相关背景知识。"
        else:
            # 低质量：警告可能不相关
            system_msg = "你是一个AI助手。提供的上下文可能不太相关，请谨慎使用，必要时可以主要依靠你的知识回答。"

        # 格式化上下文（可能需要截断）
        context_text = self._format_and_truncate_context(context)

        template = f"""{system_msg}

上下文信息：
{context_text}

用户问题：{{query}}

答案："""

        return template.format(query=query)

    def _assess_context_quality(self, documents: List[Document]) -> str:
        """
        评估上下文质量

        Args:
            documents: 文档列表

        Returns:
            质量等级：high, medium, low
        """
        if len(documents) >= 3:
            # 假设有足够多的文档
            avg_length = sum(len(doc.page_content) for doc in documents) / len(documents)
            if avg_length > 200:
                return "high"
            elif avg_length > 100:
                return "medium"
            else:
                return "low"
        elif len(documents) >= 1:
            return "medium"
        else:
            return "low"

    def _format_and_truncate_context(self, documents: List[Document]) -> str:
        """
        格式化并截断上下文（如果太长）

        Args:
            documents: 文档列表

        Returns:
            格式化的上下文
        """
        formatted_parts = []
        current_length = 0

        for i, doc in enumerate(documents, 1):
            content = doc.page_content

            # 检查是否会超出长度限制
            if current_length + len(content) > self.max_context_length:
                # 截断这个文档
                remaining = self.max_context_length - current_length
                if remaining > 100:  # 至少保留100字符
                    content = content[:remaining] + "..."
                    formatted_parts.append(f"[文档 {i}]\n{content}")
                break

            formatted_parts.append(f"[文档 {i}]\n{content}")
            current_length += len(content)

        return "\n\n".join(formatted_parts)


class ChainOfThoughtPromptOperator(BaseGenerationOperator):
    """
    Chain-of-Thought Prompt 操作器（思维链提示）

    功能：
    - 引导 LLM 进行逐步推理
    - 提高复杂问题的回答质量
    - 增强可解释性

    应用场景：
    - 需要多步推理的问题
    - 复杂的分析任务
    - 需要展示推理过程
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.steps = self.config.get("steps", ["理解问题", "分析上下文", "推理", "得出结论"])

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        构建 CoT 提示

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            CoT 提示文本
        """
        # 格式化上下文
        context_text = self._format_context(context) if context else "暂无上下文信息。"

        # 构建思维链提示
        steps_text = "\n".join([f"{i}. {step}" for i, step in enumerate(self.steps, 1)])

        template = f"""你是一个擅长逐步推理的AI助手。请按照以下步骤仔细分析并回答问题：

推理步骤：
{steps_text}

上下文信息：
{context_text}

用户问题：{{query}}

请按照上述步骤逐步展开你的推理过程，最后给出明确的答案。

推理过程："""

        return template.format(query=query)

    def _format_context(self, documents: List[Document]) -> str:
        """格式化上下文"""
        if not documents:
            return "暂无上下文信息。"

        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            formatted_parts.append(f"[文档 {i}] {doc.page_content}")

        return "\n\n".join(formatted_parts)


class FewShotPromptOperator(BaseGenerationOperator):
    """
    Few-Shot Prompt 操作器（少样本提示）

    功能：
    - 提供示例来引导 LLM
    - 定义期望的输出格式
    - 提高输出一致性

    应用场景：
    - 需要特定格式输出
    - 复杂的结构化任务
    - 需要风格一致性
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.examples = self.config.get("examples", self._default_examples())
        self.max_examples = self.config.get("max_examples", 2)

    def _default_examples(self) -> List[Dict[str, str]]:
        """默认示例"""
        return [
            {
                "query": "什么是机器学习？",
                "context": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习而无需明确编程。",
                "answer": "机器学习是人工智能的一个重要分支。它的核心思想是让计算机通过数据学习，而不是依赖明确的编程指令。这种方法使得计算机能够自动改进性能并适应新情况。"
            }
        ]

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        构建 Few-Shot 提示

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            Few-Shot 提示文本
        """
        # 构建示例部分
        examples_text = self._format_examples()

        # 格式化当前上下文
        context_text = self._format_context(context) if context else "暂无上下文信息。"

        template = f"""你是一个AI助手。以下是一些示例，展示了如何基于上下文回答问题：

{examples_text}

现在，请用同样的方式回答以下问题：

上下文信息：
{context_text}

用户问题：{{query}}

答案："""

        return template.format(query=query)

    def _format_examples(self) -> str:
        """格式化示例"""
        formatted_examples = []

        for i, example in enumerate(self.examples[:self.max_examples], 1):
            example_text = f"""示例 {i}:
问题：{example['query']}
上下文：{example['context']}
答案：{example['answer']}"""
            formatted_examples.append(example_text)

        return "\n\n".join(formatted_examples)

    def _format_context(self, documents: List[Document]) -> str:
        """格式化上下文"""
        if not documents:
            return "暂无上下文信息。"

        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            formatted_parts.append(f"{doc.page_content}")

        return "\n".join(formatted_parts)


class InstructPromptOperator(BaseGenerationOperator):
    """
    Instruct Prompt 操作器（指令提示）

    功能：
    - 明确的任务指令
    - 约束和要求
    - 输出格式规范

    应用场景：
    - 需要严格遵守要求
    - 特定任务类型
    - 结构化输出
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.instructions = self.config.get("instructions", [])
        self.constraints = self.config.get("constraints", [])
        self.output_format = self.config.get("output_format", None)

    def execute(self, query: str, context: List[Document] = None, **kwargs) -> str:
        """
        构建指令提示

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 额外参数

        Returns:
            指令提示文本
        """
        # 构建指令部分
        instructions_text = self._format_instructions()

        # 构建约束部分
        constraints_text = self._format_constraints()

        # 构建输出格式部分
        format_text = self._format_output_format()

        # 格式化上下文
        context_text = self._format_context(context) if context else "暂无上下文信息。"

        # 组合提示
        parts = ["你是一个AI助手。请严格按照以下要求完成任务："]

        if instructions_text:
            parts.append(f"\n任务指令：\n{instructions_text}")

        if constraints_text:
            parts.append(f"\n约束条件：\n{constraints_text}")

        if format_text:
            parts.append(f"\n输出格式：\n{format_text}")

        parts.append(f"\n上下文信息：\n{context_text}")
        parts.append(f"\n用户问题：{{query}}")
        parts.append("\n答案：")

        template = "".join(parts)
        return template.format(query=query)

    def _format_instructions(self) -> str:
        """格式化指令"""
        if not self.instructions:
            return ""
        return "\n".join([f"- {inst}" for inst in self.instructions])

    def _format_constraints(self) -> str:
        """格式化约束"""
        if not self.constraints:
            return ""
        return "\n".join([f"- {cons}" for cons in self.constraints])

    def _format_output_format(self) -> str:
        """格式化输出格式"""
        if not self.output_format:
            return ""
        return self.output_format

    def _format_context(self, documents: List[Document]) -> str:
        """格式化上下文"""
        if not documents:
            return "暂无上下文信息。"

        formatted_parts = []
        for i, doc in enumerate(documents, 1):
            formatted_parts.append(f"[文档 {i}] {doc.page_content}")

        return "\n\n".join(formatted_parts)
