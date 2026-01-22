"""
后处理操作器 (Post-processing Operators)

实现论文中的后处理技术：
1. Output Formatting: 输出格式化
2. Citation: 引用标注
3. Answer Refinement: 答案精炼
4. Summary: 摘要生成
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from .base import BaseGenerationOperator


class OutputFormatterOperator(BaseGenerationOperator):
    """
    输出格式化操作器

    将生成的答案格式化为指定格式
    支持: markdown, json, plain, structured
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.format_type = self.config.get("format", "markdown")
        self.add_metadata = self.config.get("add_metadata", False)

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> str:
        """
        执行格式化

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            格式化后的答案
        """
        answer = kwargs.get("answer", "")

        if self.format_type == "markdown":
            return self._format_markdown(answer, query, context)
        elif self.format_type == "json":
            return self._format_json(answer, query, context)
        elif self.format_type == "structured":
            return self._format_structured(answer, query, context)
        else:
            return answer

    def _format_markdown(
        self,
        answer: str,
        query: str,
        context: List[Document] = None
    ) -> str:
        """格式化为 Markdown"""
        output = f"# 问题\n\n{query}\n\n"
        output += f"## 答案\n\n{answer}\n\n"

        if self.add_metadata and context:
            output += f"## 来源\n\n"
            for i, doc in enumerate(context[:3], 1):
                source = doc.metadata.get("source", "未知")
                output += f"{i}. {source}\n"

        return output

    def _format_json(
        self,
        answer: str,
        query: str,
        context: List[Document] = None
    ) -> str:
        """格式化为 JSON"""
        import json

        result = {
            "query": query,
            "answer": answer
        }

        if self.add_metadata and context:
            result["sources"] = [
                {
                    "index": i,
                    "source": doc.metadata.get("source", "未知"),
                    "content": doc.page_content[:100] + "..."
                }
                for i, doc in enumerate(context[:3])
            ]

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _format_structured(
        self,
        answer: str,
        query: str,
        context: List[Document] = None
    ) -> str:
        """格式化为结构化文本"""
        output = "=" * 60 + "\n"
        output += "问题：\n" + query + "\n"
        output += "-" * 60 + "\n"
        output += "答案：\n" + answer + "\n"

        if self.add_metadata and context:
            output += "-" * 60 + "\n"
            output += "参考来源：\n"
            for i, doc in enumerate(context[:3], 1):
                source = doc.metadata.get("source", "未知")
                output += f"  [{i}] {source}\n"

        output += "=" * 60 + "\n"
        return output


class CitationOperator(BaseGenerationOperator):
    """
    引用标注操作器

    在答案中添加引用标注，标明信息来源
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.citation_style = self.config.get("style", "inline")  # inline, footnote, numbered
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM（用于智能引用标注）"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.config.get("model", "qwen-plus"),
                temperature=0.0
            )
        except Exception as e:
            print(f"⚠️  初始化 LLM 失败: {e}")
            return None

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> str:
        """
        执行引用标注

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            添加引用后的答案
        """
        answer = kwargs.get("answer", "")

        if not context:
            return answer

        if self.llm:
            return self._add_smart_citations(answer, context)
        else:
            return self._add_simple_citations(answer, context)

    def _add_smart_citations(self, answer: str, context: List[Document]) -> str:
        """使用 LLM 智能添加引用"""
        # 构建引用信息
        sources = []
        for i, doc in enumerate(context, 1):
            source = doc.metadata.get("source", f"文档{i}")
            sources.append(f"[{i}] {source}: {doc.page_content[:200]}")

        sources_text = "\n".join(sources)

        prompt = f"""请在答案中添加引用标注。对于答案中的每个事实陈述，如果能在来源中找到支持，请在句尾添加 [数字] 标注。

答案：
{answer}

可用来源：
{sources_text}

请返回添加了引用标注的答案。保持答案内容不变，只添加引用标记。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            cited_answer = response.content

            # 添加引用列表
            if self.citation_style == "footnote":
                cited_answer += "\n\n---\n参考文献：\n"
                for i, doc in enumerate(context, 1):
                    source = doc.metadata.get("source", f"文档{i}")
                    cited_answer += f"[{i}] {source}\n"

            return cited_answer
        except Exception as e:
            print(f"⚠️  智能引用失败: {e}")
            return self._add_simple_citations(answer, context)

    def _add_simple_citations(self, answer: str, context: List[Document]) -> str:
        """简单引用：在答案末尾添加来源列表"""
        cited_answer = answer + "\n\n"

        if self.citation_style == "numbered":
            cited_answer += "参考来源：\n"
            for i, doc in enumerate(context, 1):
                source = doc.metadata.get("source", f"文档{i}")
                cited_answer += f"[{i}] {source}\n"
        elif self.citation_style == "inline":
            sources = [doc.metadata.get("source", f"文档{i}") for i, doc in enumerate(context, 1)]
            cited_answer += f"（来源：{', '.join(sources)}）"

        return cited_answer


class AnswerRefinementOperator(BaseGenerationOperator):
    """
    答案精炼操作器

    使用 LLM 对生成的答案进行精炼和改进
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.refinement_goals = self.config.get(
            "goals",
            ["clarity", "conciseness", "completeness"]
        )
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.config.get("model", "qwen-plus"),
                temperature=0.3
            )
        except Exception as e:
            print(f"⚠️  初始化 LLM 失败: {e}")
            return None

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> str:
        """
        执行答案精炼

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            精炼后的答案
        """
        answer = kwargs.get("answer", "")

        if not self.llm:
            return answer

        # 构建精炼目标描述
        goals_desc = self._build_goals_description()

        prompt = f"""请精炼以下答案，使其更加{goals_desc}。

原始问题：{query}

原始答案：
{answer}

要求：
- 保持原答案的核心信息和准确性
- 改善表达方式和结构
- 确保逻辑清晰、语言流畅

请直接返回精炼后的答案，不要添加额外说明。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"⚠️  答案精炼失败: {e}")
            return answer

    def _build_goals_description(self) -> str:
        """构建精炼目标描述"""
        goal_map = {
            "clarity": "清晰易懂",
            "conciseness": "简洁明了",
            "completeness": "完整全面",
            "professional": "专业准确",
            "engaging": "生动有趣"
        }

        descriptions = [goal_map.get(g, g) for g in self.refinement_goals]
        return "、".join(descriptions)


class SummaryGeneratorOperator(BaseGenerationOperator):
    """
    摘要生成操作器

    为长答案生成简洁摘要
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.summary_length = self.config.get("length", "short")  # short, medium, long
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.config.get("model", "qwen-plus"),
                temperature=0.3
            )
        except Exception as e:
            print(f"⚠️  初始化 LLM 失败: {e}")
            return None

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> str:
        """
        执行摘要生成

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            摘要 + 原答案
        """
        answer = kwargs.get("answer", "")

        if not self.llm or len(answer) < 200:
            return answer

        # 根据长度要求设置字数限制
        word_limit = {
            "short": 50,
            "medium": 100,
            "long": 200
        }.get(self.summary_length, 100)

        prompt = f"""请为以下答案生成一个简洁的摘要（不超过{word_limit}字）。

问题：{query}

答案：
{answer}

请只返回摘要内容，不要添加"摘要："等前缀。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content

            # 组合摘要和原答案
            result = f"**摘要：**\n{summary}\n\n---\n\n**详细答案：**\n{answer}"
            return result
        except Exception as e:
            print(f"⚠️  摘要生成失败: {e}")
            return answer


class StructuredOutputOperator(BaseGenerationOperator):
    """
    结构化输出操作器

    将答案转换为结构化格式（如：要点列表、表格等）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.structure_type = self.config.get("type", "bullet")  # bullet, numbered, table
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.config.get("model", "qwen-plus"),
                temperature=0.0
            )
        except Exception as e:
            print(f"⚠️  初始化 LLM 失败: {e}")
            return None

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> str:
        """
        执行结构化输出

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            结构化后的答案
        """
        answer = kwargs.get("answer", "")

        if not self.llm:
            return answer

        structure_instructions = {
            "bullet": "使用无序列表（-）的形式重新组织答案",
            "numbered": "使用有序列表（1. 2. 3.）的形式重新组织答案",
            "table": "使用 Markdown 表格的形式重新组织答案"
        }

        instruction = structure_instructions.get(
            self.structure_type,
            structure_instructions["bullet"]
        )

        prompt = f"""请{instruction}。

原始答案：
{answer}

要求：
- 保持所有关键信息
- 提高可读性和条理性
- 使用清晰的结构

请直接返回重新组织后的答案。"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"⚠️  结构化输出失败: {e}")
            return answer
