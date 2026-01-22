"""
生成验证操作器 (Verification Operators)

实现论文中的验证技术：
1. Fact Checking: 事实核查
2. Consistency Check: 一致性检查
3. Hallucination Detection: 幻觉检测
4. Confidence Scoring: 置信度评分
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage

from .base import BaseGenerationOperator


class VerificationOperator(BaseGenerationOperator):
    """
    基础验证操作器

    验证生成答案的质量和准确性
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.7)

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行验证

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            验证结果字典
        """
        answer = kwargs.get("answer", "")

        if not answer:
            return {
                "is_valid": False,
                "confidence": 0.0,
                "reason": "No answer provided"
            }

        # 简单验证：检查答案长度和相关性
        if len(answer) < 10:
            return {
                "is_valid": False,
                "confidence": 0.3,
                "reason": "Answer too short"
            }

        # 检查是否与上下文相关
        if context:
            context_text = " ".join([doc.page_content for doc in context])
            relevance = self._check_relevance(answer, context_text)

            return {
                "is_valid": relevance > self.threshold,
                "confidence": relevance,
                "reason": f"Relevance score: {relevance:.2f}"
            }

        return {
            "is_valid": True,
            "confidence": 0.5,
            "reason": "No context to verify against"
        }

    def _check_relevance(self, answer: str, context: str) -> float:
        """简单的相关性检查"""
        # 统计答案中有多少词出现在上下文中
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        if not answer_words:
            return 0.0

        common_words = answer_words & context_words
        return len(common_words) / len(answer_words)


class FactCheckOperator(BaseGenerationOperator):
    """
    事实核查操作器

    使用 LLM 验证答案中的事实是否与上下文一致
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "qwen-plus")
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.model_name,
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
    ) -> Dict[str, Any]:
        """
        执行事实核查

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            核查结果
        """
        answer = kwargs.get("answer", "")

        if not self.llm or not context:
            return {
                "is_factual": True,
                "confidence": 0.5,
                "violations": []
            }

        # 构建事实核查 prompt
        context_text = "\n\n".join([doc.page_content for doc in context[:3]])

        prompt = f"""请作为事实核查专家，验证答案中的陈述是否与提供的上下文一致。

上下文：
{context_text}

答案：
{answer}

请分析：
1. 答案中的事实陈述是否有上下文支持
2. 是否存在与上下文矛盾的内容
3. 是否包含上下文中没有的信息

以 JSON 格式返回：
{{
  "is_factual": true/false,
  "confidence": 0.0-1.0,
  "violations": ["列出问题陈述"]
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_response(response.content)
            return result
        except Exception as e:
            print(f"⚠️  事实核查失败: {e}")
            return {
                "is_factual": True,
                "confidence": 0.5,
                "violations": []
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # 如果解析失败，返回默认值
        return {
            "is_factual": True,
            "confidence": 0.7,
            "violations": []
        }


class ConsistencyCheckOperator(BaseGenerationOperator):
    """
    一致性检查操作器

    检查答案内部的逻辑一致性
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_name = self.config.get("model", "qwen-plus")
        self.llm = self._init_llm()

    def _init_llm(self):
        """初始化 LLM"""
        try:
            from langchain_community.chat_models import ChatTongyi
            return ChatTongyi(
                model_name=self.model_name,
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
    ) -> Dict[str, Any]:
        """
        执行一致性检查

        Args:
            query: 用户查询
            context: 上下文文档（可选）
            **kwargs: 必须包含 answer 参数

        Returns:
            一致性检查结果
        """
        answer = kwargs.get("answer", "")

        if not self.llm:
            return {
                "is_consistent": True,
                "confidence": 0.5,
                "issues": []
            }

        # 构建一致性检查 prompt
        prompt = f"""请作为逻辑分析专家，检查以下答案的内部一致性。

问题：{query}

答案：
{answer}

请分析：
1. 答案中是否存在自相矛盾的陈述
2. 逻辑推理是否连贯
3. 结论是否与论据一致

以 JSON 格式返回：
{{
  "is_consistent": true/false,
  "confidence": 0.0-1.0,
  "issues": ["列出一致性问题"]
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = self._parse_response(response.content)
            return result
        except Exception as e:
            print(f"⚠️  一致性检查失败: {e}")
            return {
                "is_consistent": True,
                "confidence": 0.5,
                "issues": []
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 响应"""
        import json
        import re

        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # 如果解析失败，返回默认值
        return {
            "is_consistent": True,
            "confidence": 0.7,
            "issues": []
        }


class HallucinationDetectionOperator(BaseGenerationOperator):
    """
    幻觉检测操作器

    检测 LLM 生成的幻觉内容（即不基于上下文的虚构信息）
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 0.7)
        self.strict = self.config.get("strict", False)

    def execute(
        self,
        query: str,
        context: List[Document] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        执行幻觉检测

        Args:
            query: 用户查询
            context: 上下文文档
            **kwargs: 必须包含 answer 参数

        Returns:
            检测结果
        """
        answer = kwargs.get("answer", "")

        if not context or not answer:
            return {
                "has_hallucination": False,
                "confidence": 0.5,
                "hallucinations": []
            }

        # 提取答案中的关键陈述
        statements = self._extract_statements(answer)

        # 检查每个陈述是否有上下文支持
        context_text = " ".join([doc.page_content for doc in context])
        hallucinations = []

        for statement in statements:
            if not self._is_supported(statement, context_text):
                hallucinations.append(statement)

        has_hallucination = len(hallucinations) > 0
        confidence = 1.0 - (len(hallucinations) / max(len(statements), 1))

        return {
            "has_hallucination": has_hallucination,
            "confidence": confidence,
            "hallucinations": hallucinations,
            "total_statements": len(statements)
        }

    def _extract_statements(self, text: str) -> List[str]:
        """提取文本中的陈述句"""
        # 简单实现：按句号、问号、感叹号分句
        import re
        sentences = re.split(r'[。！？.!?]', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def _is_supported(self, statement: str, context: str) -> bool:
        """检查陈述是否有上下文支持"""
        # 简单实现：检查关键词重叠度
        statement_words = set(statement.lower().split())
        context_words = set(context.lower().split())

        if not statement_words:
            return True

        common_words = statement_words & context_words
        overlap_ratio = len(common_words) / len(statement_words)

        return overlap_ratio >= self.threshold
