"""
生成模块 (Generation Module)

基于论文《Modular RAG》的三层架构设计：
- 顶层：GenerationModule（生成模块）
- 中层：生成策略（Prompt Engineering, LLM Generation, Verification）
- 底层：Operators（具体的生成技术）

核心功能：
基于检索到的上下文生成高质量答案

主要技术：
1. Prompt Engineering（提示工程）
   - Template: 模板化提示
   - Contextual: 上下文感知提示
   - CoT: 思维链提示
   - Few-Shot: 少样本提示
   - Instruct: 指令提示

2. LLM Generation（LLM 生成）
   - Standard: 标准生成
   - Stream: 流式生成
   - Ensemble: 集成生成
   - Adaptive: 自适应生成

3. Post-processing（后处理）
   - Formatting: 格式化
   - Citation: 引用标注
   - Refinement: 答案精炼
"""

from typing import List, Dict, Any
from langchain_core.documents import Document

from .generation_operators import (
    BaseGenerationOperator,
    PromptTemplateOperator,
    ContextualPromptOperator,
    ChainOfThoughtPromptOperator,
    LLMGeneratorOperator,
    StreamGeneratorOperator,
    EnsembleGeneratorOperator,
)


class GenerationModule:
    """
    生成模块（顶层）

    使用方式：
    1. 选择生成策略
    2. 提供查询和上下文
    3. 生成答案

    Example:
        config = {
            "prompt_strategy": "contextual",
            "generator": "llm",
            "model": "qwen-plus",
            "temperature": 0.7
        }

        generation = GenerationModule(config)
        answer = generation.generate(query, context_docs)
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化生成模块

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.prompt_strategy = self.config.get("prompt_strategy", "template")
        self.generator_type = self.config.get("generator", "llm")

        # 初始化 prompt operator
        self.prompt_operator = self._init_prompt_operator()

        # 初始化 generator operator
        self.generator_operator = self._init_generator_operator()

    def _init_prompt_operator(self) -> BaseGenerationOperator:
        """初始化 prompt operator"""
        strategy = self.prompt_strategy.lower()

        if strategy == "template":
            return PromptTemplateOperator(self.config)
        elif strategy == "contextual":
            return ContextualPromptOperator(self.config)
        elif strategy == "cot" or strategy == "chain_of_thought":
            return ChainOfThoughtPromptOperator(self.config)
        else:
            print(f"⚠️  未知的 prompt 策略: {strategy}，使用默认 template")
            return PromptTemplateOperator(self.config)

    def _init_generator_operator(self) -> BaseGenerationOperator:
        """初始化 generator operator"""
        gen_type = self.generator_type.lower()

        if gen_type == "llm":
            return LLMGeneratorOperator(self.config)
        elif gen_type == "stream":
            return StreamGeneratorOperator(self.config)
        elif gen_type == "ensemble":
            return EnsembleGeneratorOperator(self.config)
        else:
            print(f"⚠️  未知的 generator 类型: {gen_type}，使用默认 llm")
            return LLMGeneratorOperator(self.config)

    def generate(
        self,
        query: str,
        context: List[Document] = None,
        verbose: bool = True
    ) -> str:
        """
        生成答案

        Args:
            query: 用户查询
            context: 检索到的上下文文档
            verbose: 是否打印详细信息

        Returns:
            生成的答案
        """
        if verbose:
            print("\n" + "=" * 60)
            print(f"🤖 生成模块")
            print("=" * 60)
            print(f"Prompt 策略: {self.prompt_strategy}")
            print(f"Generator 类型: {self.generator_type}")
            if context:
                print(f"上下文文档数: {len(context)}")

        # 步骤1: 构建 prompt
        if verbose:
            print("\n📝 步骤 1: 构建 Prompt")

        prompt = self.prompt_operator.execute(query, context)

        # 步骤2: 生成答案
        if verbose:
            print("\n🔄 步骤 2: 生成答案")

        answer = self.generator_operator.execute(
            query,
            context,
            prompt=prompt
        )

        if verbose:
            print("\n" + "=" * 60)
            print("✅ 生成完成")
            print("=" * 60)

        return answer

    def change_strategy(
        self,
        prompt_strategy: str = None,
        generator_type: str = None,
        new_config: Dict[str, Any] = None
    ):
        """
        动态更换生成策略

        Args:
            prompt_strategy: 新的 prompt 策略
            generator_type: 新的 generator 类型
            new_config: 新配置
        """
        if prompt_strategy:
            self.prompt_strategy = prompt_strategy
            self.prompt_operator = self._init_prompt_operator()
            print(f"✅ 已切换 Prompt 策略: {prompt_strategy}")

        if generator_type:
            self.generator_type = generator_type
            self.generator_operator = self._init_generator_operator()
            print(f"✅ 已切换 Generator 类型: {generator_type}")

        if new_config:
            self.config.update(new_config)

    def summary(self) -> Dict[str, Any]:
        """
        返回模块摘要信息

        Returns:
            摘要字典
        """
        return {
            "module": "GenerationModule",
            "prompt_strategy": self.prompt_strategy,
            "generator_type": self.generator_type,
            "prompt_operator": self.prompt_operator.name,
            "generator_operator": self.generator_operator.name,
            "config": self.config,
        }
