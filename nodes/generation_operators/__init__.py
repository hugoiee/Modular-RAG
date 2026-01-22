"""
Generation Operators 包

实现论文中的生成技术：
1. Prompt Engineering（提示工程）
2. LLM Generation（LLM 生成）
3. Verification（验证）
4. Post-processing（后处理）
"""

from .base import BaseGenerationOperator
from .prompt import (
    PromptTemplateOperator,
    ContextualPromptOperator,
    ChainOfThoughtPromptOperator,
)
from .generator import (
    LLMGeneratorOperator,
    StreamGeneratorOperator,
    EnsembleGeneratorOperator,
)
from .verification import (
    VerificationOperator,
    FactCheckOperator,
    ConsistencyCheckOperator,
)
from .postprocess import (
    OutputFormatterOperator,
    CitationOperator,
    AnswerRefinementOperator,
)

__all__ = [
    "BaseGenerationOperator",
    "PromptTemplateOperator",
    "ContextualPromptOperator",
    "ChainOfThoughtPromptOperator",
    "LLMGeneratorOperator",
    "StreamGeneratorOperator",
    "EnsembleGeneratorOperator",
    "VerificationOperator",
    "FactCheckOperator",
    "ConsistencyCheckOperator",
    "OutputFormatterOperator",
    "CitationOperator",
    "AnswerRefinementOperator",
]
