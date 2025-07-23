from typing import Dict, Type

from openCHA.llms.llm_types import LLMType
from openCHA.llms.llm import BaseLLM
from openCHA.llms.openai import OpenAILLM
from openCHA.llms.anthropic import AntropicLLM
from openCHA.llms.llama import LlamaLLM

LLM_TO_CLASS: Dict[LLMType, Type[BaseLLM]] = {
    LLMType.OPENAI:    OpenAILLM,
    LLMType.ANTHROPIC: AntropicLLM,
    LLMType.LLAMA:     LlamaLLM,
}
