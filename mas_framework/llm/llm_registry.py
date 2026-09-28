from typing import Optional
from class_registry import ClassRegistry

from mas_framework.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        else:
            # Route any model name to the OpenAI-compatible client (GPTChat),
            # which talks to the vLLM server at BASE_URL. The name must be the
            # model id vLLM serves (e.g. "Qwen/Qwen2.5-7B-Instruct").
            model = cls.registry.get('GPTChat', model_name)

        return model
