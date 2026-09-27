from mas_framework.prompt.prompt_set_registry import PromptSetRegistry

# Each prompt set registers itself with a @PromptSetRegistry.register('<domain>')
# decorator, and a decorator only runs when its module is imported. So every
# supported domain MUST be imported here or PromptSetRegistry.get(domain) raises
# RegistryKeyError -- which is exactly what AQuA did, since it was the one
# module missing from this list.
from mas_framework.prompt.mmlu_prompt_set import MMLUPromptSet
from mas_framework.prompt.humaneval_prompt_set import HumanEvalPromptSet
from mas_framework.prompt.gsm8k_prompt_set import GSM8KPromptSet, MixChainPromptSet
from mas_framework.prompt.AQuA_prompt_set import AQUAPromptSet

__all__ = ['MMLUPromptSet',
           'HumanEvalPromptSet',
           'GSM8KPromptSet',
           'MixChainPromptSet',
           'AQUAPromptSet',
           'PromptSetRegistry',]