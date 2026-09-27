"""MixChain uses GSM8K's agent roles verbatim.

Re-exported rather than copied on purpose: a designer trained on MixChain is
evaluated on gsm8k/aqua/multiarith/svamp, and that only works if every one of
them shares the same role vocabulary. Duplicating the dict would let the two
drift and silently break the transfer.
"""
from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION

__all__ = ['ROLE_DESCRIPTION']
