"""OpenAI-compatible chat client.

Works against the real OpenAI API and against any OpenAI-compatible server,
notably a local vLLM host serving e.g. ``Qwen/Qwen2.5-7B-Instruct``.

Environment (.env):
    BASE_URL          e.g. http://localhost:8000/v1  (empty -> real OpenAI)
    API_KEY           vLLM ignores the value but the client requires one
    LLM_MAX_CONCURRENCY   cap on in-flight requests (default 32)
    LLM_TIMEOUT           per-request timeout in seconds (default 180)
    LLM_MAX_RETRIES       retry attempts on transient failures (default 5)
    LLM_MAX_TOKENS        generation cap when the caller sets none (default 4096)
"""
import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from mas_framework.llm.format import Message
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry
from mas_framework.llm.price import cost_count, record_usage

# Resolve .env from the repo root explicitly. Bare load_dotenv() walks up from
# the *calling module's* directory, which silently finds nothing depending on how
# a script was launched -- and a missing BASE_URL does not fail, it quietly points
# the client at api.openai.com. Shell exports still win: load_dotenv does not
# override variables already set.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DOTENV = os.path.join(_REPO_ROOT, '.env')
load_dotenv(_DOTENV if os.path.exists(_DOTENV) else None)

MINE_BASE_URL = os.getenv("BASE_URL") or None          # None -> default OpenAI endpoint
MINE_API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"


def _check_backend(model: str) -> None:
    """Refuse to send a self-hosted model id to OpenAI with a placeholder key.

    Without this the failure surfaces as a 401 from api.openai.com several
    minutes into a run, which reads like a credentials problem rather than what
    it is: BASE_URL never reached the client.
    """
    if MINE_BASE_URL or model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return
    raise RuntimeError("\n".join([
        f"BASE_URL is not set, so requests for '{model}' would go to api.openai.com.",
        f"  Looked for: {_DOTENV}",
        f"  Fix: cp template.env .env   (BASE_URL=http://localhost:8000/v1)",
        f"       or: export BASE_URL=http://localhost:8000/v1 API_KEY=EMPTY",
        f"  Verify: curl http://localhost:8000/v1/models   (must list '{model}')",
    ]))

MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "32"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))
# Generation cap when a caller does not set one. Without it max_tokens is None,
# so vLLM generates until --max-model-len (16384). A model that fails to stop --
# routine on competition-level math, where it loops on an algebra step -- then
# spends minutes on ONE request, blows past LLM_TIMEOUT, and is RETRIED up to
# MAX_RETRIES times with exponential backoff. Since every batch is an
# asyncio.gather barrier, that single query freezes the whole batch for 15+
# minutes while the GPU sits idle through the backoff sleeps.
# 4096 is far above a normal solution, so it truncates runaways only.
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# A single shared client: creating one per request leaks connections and is the
# fastest way to make a local vLLM host fall over under the evaluation load.
_client: Optional[AsyncOpenAI] = None
_client_loop: Optional[asyncio.AbstractEventLoop] = None
_semaphore: Optional[asyncio.Semaphore] = None
_sem_loop: Optional[asyncio.AbstractEventLoop] = None


def get_client() -> AsyncOpenAI:
    """Return a per-event-loop AsyncOpenAI client.

    The pipeline calls ``asyncio.run`` several times in one process, and an
    httpx client bound to a closed loop raises on its next use, so the client is
    rebuilt whenever the running loop changes.
    """
    global _client, _client_loop
    loop = asyncio.get_event_loop()
    if _client is None or _client_loop is not loop:
        _client = AsyncOpenAI(
            base_url=MINE_BASE_URL,
            api_key=MINE_API_KEY,
            timeout=REQUEST_TIMEOUT,
            max_retries=0,          # tenacity owns retries
        )
        _client_loop = loop
    return _client


def get_semaphore() -> asyncio.Semaphore:
    """Concurrency cap, rebuilt per event loop for the same reason as the client."""
    global _semaphore, _sem_loop
    loop = asyncio.get_event_loop()
    if _semaphore is None or _sem_loop is not loop:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        _sem_loop = loop
    return _semaphore


def _to_dicts(messages: Union[str, List[Any]]) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages:
        if isinstance(m, Message):
            out.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:  # dataclass-like with .role/.content
            out.append({"role": m.role, "content": m.content})
    return out


def _log_retry(state):
    """A stalled batch should say so.

    Retries happen inside asyncio.gather, so without this the only symptom is a
    progress bar that stops moving and a vLLM log reading "Running: 0 reqs" --
    which looks like a dead server rather than a client sleeping in backoff.
    """
    exc = state.outcome.exception() if state.outcome else None
    print(f"[llm] attempt {state.attempt_number}/{MAX_RETRIES} failed "
          f"({type(exc).__name__}: {exc}); retrying in "
          f"{getattr(state, 'idle_for', 0):.1f}s", flush=True)


@retry(
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(Exception),
    before_sleep=_log_retry,
    reraise=True,
)
async def achat(
    model: str,
    msg: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.2,
    num_comps: Optional[int] = 1,
) -> str:
    _check_backend(model)
    async with get_semaphore():
        completion = await get_client().chat.completions.create(
            messages=msg,
            model=model,
            max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
            temperature=temperature,
            n=num_comps or 1,
        )

    response = completion.choices[0].message.content or ""

    # Token accounting. Prefer the server's own usage numbers (exact for vLLM);
    # fall back to the tiktoken estimate only when the server omits them.
    usage = getattr(completion, "usage", None)
    if usage is not None and getattr(usage, "prompt_tokens", None) is not None:
        record_usage(model, usage.prompt_tokens, usage.completion_tokens or 0)
    else:
        prompt_text = "\n".join(m["content"] for m in msg)
        cost_count(prompt_text, response, model)

    return response


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        return await achat(self.model_name, _to_dicts(messages), max_tokens, temperature, num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        return asyncio.run(self.agen(messages, max_tokens, temperature, num_comps))
