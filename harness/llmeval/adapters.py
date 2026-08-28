"""
Provider adapters.

Every backend returns the same LLMResponse object, so the pipeline code never
knows or cares whether it is talking to a paid API or a model running on your
own machine. Switching backends is a config change, not a code change.

Backends
--------
MockAdapter        no network, no cost. Use it to debug the harness.
OllamaAdapter      local models. Free. Exact token counts from the runtime.
OpenAICompatAdapter  OpenAI, OpenRouter, vLLM, LM Studio, Together, ...
AnthropicAdapter   Claude API, with extended-thinking accounting.

Token accounting note
---------------------
`reasoning_tokens` is the field your budget-matching analysis depends on, and
not every backend reports it honestly. Where a backend gives an exact count we
use it and set reasoning_tokens_exact=True. Where it does not, we fall back to
a character-based estimate over the visible reasoning text and set the flag
False. Report both in the paper; Tran & Kiela (2026) found API-level budget
accounting can be leaky, and you want to be able to show you checked.
"""

from __future__ import annotations

import os
import time
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any


# Rough chars-per-token for estimation fallbacks. Uzbek in Latin script
# tokenizes worse than English on most BPE vocabularies; 3.0 is a
# deliberately conservative divisor. Only used when a backend does not
# report exact counts.
CHARS_PER_TOKEN_ESTIMATE = 3.0


class DailyQuotaExceeded(Exception):
    """Raised when the API's per-DAY free quota is hit. Waiting won't help;
    the run should stop cleanly and resume after the quota resets."""


@dataclass
class LLMResponse:
    """Normalized response from any backend."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int = 0
    reasoning_text: str = ""
    reasoning_tokens_exact: bool = True
    latency_s: float = 0.0
    model: str = ""
    stop_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)  # raw goes to a separate transcript file
        return d


def _estimate_tokens(text: str) -> int:
    return max(0, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


def _parse_retry_delay(msg: str) -> float | None:
    """
    Pull a retry delay out of an API error message if present.

    Gemini 429s include e.g. 'Please retry in 48.5s' or a retryDelay: '48s'
    field. Returns seconds (capped), or None if nothing parseable.
    """
    import re
    for pat in (r"retry in ([\d.]+)s", r"retryDelay['\":\s]+([\d.]+)s",
                r"'([\d.]+)s'"):
        m = re.search(pat, msg)
        if m:
            try:
                return min(float(m.group(1)) + 2.0, 120.0)
            except ValueError:
                pass
    return None


class BaseAdapter(ABC):
    """Interface every backend implements."""

    name: str = "base"

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    def _call(
        self,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float,
        thinking_budget: int | None,
    ) -> LLMResponse:
        ...

    def generate(
        self,
        user: str,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        thinking_budget: int | None = None,
        max_retries: int = 6,
    ) -> LLMResponse:
        """
        Call the backend, retrying on transient failure.

        Rate-limit aware: on a 429 the wait is longer, because free tiers
        (e.g. Gemini's 5 requests/minute) need a real pause, not a 2-second
        backoff. Where the error carries a retry delay, we honor it.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                resp = self._call(
                    system=system,
                    user=user,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_budget=thinking_budget,
                )
                resp.latency_s = time.perf_counter() - t0
                resp.model = resp.model or self.model
                return resp
            except Exception as e:  # noqa: BLE001 - we genuinely want any error
                last_err = e
                msg = str(e)

                # A DAILY limit will not reset by waiting. Retrying just burns
                # more of an already-exhausted quota and wastes ~minutes per
                # call. Detect it and stop the whole run immediately.
                is_daily = ("PerDay" in msg or "RequestsPerDay" in msg
                            or "per day" in msg.lower())
                if is_daily:
                    raise DailyQuotaExceeded(
                        f"{self.name}/{self.model}: daily free quota reached. "
                        f"Stopping cleanly — resume after the quota resets "
                        f"(midnight US Pacific). Data so far is saved."
                    ) from e

                if attempt == max_retries - 1:
                    break
                is_rate = ("429" in msg or "quota" in msg.lower()
                           or "rate" in msg.lower() or "RESOURCE_EXHAUSTED" in msg)
                if is_rate:
                    # Per-minute limit: a real wait fixes it. Honor the API's
                    # requested delay if given, else wait a minute.
                    sleep = _parse_retry_delay(msg) or 60.0
                else:
                    sleep = (2**attempt) + random.random()
                time.sleep(sleep)
        raise RuntimeError(
            f"{self.name}/{self.model} failed after {max_retries} attempts: {last_err}"
        ) from last_err


# --------------------------------------------------------------------------
# Mock: no network, no cost, no API key. Debug your harness with this first.
# --------------------------------------------------------------------------
class MockAdapter(BaseAdapter):
    name = "mock"

    def _call(self, system, user, max_tokens, temperature, thinking_budget):
        time.sleep(0.01)
        text = f"[mock reply to {len(user)} chars of prompt]"
        reasoning = "[mock reasoning]" if thinking_budget else ""
        return LLMResponse(
            text=text,
            prompt_tokens=_estimate_tokens(system + user),
            completion_tokens=_estimate_tokens(text),
            reasoning_tokens=_estimate_tokens(reasoning),
            reasoning_text=reasoning,
            reasoning_tokens_exact=True,
            stop_reason="end_turn",
            raw={"mock": True},
        )


# --------------------------------------------------------------------------
# Ollama: fully local, free. `ollama serve` then `ollama pull qwen3:14b`.
# --------------------------------------------------------------------------
class OllamaAdapter(BaseAdapter):
    """
    Local inference. No API key, no per-token cost.

    Ollama reports exact prompt_eval_count and eval_count, so token
    accounting here is as trustworthy as any paid API — arguably more so,
    since nothing is hidden behind a service boundary.
    """

    name = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.host = host.rstrip("/")

    def _call(self, system, user, max_tokens, temperature, thinking_budget):
        import requests

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        # Ollama exposes thinking on models that support it (qwen3, deepseek-r1).
        if thinking_budget is not None:
            payload["think"] = True

        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()

        msg = data.get("message", {})
        text = msg.get("content", "")
        reasoning = msg.get("thinking", "") or ""

        return LLMResponse(
            text=text,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            # Ollama folds thinking into eval_count and does not break it out,
            # so this one is an estimate.
            reasoning_tokens=_estimate_tokens(reasoning),
            reasoning_text=reasoning,
            reasoning_tokens_exact=False,
            stop_reason=data.get("done_reason", ""),
            raw=data,
        )


# --------------------------------------------------------------------------
# OpenAI-compatible: OpenAI, OpenRouter, vLLM, LM Studio, Together, Groq...
# --------------------------------------------------------------------------
class OpenAICompatAdapter(BaseAdapter):
    name = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ.get(api_key_env, "not-needed-for-local"),
            base_url=base_url,
        )

    def _call(self, system, user, max_tokens, temperature, thinking_budget):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )

        usage = resp.usage
        reasoning_tokens = 0
        exact = True
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_tokens_exact=exact,
            stop_reason=resp.choices[0].finish_reason or "",
            raw=resp.model_dump(),
        )


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicAdapter(BaseAdapter):
    """
    Claude API.

    Extended thinking: output_tokens includes thinking tokens, and the API
    does not break them out. We therefore log the requested budget, the total
    output tokens, and a character-based estimate over the returned thinking
    blocks, flagged as inexact. Reporting the discrepancy between requested
    and effective budget is itself a finding worth including.
    """

    name = "anthropic"

    def __init__(self, model: str, api_key_env: str = "ANTHROPIC_API_KEY", **kwargs):
        super().__init__(model, **kwargs)
        import anthropic

        self.client = anthropic.Anthropic(api_key=os.environ[api_key_env])

    def _call(self, system, user, max_tokens, temperature, thinking_budget):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
        }
        if system:
            kwargs["system"] = system

        if thinking_budget:
            # max_tokens must exceed the thinking budget, and extended
            # thinking requires temperature=1.
            kwargs["max_tokens"] = max(max_tokens, thinking_budget + 512)
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            kwargs["temperature"] = 1.0
        else:
            kwargs["temperature"] = temperature

        resp = self.client.messages.create(**kwargs)

        text_parts, thinking_parts = [], []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "thinking":
                thinking_parts.append(getattr(block, "thinking", ""))

        reasoning_text = "\n".join(thinking_parts)

        return LLMResponse(
            text="\n".join(text_parts),
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            reasoning_tokens=_estimate_tokens(reasoning_text),
            reasoning_text=reasoning_text,
            reasoning_tokens_exact=False,
            stop_reason=resp.stop_reason or "",
            raw=json.loads(resp.model_dump_json()),
        )


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Gemini (Google) — free tier, no credit card.
# Uses Google's OpenAI-compatible endpoint, so it reuses the OpenAI client.
# Get a free key at https://aistudio.google.com/apikey and set it as
# the GEMINI_API_KEY environment variable.
#
# Free-tier note for your paper's methods: Google may use free-tier prompts
# for model training. Fine for public academic prompts; worth stating.
# --------------------------------------------------------------------------
class GeminiAdapter(OpenAICompatAdapter):
    name = "gemini"

    def __init__(self, model: str = "gemini-3.5-flash-lite",
                 api_key_env: str = "GEMINI_API_KEY", **kwargs):
        # Google's OpenAI-compatible base URL.
        super().__init__(
            model=model,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env=api_key_env,
            **kwargs,
        )


ADAPTERS = {
    "mock": MockAdapter,
    "ollama": OllamaAdapter,
    "openai_compat": OpenAICompatAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
}


def build_adapter(backend: str, model: str, **kwargs) -> BaseAdapter:
    if backend not in ADAPTERS:
        raise ValueError(f"Unknown backend {backend!r}. Options: {list(ADAPTERS)}")
    return ADAPTERS[backend](model=model, **kwargs)
