"""OpenAI-compatible chat client with caching, structured output, and usage accounting."""

import json
import threading
import time
from dataclasses import dataclass, field
from typing import TypeVar

from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
)
from pydantic import BaseModel, ValidationError

from clinsynapse.config import Settings, get_settings
from clinsynapse.llm.cache import CacheMiss, ResponseCache, prompt_key

T = TypeVar("T", bound=BaseModel)

# Reference USD per 1M tokens (input, output) at public list prices, keyed by model basename.
# ARC is free for VT users, so $/1k-queries is "what this would cost on a paid provider".
# Models missing here report $0; update when prices change.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-oss-120b": (0.15, 0.75),
    "gpt-oss-20b": (0.10, 0.50),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


def _price(model: str) -> tuple[float, float]:
    return PRICES.get(model.split("/")[-1], (0.0, 0.0))


def _strip_fences(raw: str) -> str:
    """Without JSON mode some models wrap JSON in ``` fences or add prose around it."""
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end > start else raw


_SLOTS: dict[str, threading.BoundedSemaphore] = {}
_SLOTS_LOCK = threading.Lock()


def _model_slots(model: str, limit: int) -> threading.BoundedSemaphore:
    """One concurrency cap per model for the whole process.

    Providers limit concurrent requests per model (ARC: 10). Per-client caps let the agent and
    the judge together exceed that when they use the same model, which turns into 429 backoff.
    """
    with _SLOTS_LOCK:
        if model not in _SLOTS:
            _SLOTS[model] = threading.BoundedSemaphore(limit)
        return _SLOTS[model]


class LLMError(RuntimeError):
    pass


class LLMFatal(RuntimeError):
    """Retrying can't help (bad key, no access e.g. off the VT VPN, unknown model).

    Deliberately not an LLMError: nodes degrade on LLMError, but this must stop the run.
    """


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    by_node: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, model: str, prompt: int, completion: int, seconds: float, node: str) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.seconds += seconds
        p_in, p_out = _price(model)
        self.cost_usd += (prompt * p_in + completion * p_out) / 1e6
        self.by_node[node] = self.by_node.get(node, 0) + prompt + completion


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client: OpenAI | None = None
        self.cache = ResponseCache(self.settings.cache_path) if self.settings.llm_cache else None
        self._json_mode = self.settings.json_mode

    @property
    def client(self) -> OpenAI:
        # Created lazily so fully cached runs (e.g. CI smoke evals) need no API key.
        if self._client is None:
            if not self.settings.llm_api_key:
                raise LLMError("No API key set (OPEN_AI_API_KEY) and response not cached")
            self._client = OpenAI(base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key)
        return self._client

    def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        json_mode: bool = False,
        usage: Usage | None = None,
        node: str = "unknown",
        max_tokens: int = 2048,
    ) -> str:
        model = model or self.settings.main_model
        params = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
        }
        if "gpt-oss" in model:
            params["reasoning_effort"] = self.settings.reasoning_effort
        # The cache key ignores response_format so a JSON-mode fallback doesn't invalidate it.
        key = prompt_key({**params, "json": json_mode})
        if json_mode and self._json_mode:
            params["response_format"] = {"type": "json_object"}

        if self.cache and (hit := self.cache.get(key)):
            if usage:
                usage.cache_hits += 1
                usage.add(model, hit["prompt_tokens"], hit["completion_tokens"], 0.0, node)
            return hit["content"]
        if self.settings.replay_only:
            raise CacheMiss(f"LLM call not in cache (node={node}); re-record the smoke cache")

        start = time.perf_counter()
        resp = self._with_retries(params)
        content = resp.choices[0].message.content or ""
        prompt_t = resp.usage.prompt_tokens if resp.usage else 0
        completion_t = resp.usage.completion_tokens if resp.usage else 0
        if not content.strip() and resp.choices[0].finish_reason == "length":
            # Reasoning models can spend the whole budget thinking; retry once with double room.
            if usage:
                usage.add(model, prompt_t, completion_t, 0.0, node)
            resp = self._with_retries({**params, "max_tokens": min(params["max_tokens"] * 2, 8000)})
            content = resp.choices[0].message.content or ""
            prompt_t = resp.usage.prompt_tokens if resp.usage else 0
            completion_t = resp.usage.completion_tokens if resp.usage else 0
        elapsed = time.perf_counter() - start

        if usage:
            usage.add(model, prompt_t, completion_t, elapsed, node)
        if not content.strip():
            raise LLMError(f"empty completion from {model} (node={node})")  # never cache empties
        if self.cache:
            self.cache.put(
                key, {"content": content, "prompt_tokens": prompt_t, "completion_tokens": completion_t}
            )
        return content

    def structured(
        self,
        messages: list[dict],
        schema: type[T],
        *,
        model: str | None = None,
        usage: Usage | None = None,
        node: str = "unknown",
        max_tokens: int = 2048,
    ) -> T:
        """JSON-mode call validated against a Pydantic schema, with one repair attempt."""
        schema_msg = {
            "role": "system",
            "content": "Respond with a single JSON object matching this JSON Schema:\n"
            + json.dumps(schema.model_json_schema()),
        }
        msgs = [schema_msg, *messages]
        raw = self.chat(msgs, model=model, json_mode=True, usage=usage, node=node, max_tokens=max_tokens)
        try:
            return schema.model_validate_json(_strip_fences(raw))
        except ValidationError as err:
            repair = [
                *msgs,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That JSON failed validation:\n{err}\nReturn corrected JSON only."},
            ]
            raw = self.chat(repair, model=model, json_mode=True, usage=usage, node=node, max_tokens=max_tokens)
            try:
                return schema.model_validate_json(_strip_fences(raw))
            except ValidationError as err2:
                raise LLMError(f"Structured output failed for {schema.__name__}: {err2}") from err2

    def _with_retries(self, params: dict, attempts: int = 6):
        delay = 2.0
        for i in range(attempts):
            try:
                with _model_slots(params["model"], self.settings.max_concurrency):
                    return self.client.chat.completions.create(**params)
            except (AuthenticationError, PermissionDeniedError, NotFoundError) as exc:
                raise LLMFatal(f"{type(exc).__name__}: {exc}") from exc
            except BadRequestError as exc:
                # Endpoints without JSON mode reject response_format: fall back to prompt-only JSON.
                if "response_format" in params:
                    self._json_mode = False
                    params = {k: v for k, v in params.items() if k != "response_format"}
                    continue
                raise LLMError(str(exc)) from exc
            except Exception as exc:
                if i == attempts - 1:
                    raise LLMError(str(exc)) from exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise LLMError("retries exhausted")
