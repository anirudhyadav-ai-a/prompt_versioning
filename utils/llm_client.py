"""Thin async wrapper around the OpenAI API.

Provides chat completion and embedding helpers with automatic
retry and exponential backoff (3 attempts).

MultiModelClient uses litellm for native multi-provider support.
"""

from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
import openai

from .config import get_settings
from .logging_utils import get_logger

# Exceptions worth retrying — transient network / rate-limit errors.
# Authentication, validation, and permission errors should fail immediately.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

logger = get_logger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class LLMClient:
    """Async OpenAI wrapper with retry logic."""

    def __init__(self, settings=None):
        self._settings = settings or get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.api_key)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatCompletion:
        """Send a chat completion request with retry.

        *temperature* overrides the value from settings when provided.
        Pass temperature=0.0 for deterministic outputs (e.g. drift detection).
        """
        kwargs: dict[str, Any] = {
            "model": model or self._settings.model_name,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._settings.temperature
            ),
            "max_tokens": self._settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        return await self._retry(self._client.chat.completions.create, **kwargs)

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for *text*."""
        response = await self._retry(
            self._client.embeddings.create,
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    async def _retry(self, fn, **kwargs):
        """Call *fn* with exponential backoff on transient failures."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await fn(**kwargs)
            except _RETRYABLE as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "LLM call failed after %d attempts: %s", _MAX_RETRIES, exc
                    )
                    raise
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "Attempt %d failed (%s), retrying in %.1fs", attempt, exc, wait
                )
                await asyncio.sleep(wait)


class LiteLLMClient:
    """Wrapper around litellm providing the same interface as LLMClient."""

    def __init__(self, model: str, settings=None) -> None:
        self._model = model
        self._settings = settings or get_settings()

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict | None = None,
    ) -> ChatCompletion:
        """Send a chat completion via litellm with retry.

        *temperature* overrides the value from settings when provided.
        Pass temperature=0.0 for deterministic outputs (e.g. drift detection).
        """
        import litellm  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": (
                temperature if temperature is not None else self._settings.temperature
            ),
            "max_tokens": self._settings.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return await litellm.acompletion(**kwargs)
            except _RETRYABLE as exc:
                if attempt == _MAX_RETRIES:
                    logger.error(
                        "litellm call failed after %d attempts: %s", attempt, exc
                    )
                    raise
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "Attempt %d failed (%s), retrying in %.1fs", attempt, exc, wait
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                # litellm may wrap provider errors in its own types — only
                # retry if the message looks transient.
                is_retryable = (
                    "rate" in str(exc).lower() or "timeout" in str(exc).lower()
                )
                if attempt == _MAX_RETRIES or not is_retryable:
                    logger.error(
                        "litellm call failed after %d attempts: %s", attempt, exc
                    )
                    raise
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    "Attempt %d failed (%s), retrying in %.1fs", attempt, exc, wait
                )
                await asyncio.sleep(wait)
        raise RuntimeError("Unreachable")  # pragma: no cover

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector via litellm."""
        import litellm  # type: ignore[import-untyped]

        response = await litellm.aembedding(
            model="text-embedding-3-small",
            input=[text],
        )
        return response.data[0]["embedding"]


# Default model mapping for MultiModelClient
DEFAULT_MODELS: dict[str, str] = {
    "claude": "claude-sonnet-4-20250514",
    "gpt-4o": "gpt-4o",
    "gemini": "gemini/gemini-2.0-flash",
}


class MultiModelClient:
    """Wraps multiple LiteLLMClients for cross-model operations."""

    def __init__(self, models: dict[str, str] | None = None):
        """
        models: mapping of label -> litellm model string, e.g.:
            {"claude": "claude-sonnet-4-20250514", "gpt-4o": "gpt-4o",
             "gemini": "gemini/gemini-2.0-flash"}
        """
        self.models = models or dict(DEFAULT_MODELS)
        self._clients: dict[str, LiteLLMClient] = {}
        for label, model_name in self.models.items():
            self._clients[label] = LiteLLMClient(model_name)

    def get_client(self, label: str) -> LiteLLMClient:
        """Return the LiteLLMClient for a given model label."""
        return self._clients[label]

    def labels(self) -> list[str]:
        """Return all configured model labels."""
        return list(self._clients.keys())
