"""Async OpenAI client and embedding helpers."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openai import AsyncOpenAI

_async_clients: dict[str, AsyncOpenAI] = {}
_lock = threading.Lock()


def get_async_client(api_key: str = "") -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client for the given API key."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError(
            "Async LLM features require the openai package. "
            "Install with: pip install distillcore[openai]"
        ) from None

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    with _lock:
        if key not in _async_clients:
            _async_clients[key] = AsyncOpenAI(api_key=key)
        return _async_clients[key]


async def embed_texts_async(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_key: str = "",
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[list[float]]:
    """Async batch-embed texts. Uses embed_fn if provided, otherwise OpenAI."""
    if embed_fn is not None:
        if asyncio.iscoroutinefunction(embed_fn):
            return await embed_fn(texts)
        return embed_fn(texts)
    resp = await get_async_client(api_key).embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]
