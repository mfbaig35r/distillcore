"""OpenAI client and embedding helpers."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from openai import OpenAI

_clients: dict[str, OpenAI] = {}
_lock = threading.Lock()


def get_client(api_key: str = "") -> OpenAI:
    """Return a cached OpenAI client for the given API key."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "LLM features require the openai package. "
            "Install with: pip install distillcore[openai]"
        ) from None

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    with _lock:
        if key not in _clients:
            _clients[key] = OpenAI(api_key=key)
        return _clients[key]


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    api_key: str = "",
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[list[float]]:
    """Batch-embed texts. Uses embed_fn if provided, otherwise OpenAI."""
    if embed_fn is not None:
        return embed_fn(texts)
    resp = get_client(api_key).embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]
