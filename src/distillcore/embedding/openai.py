"""OpenAI embedding provider."""

from __future__ import annotations

from typing import Callable

from ..llm.client import get_client


def openai_embedder(
    model: str = "text-embedding-3-small",
    api_key: str = "",
) -> Callable[[list[str]], list[list[float]]]:
    """Create an OpenAI embedding function.

    Args:
        model: OpenAI embedding model name. Default "text-embedding-3-small".
               Other options: "text-embedding-3-large" (3072d, higher quality).
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        A callable that takes a list of texts and returns a list of embedding vectors.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        resp = get_client(api_key).embeddings.create(input=texts, model=model)
        return [item.embedding for item in resp.data]

    return embed
