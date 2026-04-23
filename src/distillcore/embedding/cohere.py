"""Cohere embedding provider.

Requires: pip install distillcore[cohere]
"""

from __future__ import annotations

import os
from typing import Callable

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore[assignment]


def cohere_embedder(
    model: str = "embed-english-v3.0",
    api_key: str = "",
    input_type: str = "search_document",
) -> Callable[[list[str]], list[list[float]]]:
    """Create a Cohere embedding function.

    Args:
        model: Cohere embedding model. Default "embed-english-v3.0".
               Other options:
               - "embed-multilingual-v3.0" (multilingual)
               - "embed-english-light-v3.0" (faster, smaller)
        api_key: Cohere API key. Falls back to COHERE_API_KEY env var.
        input_type: Cohere input type. Default "search_document" (for indexing).
                    Use "search_query" when embedding search queries.

    Returns:
        A callable that takes a list of texts and returns a list of embedding vectors.
    """
    if cohere is None:
        raise ImportError(
            "cohere is required for Cohere embeddings. "
            "Install with: pip install distillcore[cohere]"
        )

    key = api_key or os.environ.get("COHERE_API_KEY", "")
    client = cohere.ClientV2(api_key=key)

    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embed(
            texts=texts,
            model=model,
            input_type=input_type,
            embedding_types=["float"],
        )
        return list(resp.embeddings.float_)

    return embed
