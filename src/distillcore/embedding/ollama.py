"""Ollama embedding provider — local, no pip deps required.

Requires Ollama running locally: https://ollama.com
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable


def ollama_embedder(
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> Callable[[list[str]], list[list[float]]]:
    """Create an Ollama embedding function.

    Calls the local Ollama REST API. No pip dependencies — uses stdlib urllib.
    Ollama must be running at base_url.

    Args:
        model: Ollama model name. Default "nomic-embed-text" (768d).
               Other options:
               - "mxbai-embed-large" (1024d, higher quality)
               - "all-minilm" (384d, fast)
        base_url: Ollama server URL. Default "http://localhost:11434".

    Returns:
        A callable that takes a list of texts and returns a list of embedding vectors.
    """

    def embed(texts: list[str]) -> list[list[float]]:
        req = urllib.request.Request(
            f"{base_url}/api/embed",
            data=json.dumps({"model": model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
        return data["embeddings"]

    return embed
