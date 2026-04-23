"""Embedding provider factories for distillcore.

Built-in providers:
- openai_embedder: OpenAI API (always available)
- ollama_embedder: Local Ollama server (always available, no pip deps)
- local_embedder: sentence-transformers (requires distillcore[local])
- cohere_embedder: Cohere API (requires distillcore[cohere])
"""

from .ollama import ollama_embedder
from .openai import openai_embedder

__all__ = ["openai_embedder", "ollama_embedder"]

try:
    from .local import local_embedder  # noqa: F401

    __all__.append("local_embedder")
except ImportError:
    pass

try:
    from .cohere import cohere_embedder  # noqa: F401

    __all__.append("cohere_embedder")
except ImportError:
    pass
