"""Local embedding provider using sentence-transformers.

Requires: pip install distillcore[local]
"""

from __future__ import annotations

from typing import Callable

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment,misc]


def local_embedder(
    model: str = "all-MiniLM-L6-v2",
    device: str | None = None,
) -> Callable[[list[str]], list[list[float]]]:
    """Create a local embedding function using sentence-transformers.

    The model is loaded once when this factory is called. Subsequent embed
    calls reuse the loaded model.

    Args:
        model: HuggingFace model name. Default "all-MiniLM-L6-v2" (384d, fast).
               Other options:
               - "all-mpnet-base-v2" (768d, best quality)
               - "BAAI/bge-small-en-v1.5" (384d, strong alternative)
        device: Device to run on. None auto-detects (GPU if available, else CPU).
                Options: "cpu", "cuda", "mps".

    Returns:
        A callable that takes a list of texts and returns a list of embedding vectors.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install with: pip install distillcore[local]"
        )

    st_model = SentenceTransformer(model, device=device)

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = st_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return embed
