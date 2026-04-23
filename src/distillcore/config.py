"""Configuration dataclasses for the distillcore pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class EmbedFn(Protocol):
    """Protocol for custom embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class ChunkConfig:
    """Chunking parameters."""

    target_tokens: int = 500
    overlap_chars: int = 200
    max_tokens: int = 1000


@dataclass
class EmbeddingConfig:
    """Embedding parameters."""

    model: str = "text-embedding-3-small"
    embed_fn: EmbedFn | None = None


@dataclass
class DomainConfig:
    """Pluggable prompts for LLM stages. See presets/ for examples."""

    name: str = "generic"
    classification_prompt: str = ""
    structuring_prompt: str = ""
    transcript_prompt: str = ""
    enrichment_prompt: str = ""
    parse_classification: Callable[[dict[str, Any], str, int], Any] | None = None


@dataclass
class DistillConfig:
    """Top-level pipeline configuration."""

    # LLM
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    max_tokens: int = 16384

    # Pipeline stages
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)

    # Feature flags
    enrich_chunks: bool = True
    enable_ocr: bool = True

    # Large document handling
    large_doc_char_threshold: int = 80_000
    llm_page_window_size: int = 15
    llm_page_window_overlap: int = 2

    # Validation thresholds
    structuring_coverage_threshold: float = 0.95
    chunking_coverage_threshold: float = 0.98
    end_to_end_coverage_threshold: float = 0.93

    # Security
    allowed_dirs: list[str] | None = None  # None = unrestricted; list restricts file access

    # Storage
    store_path: str = "~/.distillcore/store.db"

    # Progress callback
    on_progress: Callable[[str, dict[str, Any]], None] | None = None

    def resolve_api_key(self) -> str:
        """Return the API key, falling back to OPENAI_API_KEY env var."""
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    def validate(self) -> list[str]:
        """Check config for potential issues. Returns list of warnings."""
        warnings: list[str] = []
        if not self.resolve_api_key() and self.embedding.embed_fn is None:
            warnings.append(
                "No OpenAI API key configured and no custom embed_fn set. "
                "LLM features (classification, structuring, enrichment, embedding) "
                "will fail."
            )
        return warnings
