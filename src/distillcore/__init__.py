"""distillcore — universal document processing: extract, chunk, enrich, embed, validate."""

__version__ = "0.7.0"

# Chunking (standalone)
from .chunking import achunk, chunk, estimate_tokens

# Config
from .config import ChunkConfig, DistillConfig, DomainConfig, EmbeddingConfig

# Embedding providers
from .embedding import ollama_embedder, openai_embedder

# Extractors
from .extractors import extract, register_extractor

# LLM utilities
from .llm.json_repair import safe_parse, try_fix_truncated_json

# Models
from .models import (
    BatchResult,
    ChunkedDocument,
    Document,
    DocumentChunk,
    DocumentMetadata,
    ExtractionResult,
    PageText,
    ProcessingResult,
    Section,
    TranscriptTurn,
    ValidationReport,
)

# Pipeline entry points (async + batch)
from .pipeline.async_orchestrator import (
    process_batch,
    process_batch_sync,
    process_document_async,
    process_text_async,
)

# Pipeline entry points (sync)
from .pipeline.orchestrator import process_document, process_text

# Presets
from .presets import load_preset

# Storage
from .storage import Store

# Validation
from .validation.coverage import compute_coverage, find_missing_segments

__all__ = [
    # Chunking (standalone)
    "chunk",
    "achunk",
    "estimate_tokens",
    # Pipeline (sync)
    "process_document",
    "process_text",
    # Pipeline (async + batch)
    "process_document_async",
    "process_text_async",
    "process_batch",
    "process_batch_sync",
    # Config
    "DistillConfig",
    "ChunkConfig",
    "EmbeddingConfig",
    "DomainConfig",
    # Models
    "PageText",
    "ExtractionResult",
    "Section",
    "TranscriptTurn",
    "Document",
    "DocumentMetadata",
    "DocumentChunk",
    "ChunkedDocument",
    "ValidationReport",
    "ProcessingResult",
    "BatchResult",
    # Extractors
    "extract",
    "register_extractor",
    # Validation
    "compute_coverage",
    "find_missing_segments",
    # LLM
    "safe_parse",
    "try_fix_truncated_json",
    # Presets
    "load_preset",
    # Embedding providers
    "openai_embedder",
    "ollama_embedder",
    # Storage
    "Store",
]
