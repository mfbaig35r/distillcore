"""distillcore — universal document processing: extract, chunk, enrich, embed, validate."""

__version__ = "0.2.0"

# Config
from .config import ChunkConfig, DistillConfig, DomainConfig, EmbeddingConfig

# Extractors
from .extractors import extract, register_extractor

# LLM utilities
from .llm.json_repair import safe_parse, try_fix_truncated_json

# Models
from .models import (
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

# Pipeline entry points
from .pipeline.orchestrator import process_document, process_text

# Presets
from .presets import load_preset

# Storage
from .storage import Store

# Validation
from .validation.coverage import compute_coverage, find_missing_segments

__all__ = [
    # Pipeline
    "process_document",
    "process_text",
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
    # Storage
    "Store",
]
