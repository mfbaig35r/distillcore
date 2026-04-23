"""distillcore FastMCP server — optional, install with pip install distillcore[mcp]."""

from __future__ import annotations

from fastmcp import FastMCP

from .config import ChunkConfig, DistillConfig
from .models import Document, DocumentMetadata
from .pipeline.chunking import chunk_document
from .pipeline.orchestrator import process_document, process_text
from .presets import load_preset
from .validation.coverage import compute_coverage, find_missing_segments

mcp = FastMCP(
    "distillcore",
    instructions=(
        "Universal document processing: extract, chunk, enrich, embed, validate. "
        "Use distill_file to process a document file through the full pipeline. "
        "Use distill_text to process raw text (skips extraction). "
        "Use distill_chunks_only to chunk text without LLM calls. "
        "Use distill_validate to check coverage between original text and chunks."
    ),
)


# -- Implementations ----------------------------------------------------------


def _impl_distill_file(
    file_path: str,
    format: str | None = None,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
) -> dict:
    domain_config = load_preset(domain)
    config = DistillConfig(
        domain=domain_config,
        chunk=ChunkConfig(target_tokens=chunk_target_tokens),
        enrich_chunks=enrich,
    )
    result = process_document(file_path, config=config, format=format, embed=embed)
    return result.model_dump()


def _impl_distill_text(
    text: str,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
) -> dict:
    domain_config = load_preset(domain)
    config = DistillConfig(
        domain=domain_config,
        chunk=ChunkConfig(target_tokens=chunk_target_tokens),
        enrich_chunks=enrich,
    )
    result = process_text(text, config=config, embed=embed)
    return result.model_dump()


def _impl_distill_chunks_only(
    text: str,
    chunk_target_tokens: int = 500,
    overlap_chars: int = 200,
) -> dict:
    config = ChunkConfig(target_tokens=chunk_target_tokens, overlap_chars=overlap_chars)
    doc = Document(
        metadata=DocumentMetadata(source_filename="input.txt"),
        full_text=text,
    )
    chunked = chunk_document(doc, config=config)
    return chunked.model_dump()


def _impl_distill_validate(
    original_text: str,
    chunk_texts: list[str],
) -> dict:
    derived = "\n".join(chunk_texts)
    coverage = compute_coverage(original_text, derived)
    missing = find_missing_segments(original_text, derived)
    return {
        "coverage": coverage,
        "missing_segments": missing,
        "chunk_count": len(chunk_texts),
    }


# -- Tools --------------------------------------------------------------------


@mcp.tool()
def distill_file(
    file_path: str,
    format: str | None = None,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
) -> dict:
    """Process a document file through the full distillcore pipeline.

    Runs extraction, classification, structuring, chunking, enrichment,
    embedding, and validation.

    Args:
        file_path: Path to the document file.
        format: File format override (e.g., "pdf", "txt"). Auto-detected if omitted.
        domain: Domain preset name ("generic" or "legal"). Default "generic".
        embed: Whether to generate embeddings. Default True.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        enrich: Whether to run LLM enrichment on chunks. Default True.
    """
    return _impl_distill_file(file_path, format, domain, embed, chunk_target_tokens, enrich)


@mcp.tool()
def distill_text(
    text: str,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
) -> dict:
    """Process raw text through the distillcore pipeline (skips extraction).

    Useful when you already have the text content and don't need file extraction.

    Args:
        text: The text content to process.
        domain: Domain preset name ("generic" or "legal"). Default "generic".
        embed: Whether to generate embeddings. Default True.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        enrich: Whether to run LLM enrichment on chunks. Default True.
    """
    return _impl_distill_text(text, domain, embed, chunk_target_tokens, enrich)


@mcp.tool()
def distill_chunks_only(
    text: str,
    chunk_target_tokens: int = 500,
    overlap_chars: int = 200,
) -> dict:
    """Chunk text without any LLM calls (no classification, structuring, or enrichment).

    Fast, deterministic chunking using paragraph boundary detection.

    Args:
        text: The text to chunk.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        overlap_chars: Character overlap between chunks. Default 200.
    """
    return _impl_distill_chunks_only(text, chunk_target_tokens, overlap_chars)


@mcp.tool()
def distill_validate(
    original_text: str,
    chunk_texts: list[str],
) -> dict:
    """Validate coverage between original text and a set of chunks.

    Checks what fraction of the original text is preserved in the chunks
    and identifies any missing segments.

    Args:
        original_text: The original source text.
        chunk_texts: List of chunk text strings to validate against.
    """
    return _impl_distill_validate(original_text, chunk_texts)


# -- Entry point --------------------------------------------------------------


def main() -> None:
    mcp.run()
