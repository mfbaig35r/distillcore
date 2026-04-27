"""Pipeline orchestrator — the main entry points for distillcore."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import DistillConfig
from ..extractors import extract
from ..llm.client import embed_texts
from ..models import (
    Document,
    ProcessingResult,
    ValidationReport,
)
from ..validation.checks import validate_chunking, validate_end_to_end, validate_structuring
from .chunking import chunk_document
from .classification import classify_document
from .enrichment import enrich_chunks
from .structuring import parse_structure_result, structure_document

logger = logging.getLogger(__name__)


def process_document(
    source: str | Path,
    *,
    config: DistillConfig | None = None,
    format: str | None = None,
    embed: bool = True,
) -> ProcessingResult:
    """Full pipeline: extract -> classify -> structure -> chunk -> enrich -> embed -> validate.

    Args:
        source: Path to the document file.
        config: Processing configuration. Uses defaults if None.
        format: File format override. Auto-detected from extension if None.
        embed: Whether to generate embeddings (default True).

    Returns:
        ProcessingResult with document, chunks, and validation report.
    """
    if config is None:
        config = DistillConfig()

    emit = _make_emitter(config)

    # --- Extract ---
    emit("extraction", {"source": str(source)})
    extraction = extract(source, format=format, config=config)
    emit("extraction_done", {"pages": extraction.page_count, "format": extraction.format})

    filename = Path(source).name
    pages_text = [p.text for p in extraction.pages]

    return _run_pipeline(
        full_text=extraction.full_text,
        pages_text=pages_text,
        filename=filename,
        page_count=extraction.page_count,
        config=config,
        embed=embed,
        emit=emit,
    )


def process_text(
    text: str,
    *,
    config: DistillConfig | None = None,
    filename: str = "input.txt",
    embed: bool = True,
) -> ProcessingResult:
    """Pipeline for raw text: classify -> structure -> chunk -> enrich -> embed -> validate.

    Skips extraction. Useful when you already have the text.
    """
    if config is None:
        config = DistillConfig()

    emit = _make_emitter(config)

    return _run_pipeline(
        full_text=text,
        pages_text=[text],
        filename=filename,
        page_count=1,
        config=config,
        embed=embed,
        emit=emit,
    )


def _run_pipeline(
    full_text: str,
    pages_text: list[str],
    filename: str,
    page_count: int,
    config: DistillConfig,
    embed: bool,
    emit: Any,
) -> ProcessingResult:
    """Shared pipeline logic for both entry points."""

    # --- Validate config ---
    for warning in config.validate():
        logger.warning(warning)

    # --- Classify ---
    emit("classification", {"filename": filename})
    metadata = classify_document(filename, pages_text, page_count, config)
    emit("classification_done", {"document_type": metadata.document_type})

    # --- Structure ---
    emit("structuring", {"filename": filename})
    is_transcript = bool(metadata.extra.get("is_transcript"))
    structure_result = structure_document(
        full_text=full_text,
        document_type=metadata.document_type,
        filename=filename,
        config=config,
        pages_text=pages_text,
        is_transcript=is_transcript,
    )
    sections, transcript_turns = parse_structure_result(structure_result, pages_text=pages_text)

    doc = Document(
        metadata=metadata,
        sections=sections,
        transcript_turns=transcript_turns,
        full_text=full_text,
    )
    emit("structuring_done", {"sections": len(sections), "turns": len(transcript_turns)})

    # --- Validate structuring ---
    struct_report = validate_structuring(doc, threshold=config.structuring_coverage_threshold)
    for w in struct_report.warnings:
        logger.warning(w)

    # --- Chunk ---
    emit("chunking", {"filename": filename})
    chunked = chunk_document(doc, config=config.chunk)
    emit("chunking_done", {"chunks": chunked.chunk_count})

    # --- Validate chunking ---
    chunk_report = validate_chunking(doc, chunked, threshold=config.chunking_coverage_threshold)
    for w in chunk_report.warnings:
        logger.warning(w)

    e2e_report = validate_end_to_end(
        full_text, chunked, threshold=config.end_to_end_coverage_threshold
    )
    for w in e2e_report.warnings:
        logger.warning(w)

    # --- Enrich ---
    chunks = chunked.chunks
    if config.enrich_chunks:
        emit("enrichment", {"chunks": len(chunks)})
        chunks = enrich_chunks(chunks, metadata.document_type, config)
        emit("enrichment_done", {"enriched": sum(1 for c in chunks if c.topic)})

    # --- Embed ---
    if embed:
        emit("embedding", {"chunks": len(chunks)})
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_texts(
            chunk_texts,
            model=config.embedding.model,
            api_key=config.resolve_api_key(),
            embed_fn=config.embedding.embed_fn,
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        emit("embedding_done", {"embedded": len(chunks)})

    # --- Combined validation ---
    combined = ValidationReport(
        structuring_coverage=struct_report.structuring_coverage,
        chunking_coverage=chunk_report.chunking_coverage,
        end_to_end_coverage=e2e_report.end_to_end_coverage,
        missing_segments=struct_report.missing_segments,
        warnings=struct_report.warnings + chunk_report.warnings + e2e_report.warnings,
        passed=struct_report.passed and chunk_report.passed and e2e_report.passed,
    )

    result = ProcessingResult(document=doc, chunks=chunks, validation=combined)
    emit("complete", {"chunks": len(chunks), "passed": combined.passed})
    return result


# --- Progress emission ---

def _make_emitter(config: DistillConfig) -> Any:
    """Create a progress emitter from config."""
    callback = config.on_progress

    def emit(stage: str, data: dict | None = None) -> None:
        if callback:
            callback(stage, data or {})

    return emit
