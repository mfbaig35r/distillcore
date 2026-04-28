"""Async pipeline orchestrator — async entry points and batch processing."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from ..config import DistillConfig
from ..extractors import extract
from ..llm.async_client import embed_texts_async
from ..models import (
    Document,
    DocumentMetadata,
    ProcessingResult,
    ValidationReport,
)
from ..validation.checks import validate_chunking, validate_end_to_end, validate_structuring
from ._shared import build_combined_validation, make_emitter
from .async_classification import classify_document_async
from .async_enrichment import enrich_chunks_async
from .async_structuring import structure_document_async
from .chunking import chunk_document
from .structuring import parse_structure_result

logger = logging.getLogger(__name__)


async def process_document_async(
    source: str | Path,
    *,
    config: DistillConfig | None = None,
    format: str | None = None,
    embed: bool = True,
) -> ProcessingResult:
    """Async full pipeline: extract -> classify -> structure -> chunk -> enrich -> embed.

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

    emit = make_emitter(config)

    # --- Extract (offloaded to thread — may do blocking file I/O) ---
    emit("extraction", {"source": str(source)})
    extraction = await asyncio.to_thread(extract, source, format=format, config=config)
    emit("extraction_done", {"pages": extraction.page_count, "format": extraction.format})

    filename = Path(source).name
    pages_text = [p.text for p in extraction.pages]

    return await _run_pipeline_async(
        full_text=extraction.full_text,
        pages_text=pages_text,
        filename=filename,
        page_count=extraction.page_count,
        config=config,
        embed=embed,
        emit=emit,
    )


async def process_text_async(
    text: str,
    *,
    config: DistillConfig | None = None,
    filename: str = "input.txt",
    embed: bool = True,
) -> ProcessingResult:
    """Async pipeline for raw text. Skips extraction."""
    if config is None:
        config = DistillConfig()

    emit = make_emitter(config)

    return await _run_pipeline_async(
        full_text=text,
        pages_text=[text],
        filename=filename,
        page_count=1,
        config=config,
        embed=embed,
        emit=emit,
    )


async def process_batch(
    sources: list[str | Path],
    *,
    config: DistillConfig | None = None,
    format: str | None = None,
    embed: bool = True,
    max_concurrent: int = 5,
    on_result: Callable[[str, ProcessingResult], None] | None = None,
) -> list[ProcessingResult]:
    """Process multiple documents concurrently.

    Args:
        sources: List of file paths to process.
        config: Shared config for all documents.
        format: File format override. Auto-detected if None.
        embed: Whether to generate embeddings (default True).
        max_concurrent: Max concurrent pipelines (default 5).
        on_result: Optional callback called with (source, result) as each completes.

    Returns:
        List of ProcessingResults in same order as sources.
    """
    if config is None:
        config = DistillConfig()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_one(source: str | Path) -> ProcessingResult:
        async with semaphore:
            return await process_document_async(
                source, config=config, format=format, embed=embed
            )

    tasks = [_process_one(s) for s in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    final: list[ProcessingResult] = []
    for source, result in zip(sources, results):
        if isinstance(result, BaseException):
            logger.error(f"Batch processing failed for {source}: {result}")
            final.append(
                ProcessingResult(
                    document=Document(
                        metadata=DocumentMetadata(source_filename=str(source))
                    ),
                    chunks=[],
                    validation=ValidationReport(
                        warnings=[f"Processing failed: {result}"],
                        passed=False,
                    ),
                )
            )
        else:
            final.append(result)
            if on_result:
                on_result(str(source), result)

    return final


def process_batch_sync(
    sources: list[str | Path],
    **kwargs: Any,
) -> list[ProcessingResult]:
    """Sync convenience wrapper for process_batch.

    Uses asyncio.run() — don't call from within an existing event loop.
    """
    return asyncio.run(process_batch(sources, **kwargs))


async def _run_pipeline_async(
    full_text: str,
    pages_text: list[str],
    filename: str,
    page_count: int,
    config: DistillConfig,
    embed: bool,
    emit: Any,
) -> ProcessingResult:
    """Shared async pipeline logic."""

    # --- Validate config ---
    for warning in config.validate():
        logger.warning(warning)

    # --- Classify (async) ---
    emit("classification", {"filename": filename})
    metadata = await classify_document_async(filename, pages_text, page_count, config)
    emit("classification_done", {"document_type": metadata.document_type})

    # --- Structure (async — parallel windows for large docs) ---
    emit("structuring", {"filename": filename})
    is_transcript = bool(metadata.extra.get("is_transcript"))
    structure_result = await structure_document_async(
        full_text=full_text,
        document_type=metadata.document_type,
        filename=filename,
        config=config,
        pages_text=pages_text,
        is_transcript=is_transcript,
    )
    sections, transcript_turns, structuring_error = parse_structure_result(
        structure_result, pages_text=pages_text
    )

    doc = Document(
        metadata=metadata,
        sections=sections,
        transcript_turns=transcript_turns,
        full_text=full_text,
    )
    emit("structuring_done", {"sections": len(sections), "turns": len(transcript_turns)})

    # --- Validate structuring (sync — CPU) ---
    struct_report = validate_structuring(doc, threshold=config.structuring_coverage_threshold)
    if structuring_error:
        struct_report.warnings.append(f"Structuring failed: {structuring_error}")
    for w in struct_report.warnings:
        logger.warning(w)

    # --- Chunk (sync — CPU) ---
    emit("chunking", {"filename": filename})
    chunked = chunk_document(doc, config=config.chunk)
    emit("chunking_done", {"chunks": chunked.chunk_count})

    # --- Validate chunking (sync — CPU) ---
    chunk_report = validate_chunking(doc, chunked, threshold=config.chunking_coverage_threshold)
    for w in chunk_report.warnings:
        logger.warning(w)

    e2e_report = validate_end_to_end(
        full_text, chunked, threshold=config.end_to_end_coverage_threshold
    )
    for w in e2e_report.warnings:
        logger.warning(w)

    # --- Enrich (async) ---
    chunks = chunked.chunks
    if config.enrich_chunks:
        emit("enrichment", {"chunks": len(chunks)})
        chunks = await enrich_chunks_async(chunks, metadata.document_type, config)
        emit("enrichment_done", {"enriched": sum(1 for c in chunks if c.topic)})

    # --- Embed (async) ---
    if embed:
        emit("embedding", {"chunks": len(chunks)})
        chunk_texts = [c.text for c in chunks]
        embeddings = await embed_texts_async(
            chunk_texts,
            model=config.embedding.model,
            api_key=config.resolve_api_key(),
            embed_fn=config.embedding.embed_fn,
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        emit("embedding_done", {"embedded": len(chunks)})

    # --- Combined validation ---
    combined = build_combined_validation(struct_report, chunk_report, e2e_report)

    result = ProcessingResult(document=doc, chunks=chunks, validation=combined)
    emit("complete", {"chunks": len(chunks), "passed": combined.passed})
    return result

