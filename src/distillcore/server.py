"""distillcore FastMCP server — optional, install with pip install distillcore[mcp]."""

from __future__ import annotations

import os
from pathlib import Path

from fastmcp import FastMCP

from .config import ChunkConfig, DistillConfig
from .llm.client import embed_texts
from .pipeline.async_orchestrator import process_batch
from .pipeline.orchestrator import process_document, process_text
from .presets import load_preset
from .storage import Store
from .validation.coverage import compute_coverage, find_missing_segments

# ── Configuration ─────────────────────────────────────────────────────────────

STORE_PATH = (
    Path(os.environ.get("DISTILLCORE_STORE", "~/.distillcore/store.db"))
    .expanduser()
    .resolve()
)
EMBEDDING_MODEL = os.environ.get("DISTILLCORE_EMBEDDING_MODEL", "text-embedding-3-small")
_allowed_dirs_raw = os.environ.get("DISTILLCORE_ALLOWED_DIRS", "")
ALLOWED_DIRS: list[str] | None = _allowed_dirs_raw.split(":") if _allowed_dirs_raw else None
TENANT_ID: str | None = os.environ.get("DISTILLCORE_TENANT_ID") or None

# ── Singletons ────────────────────────────────────────────────────────────────

store = Store(STORE_PATH)

# ── Server ────────────────────────────────────────────────────────────────────

mcp = FastMCP(
    "distillcore",
    instructions=(
        "Universal document processing: extract, chunk, enrich, embed, validate. "
        "Use distill_file to process a document file through the full pipeline. "
        "Use distill_text to process raw text (skips extraction). "
        "Use distill_chunks_only to chunk text without LLM calls. "
        "Use distill_validate to check coverage between original text and chunks. "
        "Use store=True on distill_file/distill_text to persist results. "
        "Use distill_search to semantic search across stored documents. "
        "Use distill_list_documents to browse stored documents. "
        "Use distill_get_document to retrieve a document and its chunks."
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
    persist: bool = False,
) -> dict:
    domain_config = load_preset(domain)
    config = DistillConfig(
        domain=domain_config,
        chunk=ChunkConfig(target_tokens=chunk_target_tokens),
        enrich_chunks=enrich,
        allowed_dirs=ALLOWED_DIRS,
    )
    result = process_document(file_path, config=config, format=format, embed=embed)
    response = result.model_dump(exclude={"chunks": {"__all__": {"embedding"}}})
    if persist:
        doc_id = store.save(result, tenant_id=TENANT_ID)
        response["stored"] = True
        response["document_id"] = doc_id
    return response


def _impl_distill_text(
    text: str,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
    persist: bool = False,
) -> dict:
    domain_config = load_preset(domain)
    config = DistillConfig(
        domain=domain_config,
        chunk=ChunkConfig(target_tokens=chunk_target_tokens),
        enrich_chunks=enrich,
    )
    result = process_text(text, config=config, embed=embed)
    response = result.model_dump(exclude={"chunks": {"__all__": {"embedding"}}})
    if persist:
        doc_id = store.save(result, tenant_id=TENANT_ID)
        response["stored"] = True
        response["document_id"] = doc_id
    return response


def _impl_distill_chunks_only(
    text: str,
    chunk_target_tokens: int = 500,
    overlap_tokens: int = 50,
    min_tokens: int = 0,
    strategy: str = "paragraph",
) -> dict:
    from .chunking import chunk, estimate_tokens

    chunks = chunk(
        text,
        strategy=strategy,
        target_tokens=chunk_target_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
    )
    return {
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_index": i,
                "text": c,
                "token_estimate": estimate_tokens(c),
            }
            for i, c in enumerate(chunks)
        ],
    }


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


def _impl_distill_search(
    query: str,
    top_k: int = 10,
    document_type: str | None = None,
) -> dict:
    query_embedding = embed_texts(
        [query],
        model=EMBEDDING_MODEL,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )[0]
    results = store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        document_type=document_type,
        tenant_id=TENANT_ID,
    )
    store.log_search(
        query=query,
        result_count=len(results),
        top_chunk_ids=[r["id"] for r in results[:5]],
    )
    return {"query": query, "result_count": len(results), "results": results}


def _impl_distill_list_documents(
    document_type: str | None = None,
    limit: int = 50,
) -> dict:
    docs = store.list_documents(document_type=document_type, limit=limit, tenant_id=TENANT_ID)
    return {"count": len(docs), "documents": docs}


def _impl_distill_get_document(document_id: str) -> dict:
    doc = store.get_document(document_id, tenant_id=TENANT_ID)
    if not doc:
        return {"error": f"Document not found: {document_id}"}
    chunks = store.get_chunks(document_id, tenant_id=TENANT_ID)
    doc["chunks"] = chunks
    doc["chunk_count"] = len(chunks)
    return doc


async def _impl_distill_batch(
    file_paths: list[str],
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
    persist: bool = False,
    max_concurrent: int = 5,
) -> dict:
    domain_config = load_preset(domain)
    config = DistillConfig(
        domain=domain_config,
        chunk=ChunkConfig(target_tokens=chunk_target_tokens),
        enrich_chunks=enrich,
        allowed_dirs=ALLOWED_DIRS,
    )

    results = await process_batch(
        file_paths, config=config, embed=embed, max_concurrent=max_concurrent
    )

    succeeded = sum(1 for r in results if r.validation.passed)
    failed = len(results) - succeeded

    response_results = []
    for path, result in zip(file_paths, results):
        entry = result.model_dump(exclude={"chunks": {"__all__": {"embedding"}}})
        entry["source"] = path
        if persist and result.validation.passed:
            doc_id = store.save(result, tenant_id=TENANT_ID)
            entry["stored"] = True
            entry["document_id"] = doc_id
        response_results.append(entry)

    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": response_results,
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
    store: bool = False,
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
        store: Whether to persist the result in the local store. Default False.
    """
    return _impl_distill_file(
        file_path, format, domain, embed, chunk_target_tokens, enrich, persist=store
    )


@mcp.tool()
def distill_text(
    text: str,
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
    store: bool = False,
) -> dict:
    """Process raw text through the distillcore pipeline (skips extraction).

    Useful when you already have the text content and don't need file extraction.

    Args:
        text: The text content to process.
        domain: Domain preset name ("generic" or "legal"). Default "generic".
        embed: Whether to generate embeddings. Default True.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        enrich: Whether to run LLM enrichment on chunks. Default True.
        store: Whether to persist the result in the local store. Default False.
    """
    return _impl_distill_text(
        text, domain, embed, chunk_target_tokens, enrich, persist=store
    )


@mcp.tool()
def distill_chunks_only(
    text: str,
    chunk_target_tokens: int = 500,
    overlap_tokens: int = 50,
    min_tokens: int = 0,
    strategy: str = "paragraph",
) -> dict:
    """Chunk text without any LLM calls.

    Uses distillcore's standalone chunking API with configurable strategies.

    Args:
        text: The text to chunk.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        overlap_tokens: Token overlap between chunks. Default 50.
        min_tokens: Minimum tokens per chunk; smaller chunks merge
            into neighbors. Default 0 (disabled).
        strategy: "paragraph" (default), "sentence", or "fixed".
            Use "llm" for LLM-driven semantic chunking (requires API key).
    """
    return _impl_distill_chunks_only(
        text, chunk_target_tokens, overlap_tokens, min_tokens, strategy
    )


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


@mcp.tool()
def distill_search(
    query: str,
    top_k: int = 10,
    document_type: str | None = None,
) -> dict:
    """Semantic search across stored documents using cosine similarity.

    Embeds the query and searches over all stored chunk embeddings.
    Requires documents to have been stored with store=True and embed=True.

    Args:
        query: Natural language search query.
        top_k: Number of results to return. Default 10.
        document_type: Optional filter by document type.
    """
    return _impl_distill_search(query, top_k, document_type)


@mcp.tool()
def distill_list_documents(
    document_type: str | None = None,
    limit: int = 50,
) -> dict:
    """List all documents in the local store.

    Args:
        document_type: Optional filter by document type.
        limit: Maximum number of documents to return. Default 50.
    """
    return _impl_distill_list_documents(document_type, limit)


@mcp.tool()
def distill_get_document(document_id: str) -> dict:
    """Get full details and chunks for a stored document.

    Args:
        document_id: The document UUID from distill_list_documents or the
                     document_id returned when store=True.
    """
    return _impl_distill_get_document(document_id)


@mcp.tool()
async def distill_batch(
    file_paths: list[str],
    domain: str = "generic",
    embed: bool = True,
    chunk_target_tokens: int = 500,
    enrich: bool = True,
    store: bool = False,
    max_concurrent: int = 5,
) -> dict:
    """Process multiple files concurrently through the pipeline.

    Each file runs through the full distillcore pipeline (extraction,
    classification, structuring, chunking, enrichment, embedding).
    Files are processed concurrently up to max_concurrent.

    Failed files don't crash the batch — they get a result with passed=False.

    Args:
        file_paths: List of file paths to process.
        domain: Domain preset name ("generic" or "legal"). Default "generic".
        embed: Whether to generate embeddings. Default True.
        chunk_target_tokens: Target chunk size in tokens. Default 500.
        enrich: Whether to run LLM enrichment on chunks. Default True.
        store: Whether to persist results in the local store. Default False.
        max_concurrent: Max concurrent pipelines. Default 5.
    """
    return await _impl_distill_batch(
        file_paths, domain, embed, chunk_target_tokens, enrich,
        persist=store, max_concurrent=max_concurrent,
    )


# -- Entry point --------------------------------------------------------------


def main() -> None:
    mcp.run()
