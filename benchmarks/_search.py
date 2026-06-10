"""B5: cosine-similarity search throughput on synthetic stores.

Builds a Store with N random embeddings (no real LLM), measures search
latency with and without the numpy cache. Numpy path uses a single matmul
against a pre-normalized matrix; Python fallback uses the per-row loop.
"""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path
from typing import Any

from distillcore.models import (
    Document,
    DocumentChunk,
    DocumentMetadata,
    ProcessingResult,
    ValidationReport,
)
from distillcore.storage import Store

SIZES = [5_000, 50_000]
DIM = 384  # mimic a small embedding model; large dims (1536) are 4x heavier
RUNS = 5
SEED = 7


def _seed_store(store: Store, n_chunks: int) -> None:
    """Populate a single document with ``n_chunks`` synthetic embedded chunks."""
    rng = random.Random(SEED)
    chunks = [
        DocumentChunk(
            chunk_index=i,
            text=f"chunk {i}",
            token_estimate=1,
            embedding=[rng.random() for _ in range(DIM)],
        )
        for i in range(n_chunks)
    ]
    result = ProcessingResult(
        document=Document(
            metadata=DocumentMetadata(
                source_filename="bench.txt",
                document_type="bench",
                page_count=1,
            ),
            full_text="",
        ),
        chunks=chunks,
        validation=ValidationReport(passed=True),
    )
    store.save(result)


def _time_search(store: Store, query: list[float], top_k: int = 10) -> float:
    store.search(query, top_k=top_k)  # warmup (builds cache)
    start = time.perf_counter()
    for _ in range(RUNS):
        store.search(query, top_k=top_k)
    return (time.perf_counter() - start) / RUNS


def run() -> dict[str, Any]:
    rng = random.Random(SEED + 1)
    results: list[dict[str, Any]] = []

    for size in SIZES:
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "bench.db")
            _seed_store(store, size)
            query = [rng.random() for _ in range(DIM)]

            # numpy path
            numpy_elapsed = _time_search(store, query)

            # Python fallback — force by stashing numpy
            original_np = store._np
            store._np = None
            store._matrix_cache = None  # ensure no stale numpy cache lingers
            try:
                python_elapsed = _time_search(store, query)
            finally:
                store._np = original_np

            results.append(
                {
                    "chunks": size,
                    "dim": DIM,
                    "numpy": {
                        "elapsed_ms": round(numpy_elapsed * 1000, 2),
                        "qps": round(1 / numpy_elapsed, 1) if numpy_elapsed > 0 else None,
                    },
                    "python_fallback": {
                        "elapsed_ms": round(python_elapsed * 1000, 2),
                        "qps": round(1 / python_elapsed, 1) if python_elapsed > 0 else None,
                    },
                    "speedup": round(python_elapsed / numpy_elapsed, 1)
                    if numpy_elapsed > 0
                    else None,
                }
            )
            store.close()

    return {"benchmark": "B5_search_throughput", "results": results}
