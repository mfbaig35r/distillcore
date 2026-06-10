"""B1: chunking throughput. distillcore vs LangChain text splitters."""

from __future__ import annotations

import time
from typing import Any

from distillcore import chunk

from ._fixtures import make_document

DOC_SIZES = [100_000, 500_000, 1_000_000]
TARGET_TOKENS = [300, 500, 1000]
STRATEGIES = ["paragraph", "sentence", "fixed"]
RUNS = 5


def _time_call(fn: Any) -> tuple[float, Any]:
    """Time a callable: 1 warmup + RUNS measured runs, return mean elapsed and last result."""
    result = fn()  # warmup
    start = time.perf_counter()
    for _ in range(RUNS):
        result = fn()
    elapsed = (time.perf_counter() - start) / RUNS
    return elapsed, result


def _bench_distillcore(text: str, strategy: str, target: int) -> dict[str, Any]:
    elapsed, chunks = _time_call(
        lambda: chunk(text, strategy=strategy, target_tokens=target)
    )
    mean_chunk = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
    return {
        "elapsed_s": round(elapsed, 6),
        "chunks": len(chunks),
        "mean_chunk_chars": round(mean_chunk, 1),
        "chars_per_sec": round(len(text) / elapsed, 0) if elapsed > 0 else None,
    }


def _bench_langchain(text: str, strategy: str, target: int) -> dict[str, Any] | None:
    """Compare against the closest LangChain splitter for the given distillcore strategy."""
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # distillcore tokens are ~4 chars; convert to char budget for LangChain.
    chunk_size = target * 4
    overlap = 50 * 4

    if strategy == "paragraph":
        # Closest match: recursive splitter with default paragraph/sentence separators.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    elif strategy == "fixed":
        # Sliding window at character boundaries.
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    elif strategy == "sentence":
        # No direct equivalent in langchain-text-splitters; recursive comes closest.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[". ", "! ", "? ", "\n\n", "\n", " "],
        )
    else:
        return None

    elapsed, chunks = _time_call(lambda: splitter.split_text(text))
    mean_chunk = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
    return {
        "elapsed_s": round(elapsed, 6),
        "chunks": len(chunks),
        "mean_chunk_chars": round(mean_chunk, 1),
        "chars_per_sec": round(len(text) / elapsed, 0) if elapsed > 0 else None,
    }


def run() -> dict[str, Any]:
    """Run the chunking benchmark suite. Returns a results dict.

    Note:
        The ``fixed`` strategy is driven by ``max_tokens``, not ``target_tokens``,
        so we run it only once per doc size (target_tokens=500 nominal).
    """
    results: list[dict[str, Any]] = []
    for size in DOC_SIZES:
        text = make_document(size)
        for strategy in STRATEGIES:
            targets = [500] if strategy == "fixed" else TARGET_TOKENS
            for target in targets:
                dc = _bench_distillcore(text, strategy, target)
                lc = _bench_langchain(text, strategy, target)
                results.append(
                    {
                        "doc_chars": size,
                        "doc_chars_actual": len(text),
                        "strategy": strategy,
                        "target_tokens": target,
                        "distillcore": dc,
                        "langchain": lc,
                    }
                )
    return {"benchmark": "B1_chunking_throughput", "results": results}
