"""Tests for distillcore batch processing."""

from pathlib import Path

import pytest

from distillcore.config import DistillConfig
from distillcore.models import ProcessingResult
from distillcore.pipeline.async_orchestrator import process_batch, process_batch_sync


@pytest.mark.asyncio
async def test_batch_multiple_files(tmp_path: Path) -> None:
    """Process multiple text files concurrently."""
    files = []
    for i in range(3):
        f = tmp_path / f"doc_{i}.txt"
        f.write_text(f"Document {i} content.\n\nParagraph two of doc {i}.")
        files.append(f)

    config = DistillConfig(enrich_chunks=False)
    results = await process_batch(files, config=config, embed=False, max_concurrent=2)

    assert len(results) == 3
    for r in results:
        assert isinstance(r, ProcessingResult)
        assert r.validation.passed is not None


@pytest.mark.asyncio
async def test_batch_handles_failures(tmp_path: Path) -> None:
    """Failed files get a ProcessingResult with passed=False, not an exception."""
    good = tmp_path / "good.txt"
    good.write_text("Good content here.")
    bad = tmp_path / "nonexistent.xyz"  # doesn't exist

    config = DistillConfig(enrich_chunks=False)
    results = await process_batch(
        [good, bad], config=config, embed=False, max_concurrent=2
    )

    assert len(results) == 2
    assert results[0].document.metadata.source_filename == "good.txt"
    assert results[0].chunks  # good file has chunks
    assert results[1].validation.passed is False
    assert any("failed" in w.lower() for w in results[1].validation.warnings)


@pytest.mark.asyncio
async def test_batch_on_result_callback(tmp_path: Path) -> None:
    """on_result callback fires for each successful file."""
    f = tmp_path / "doc.txt"
    f.write_text("Content.")

    completed: list[str] = []

    def on_result(source: str, result: ProcessingResult) -> None:
        completed.append(source)

    config = DistillConfig(enrich_chunks=False)
    await process_batch([f], config=config, embed=False, on_result=on_result)

    assert len(completed) == 1


@pytest.mark.asyncio
async def test_batch_empty(tmp_path: Path) -> None:
    """Empty source list returns empty results."""
    results = await process_batch([], config=DistillConfig(), embed=False)
    assert results == []


def test_batch_sync_wrapper(tmp_path: Path) -> None:
    """process_batch_sync runs the async batch in asyncio.run()."""
    f = tmp_path / "sync_test.txt"
    f.write_text("Sync batch content.")

    config = DistillConfig(enrich_chunks=False)
    results = process_batch_sync([f], config=config, embed=False)

    assert len(results) == 1
    assert results[0].document.metadata.source_filename == "sync_test.txt"
