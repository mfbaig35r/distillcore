"""Tests for distillcore async pipeline."""

from pathlib import Path

import pytest

from distillcore.config import DistillConfig, EmbeddingConfig
from distillcore.models import ProcessingResult
from distillcore.pipeline.async_orchestrator import process_document_async, process_text_async


@pytest.mark.asyncio
async def test_process_text_async_no_llm() -> None:
    """Minimal async pipeline: no LLM, no embed."""
    config = DistillConfig(enrich_chunks=False)
    result = await process_text_async(
        "Hello world.\n\nSecond paragraph.", config=config, embed=False
    )
    assert isinstance(result, ProcessingResult)
    assert result.document.full_text == "Hello world.\n\nSecond paragraph."
    assert len(result.chunks) >= 1


@pytest.mark.asyncio
async def test_process_text_async_with_custom_embed() -> None:
    """Async pipeline with custom sync embedding function."""

    def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]

    config = DistillConfig(
        enrich_chunks=False,
        embedding=EmbeddingConfig(embed_fn=fake_embed),
    )
    result = await process_text_async("Some text.", config=config, embed=True)
    assert result.chunks[0].embedding == [1.0, 2.0, 3.0]


@pytest.mark.asyncio
async def test_process_text_async_with_async_embed() -> None:
    """Async pipeline with async embedding function."""

    async def async_embed(texts: list[str]) -> list[list[float]]:
        return [[4.0, 5.0, 6.0] for _ in texts]

    config = DistillConfig(
        enrich_chunks=False,
        embedding=EmbeddingConfig(embed_fn=async_embed),
    )
    result = await process_text_async("Some text.", config=config, embed=True)
    assert result.chunks[0].embedding == [4.0, 5.0, 6.0]


@pytest.mark.asyncio
async def test_process_text_async_progress_callback() -> None:
    """Verify progress callback fires in async mode."""
    stages: list[str] = []

    def on_progress(stage: str, data: dict) -> None:
        stages.append(stage)

    config = DistillConfig(enrich_chunks=False, on_progress=on_progress)
    await process_text_async("Hello.", config=config, embed=False)

    assert "classification" in stages
    assert "chunking" in stages
    assert "complete" in stages


@pytest.mark.asyncio
async def test_process_document_async(tmp_path: Path) -> None:
    """Process a .txt file through async pipeline."""
    f = tmp_path / "test.txt"
    f.write_text("First paragraph.\n\nSecond paragraph.")

    config = DistillConfig(enrich_chunks=False)
    result = await process_document_async(f, config=config, embed=False)

    assert isinstance(result, ProcessingResult)
    assert result.document.metadata.source_filename == "test.txt"
    assert len(result.chunks) >= 1
