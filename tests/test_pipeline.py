"""Tests for distillcore.pipeline.orchestrator."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from distillcore.config import DistillConfig, EmbeddingConfig
from distillcore.models import ProcessingResult
from distillcore.pipeline.orchestrator import process_document, process_text
from distillcore.presets import load_preset


def _mock_llm_response(content: dict) -> MagicMock:
    """Create a mock OpenAI chat completion response."""
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=json.dumps(content)))]
    return mock


def _mock_embed_response(n: int, dim: int = 3) -> MagicMock:
    """Create a mock OpenAI embeddings response."""
    mock = MagicMock()
    mock.data = [MagicMock(embedding=[0.1] * dim) for _ in range(n)]
    return mock


class TestProcessText:
    def test_no_llm_no_embed(self) -> None:
        """Minimal pipeline: no classification, no structuring, no enrichment, no embed."""
        config = DistillConfig(enrich_chunks=False)
        result = process_text("Hello world.\n\nSecond paragraph.", config=config, embed=False)
        assert isinstance(result, ProcessingResult)
        assert result.document.full_text == "Hello world.\n\nSecond paragraph."
        assert len(result.chunks) >= 1
        assert result.chunks[0].embedding is None

    def test_with_classification_and_structuring(self) -> None:
        """Pipeline with mocked LLM for classification and structuring."""
        config = DistillConfig(
            openai_api_key="test",
            domain=load_preset("generic"),
            enrich_chunks=False,
        )

        classify_resp = _mock_llm_response(
            {"document_type": "memo", "document_title": "Test Memo"}
        )
        structure_resp = _mock_llm_response(
            {
                "sections": [
                    {
                        "heading": "Body",
                        "section_type": "body",
                        "page_range": [1, 1],
                    }
                ]
            }
        )

        with patch("distillcore.pipeline.classification.get_client") as mock_cls, patch(
            "distillcore.pipeline.structuring.get_client"
        ) as mock_str:
            mock_cls.return_value.chat.completions.create.return_value = classify_resp
            mock_str.return_value.chat.completions.create.return_value = structure_resp

            result = process_text(
                "Hello world. Second paragraph.",
                config=config,
                embed=False,
            )

        assert result.document.metadata.document_type == "memo"
        assert len(result.document.sections) == 1

    def test_with_custom_embed_fn(self) -> None:
        """Pipeline with custom embedding function."""

        def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[1.0, 2.0, 3.0] for _ in texts]

        config = DistillConfig(
            enrich_chunks=False,
            embedding=EmbeddingConfig(embed_fn=fake_embed),
        )

        result = process_text("Some text.", config=config, embed=True)
        assert result.chunks[0].embedding == [1.0, 2.0, 3.0]

    def test_progress_callback(self) -> None:
        """Verify progress callback is invoked."""
        stages: list[str] = []

        def on_progress(stage: str, data: dict) -> None:
            stages.append(stage)

        config = DistillConfig(enrich_chunks=False, on_progress=on_progress)
        process_text("Hello.", config=config, embed=False)

        assert "classification" in stages
        assert "chunking" in stages
        assert "complete" in stages


class TestProcessDocument:
    def test_process_text_file(self, tmp_path: Path) -> None:
        """Process a .txt file through the full pipeline (no LLM)."""
        f = tmp_path / "test.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.")

        config = DistillConfig(enrich_chunks=False)
        result = process_document(f, config=config, embed=False)

        assert isinstance(result, ProcessingResult)
        assert result.document.metadata.source_filename == "test.txt"
        assert len(result.chunks) >= 1
