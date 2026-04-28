"""Tests for distillcore.server."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillcore import server as server_module
from distillcore.config import DomainConfig
from distillcore.server import (
    _impl_distill_batch,
    _impl_distill_chunks_only,
    _impl_distill_get_document,
    _impl_distill_list_documents,
    _impl_distill_text,
    _impl_distill_validate,
)
from distillcore.storage import Store


class TestDistillChunksOnly:
    def test_basic(self) -> None:
        result = _impl_distill_chunks_only("Para one.\n\nPara two.\n\nPara three.")
        assert result["chunk_count"] >= 1
        assert len(result["chunks"]) >= 1

    def test_custom_target(self) -> None:
        text = "A" * 500 + "\n\n" + "B" * 500
        result = _impl_distill_chunks_only(text, chunk_target_tokens=50)
        assert result["chunk_count"] >= 2


class TestDistillValidate:
    def test_perfect_coverage(self) -> None:
        text = "hello world foo bar"
        result = _impl_distill_validate(text, [text])
        assert result["coverage"] == 1.0
        assert result["missing_segments"] == []

    def test_partial_coverage(self) -> None:
        result = _impl_distill_validate("hello world foo bar baz qux", ["hello world"])
        assert result["coverage"] < 1.0


class TestDistillText:
    def test_with_mocked_llm(self) -> None:
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "document_type": "memo",
                            "document_title": "Test",
                            "sections": [
                                {"heading": "Body", "section_type": "body", "content": "text."}
                            ],
                        }
                    )
                )
            )
        ]

        with (
            patch("distillcore.pipeline.classification.get_client") as mc,
            patch("distillcore.pipeline.structuring.get_client") as ms,
        ):
            mc.return_value.chat.completions.create.return_value = mock_response
            ms.return_value.chat.completions.create.return_value = mock_response
            result = _impl_distill_text(
                "Some text here.",
                domain="generic",
                embed=False,
                enrich=False,
            )
        assert "document" in result
        assert "chunks" in result
        assert "validation" in result

    def test_store_persists(self, tmp_path: Path) -> None:
        """When persist=True, result is saved to the store."""
        test_store = Store(tmp_path / "test.db")
        with patch.object(server_module, "store", test_store):
            result = _impl_distill_text(
                "Some text.",
                domain="generic",
                embed=False,
                enrich=False,
                persist=True,
            )
        assert result["stored"] is True
        assert "document_id" in result
        assert test_store.get_document(result["document_id"]) is not None


class TestEmbeddingExclusion:
    def test_distill_text_excludes_embedding_arrays(self) -> None:
        """MCP responses should include has_embedding but not embedding arrays."""
        result = _impl_distill_text(
            "Some text here.",
            domain="generic",
            embed=False,
            enrich=False,
        )
        for chunk in result["chunks"]:
            assert "embedding" not in chunk
            assert chunk["has_embedding"] is False

    def test_distill_text_has_embedding_flag(self, tmp_path: Path) -> None:
        """When embeddings are generated, has_embedding is True in response."""
        from distillcore.models import (
            Document,
            DocumentChunk,
            DocumentMetadata,
            ProcessingResult,
            ValidationReport,
        )

        result_with_emb = ProcessingResult(
            document=Document(
                metadata=DocumentMetadata(source_filename="test.txt"),
                full_text="test",
            ),
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    text="test",
                    token_estimate=1,
                    embedding=[0.1, 0.2, 0.3],
                )
            ],
            validation=ValidationReport(passed=True),
        )
        response = result_with_emb.model_dump(
            exclude={"chunks": {"__all__": {"embedding"}}}
        )
        assert response["chunks"][0]["has_embedding"] is True
        assert "embedding" not in response["chunks"][0]


class TestDistillListDocuments:
    def test_empty(self, tmp_path: Path) -> None:
        test_store = Store(tmp_path / "test.db")
        with patch.object(server_module, "store", test_store):
            result = _impl_distill_list_documents()
        assert result["count"] == 0

    def test_with_data(self, tmp_path: Path) -> None:
        test_store = Store(tmp_path / "test.db")
        with patch.object(server_module, "store", test_store):
            _impl_distill_text("Hello.", embed=False, enrich=False, persist=True)
            result = _impl_distill_list_documents()
        assert result["count"] == 1


class TestDistillBatch:
    @pytest.mark.asyncio
    async def test_batch_persist(self, tmp_path: Path) -> None:
        """distill_batch with persist=True saves passing results to the store.

        Verifies the store shadowing bug is fixed — the module-level Store
        instance is used, not the bool parameter.
        """
        test_store = Store(tmp_path / "test.db")
        no_llm = DomainConfig()  # no prompts → skip all LLM calls
        f = tmp_path / "doc.txt"
        f.write_text("Some content here.\n\nAnother paragraph.")

        # Mock process_batch to return a passing result without LLM calls
        from distillcore.models import (
            Document,
            DocumentChunk,
            DocumentMetadata,
            ProcessingResult,
            ValidationReport,
        )

        passing_result = ProcessingResult(
            document=Document(
                metadata=DocumentMetadata(source_filename="doc.txt"),
                full_text="Some content here.\n\nAnother paragraph.",
            ),
            chunks=[DocumentChunk(chunk_index=0, text="Some content.", token_estimate=3)],
            validation=ValidationReport(passed=True, end_to_end_coverage=1.0),
        )

        async def mock_batch(*args: object, **kwargs: object) -> list[ProcessingResult]:
            return [passing_result]

        with (
            patch.object(server_module, "store", test_store),
            patch("distillcore.server.load_preset", return_value=no_llm),
            patch("distillcore.server.process_batch", side_effect=mock_batch),
        ):
            result = await _impl_distill_batch(
                [str(f)], embed=False, enrich=False, persist=True,
            )
        assert result["total"] == 1
        assert result["succeeded"] == 1
        assert result["results"][0]["stored"] is True
        assert "document_id" in result["results"][0]
        # Verify it was actually saved
        doc_id = result["results"][0]["document_id"]
        assert test_store.get_document(doc_id) is not None

    @pytest.mark.asyncio
    async def test_batch_no_persist(self, tmp_path: Path) -> None:
        """distill_batch with persist=False does not store."""
        no_llm = DomainConfig()
        f = tmp_path / "doc.txt"
        f.write_text("Some content.")
        with patch("distillcore.server.load_preset", return_value=no_llm):
            result = await _impl_distill_batch(
                [str(f)], embed=False, enrich=False, persist=False,
            )
        assert result["total"] == 1
        assert "stored" not in result["results"][0]


class TestDistillGetDocument:
    def test_not_found(self, tmp_path: Path) -> None:
        test_store = Store(tmp_path / "test.db")
        with patch.object(server_module, "store", test_store):
            result = _impl_distill_get_document("nonexistent")
        assert "error" in result

    def test_found(self, tmp_path: Path) -> None:
        test_store = Store(tmp_path / "test.db")
        with patch.object(server_module, "store", test_store):
            saved = _impl_distill_text("Hello world.", embed=False, enrich=False, persist=True)
            result = _impl_distill_get_document(saved["document_id"])
        assert result["source_filename"] == "input.txt"
        assert "chunks" in result
        assert result["chunk_count"] >= 1
