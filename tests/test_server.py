"""Tests for distillcore.server."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from distillcore import server as server_module
from distillcore.server import (
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
