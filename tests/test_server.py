"""Tests for distillcore.server."""

from unittest.mock import patch

from distillcore.server import (
    _impl_distill_chunks_only,
    _impl_distill_text,
    _impl_distill_validate,
)


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
        result = _impl_distill_validate(
            "hello world foo bar baz qux", ["hello world"]
        )
        assert result["coverage"] < 1.0


class TestDistillText:
    def test_no_llm(self) -> None:
        """Test distill_text with mocked LLM calls."""
        import json
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "document_type": "memo",
                "document_title": "Test",
                "sections": [{"heading": "Body", "section_type": "body", "content": "Some text."}],
            })))
        ]

        with patch("distillcore.pipeline.classification.get_client") as mc, \
             patch("distillcore.pipeline.structuring.get_client") as ms:
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
