"""Tests for distillcore.pipeline.classification."""

import json
from unittest.mock import MagicMock, patch

from distillcore.config import DistillConfig, DomainConfig
from distillcore.pipeline.classification import classify_document
from distillcore.presets import load_preset


class TestClassifyDocument:
    def test_no_prompt_returns_fallback(self) -> None:
        config = DistillConfig(domain=DomainConfig(classification_prompt=""))
        meta = classify_document("test.pdf", ["page one"], 1, config)
        assert meta.document_type == "unknown"
        assert meta.source_filename == "test.pdf"

    def test_classifies_with_llm(self) -> None:
        config = DistillConfig(
            openai_api_key="test",
            domain=load_preset("generic"),
        )

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "document_type": "report",
                            "document_title": "Q4 Report",
                            "author": "Jane",
                            "date": "2024-01-15",
                            "summary": "Quarterly summary.",
                        }
                    )
                )
            )
        ]

        with patch("distillcore.pipeline.classification.get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response
            meta = classify_document("report.pdf", ["Q4 Report..."], 5, config)

        assert meta.document_type == "report"
        assert meta.document_title == "Q4 Report"
        assert meta.extra["author"] == "Jane"

    def test_llm_failure_returns_fallback(self) -> None:
        config = DistillConfig(
            openai_api_key="test",
            domain=load_preset("generic"),
        )

        with patch("distillcore.pipeline.classification.get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("fail")
            meta = classify_document("test.pdf", ["page"], 1, config)

        assert meta.document_type == "unknown"
