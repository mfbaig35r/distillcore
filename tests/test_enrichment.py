"""Tests for distillcore.pipeline.enrichment."""

import json
from unittest.mock import MagicMock, patch

from distillcore.config import DistillConfig, DomainConfig
from distillcore.models import DocumentChunk
from distillcore.pipeline.enrichment import enrich_chunks
from distillcore.presets import load_preset


class TestEnrichChunks:
    def test_no_prompt_returns_unchanged(self) -> None:
        config = DistillConfig(domain=DomainConfig(enrichment_prompt=""))
        chunks = [DocumentChunk(chunk_index=0, text="test", token_estimate=1)]
        result = enrich_chunks(chunks, "report", config)
        assert result[0].topic is None

    def test_enriches_with_llm(self) -> None:
        config = DistillConfig(
            openai_api_key="test",
            domain=load_preset("generic"),
        )
        chunks = [
            DocumentChunk(chunk_index=0, text="Budget analysis for Q4", token_estimate=5),
            DocumentChunk(chunk_index=1, text="Risk assessment details", token_estimate=4),
        ]

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "enrichments": [
                                {
                                    "chunk_index": 0,
                                    "topic": "Budget Analysis",
                                    "key_concepts": ["budget", "Q4"],
                                    "relevance": "high",
                                },
                                {
                                    "chunk_index": 1,
                                    "topic": "Risk Assessment",
                                    "key_concepts": ["risk"],
                                    "relevance": "medium",
                                },
                            ]
                        }
                    )
                )
            )
        ]

        with patch("distillcore.pipeline.enrichment.get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response
            result = enrich_chunks(chunks, "report", config)

        assert result[0].topic == "Budget Analysis"
        assert result[0].relevance == "high"
        assert result[1].topic == "Risk Assessment"

    def test_llm_failure_returns_unenriched(self) -> None:
        config = DistillConfig(
            openai_api_key="test",
            domain=load_preset("generic"),
        )
        chunks = [DocumentChunk(chunk_index=0, text="test", token_estimate=1)]

        with patch("distillcore.pipeline.enrichment.get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("API error")
            result = enrich_chunks(chunks, "report", config)

        assert result[0].topic is None
