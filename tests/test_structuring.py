"""Tests for distillcore.pipeline.structuring."""

import json
from unittest.mock import MagicMock, patch

from distillcore.config import DistillConfig, DomainConfig
from distillcore.pipeline.structuring import (
    parse_structure_result,
    structure_document,
)
from distillcore.presets import load_preset


class TestStructureDocument:
    def test_no_prompt_returns_empty(self) -> None:
        config = DistillConfig(domain=DomainConfig(structuring_prompt=""))
        result = structure_document("text", "report", "test.txt", config)
        assert result == {"sections": []}

    def test_structures_with_llm(self) -> None:
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
                            "sections": [
                                {
                                    "heading": "Introduction",
                                    "section_type": "header",
                                    "content": "Intro text",
                                    "subsections": [],
                                    "page_range": [1, 1],
                                }
                            ]
                        }
                    )
                )
            )
        ]

        with patch("distillcore.pipeline.structuring.get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response
            result = structure_document("Intro text", "report", "doc.pdf", config)

        assert len(result["sections"]) == 1
        assert result["sections"][0]["heading"] == "Introduction"


class TestParseStructureResult:
    def test_parse_sections(self) -> None:
        result = {
            "sections": [
                {
                    "heading": "Top",
                    "section_type": "header",
                    "content": "top content",
                    "subsections": [
                        {
                            "heading": "Sub",
                            "section_type": "body",
                            "content": "sub content",
                            "page_range": [2, 3],
                        }
                    ],
                    "page_range": [1, 3],
                }
            ]
        }
        sections, turns = parse_structure_result(result)
        assert len(sections) == 1
        assert sections[0].heading == "Top"
        assert sections[0].page_start == 1
        assert len(sections[0].subsections) == 1
        assert sections[0].subsections[0].heading == "Sub"
        assert turns == []

    def test_parse_transcript_turns(self) -> None:
        result = {
            "sections": [],
            "transcript_turns": [
                {"speaker": "Alice", "role": "attorney", "content": "Question?", "page": 1},
                {"speaker": "Bob", "role": "witness", "content": "Answer.", "page": 1},
            ],
        }
        sections, turns = parse_structure_result(result)
        assert len(turns) == 2
        assert turns[0].speaker == "Alice"
        assert turns[1].content == "Answer."

    def test_empty_result(self) -> None:
        sections, turns = parse_structure_result({})
        assert sections == []
        assert turns == []
