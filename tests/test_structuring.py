"""Tests for distillcore.pipeline.structuring."""

import json
from unittest.mock import MagicMock, patch

from distillcore.config import DistillConfig, DomainConfig
from distillcore.pipeline.structuring import (
    _populate_section_content,
    parse_structure_result,
    structure_document,
)
from distillcore.presets import load_preset


class TestStructureDocument:
    def test_no_prompt_returns_empty_with_error(self) -> None:
        config = DistillConfig(domain=DomainConfig(structuring_prompt=""))
        result = structure_document("text", "report", "test.txt", config)
        assert result["sections"] == []
        assert "No structuring prompt" in result["_structuring_error"]

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
    def test_parse_boundary_sections_with_pages(self) -> None:
        """Boundary-only LLM response + pages_text → content populated from pages."""
        result = {
            "sections": [
                {
                    "heading": "Top",
                    "section_type": "header",
                    "subsections": [
                        {
                            "heading": "Sub",
                            "section_type": "body",
                            "page_range": [2, 3],
                        }
                    ],
                    "page_range": [1, 3],
                }
            ]
        }
        pages_text = ["Page one text.", "Page two text.", "Page three text."]
        sections, turns, error = parse_structure_result(result, pages_text=pages_text)
        assert len(sections) == 1
        assert sections[0].heading == "Top"
        assert sections[0].page_start == 1
        assert sections[0].content == "Page one text.\n\nPage two text.\n\nPage three text."
        assert len(sections[0].subsections) == 1
        assert sections[0].subsections[0].heading == "Sub"
        assert sections[0].subsections[0].content == "Page two text.\n\nPage three text."
        assert turns == []

    def test_parse_sections_without_pages_text(self) -> None:
        """Without pages_text, content stays as whatever the LLM returned."""
        result = {
            "sections": [
                {
                    "heading": "Top",
                    "section_type": "header",
                    "content": "llm provided content",
                    "page_range": [1, 1],
                }
            ]
        }
        sections, turns, error = parse_structure_result(result, pages_text=None)
        assert sections[0].content == "llm provided content"

    def test_parse_boundary_no_content_no_pages(self) -> None:
        """Boundary-only response without pages_text → empty content."""
        result = {
            "sections": [
                {
                    "heading": "Top",
                    "section_type": "header",
                    "page_range": [1, 1],
                }
            ]
        }
        sections, _, error = parse_structure_result(result)
        assert sections[0].content == ""

    def test_parse_transcript_turns(self) -> None:
        result = {
            "sections": [],
            "transcript_turns": [
                {"speaker": "Alice", "role": "attorney", "content": "Question?", "page": 1},
                {"speaker": "Bob", "role": "witness", "content": "Answer.", "page": 1},
            ],
        }
        sections, turns, error = parse_structure_result(result)
        assert len(turns) == 2
        assert turns[0].speaker == "Alice"
        assert turns[1].content == "Answer."

    def test_empty_result(self) -> None:
        sections, turns, error = parse_structure_result({})
        assert sections == []
        assert turns == []
        assert error is None

    def test_structuring_error_propagated(self) -> None:
        result = {"sections": [], "_structuring_error": "API timeout"}
        sections, turns, error = parse_structure_result(result)
        assert sections == []
        assert error == "API timeout"


class TestPopulateSectionContent:
    def test_single_page_section(self) -> None:
        from distillcore.models import Section

        section = Section(heading="Intro", page_start=1, page_end=1)
        pages = ["First page.", "Second page."]
        _populate_section_content(section, pages)
        assert section.content == "First page."

    def test_multi_page_section(self) -> None:
        from distillcore.models import Section

        section = Section(heading="Body", page_start=2, page_end=4)
        pages = ["p1", "p2", "p3", "p4", "p5"]
        _populate_section_content(section, pages)
        assert section.content == "p2\n\np3\n\np4"

    def test_preserves_existing_content(self) -> None:
        """If LLM provided content (custom preset), don't overwrite it."""
        from distillcore.models import Section

        section = Section(heading="X", content="custom", page_start=1, page_end=1)
        _populate_section_content(section, ["page text"])
        assert section.content == "custom"

    def test_missing_page_range(self) -> None:
        from distillcore.models import Section

        section = Section(heading="No range")
        _populate_section_content(section, ["page text"])
        assert section.content == ""

    def test_out_of_bounds(self) -> None:
        from distillcore.models import Section

        section = Section(heading="Bad", page_start=5, page_end=10)
        _populate_section_content(section, ["p1", "p2"])
        assert section.content == ""

    def test_recurses_into_subsections(self) -> None:
        from distillcore.models import Section

        child = Section(heading="Child", page_start=2, page_end=2)
        parent = Section(heading="Parent", page_start=1, page_end=2, subsections=[child])
        pages = ["page one", "page two"]
        _populate_section_content(parent, pages)
        assert parent.content == "page one\n\npage two"
        assert child.content == "page two"
