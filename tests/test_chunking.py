"""Tests for distillcore.pipeline.chunking."""

from distillcore.config import ChunkConfig
from distillcore.models import Document, DocumentMetadata, Section, TranscriptTurn
from distillcore.pipeline.chunking import (
    chunk_document,
    chunk_fallback,
    chunk_sections,
    chunk_transcript,
    split_paragraphs,
)


class TestSplitParagraphs:
    def test_single_paragraph(self) -> None:
        result = split_paragraphs("Short text", heading=None, target_chars=1000, overlap=0)
        assert len(result) == 1
        assert result[0] == "Short text"

    def test_multiple_paragraphs_under_target(self) -> None:
        text = "Para one.\n\nPara two."
        result = split_paragraphs(text, heading=None, target_chars=1000, overlap=0)
        assert len(result) == 1
        assert "Para one." in result[0]
        assert "Para two." in result[0]

    def test_splits_on_target(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100 + "\n\n" + "C" * 100
        result = split_paragraphs(text, heading=None, target_chars=150, overlap=0)
        assert len(result) >= 2

    def test_heading_prepended(self) -> None:
        result = split_paragraphs("Content", heading="Title", target_chars=1000, overlap=0)
        assert result[0].startswith("Title\n\n")

    def test_overlap(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        result = split_paragraphs(text, heading=None, target_chars=20, overlap=15)
        # With overlap, later chunks should contain text from previous chunks
        assert len(result) >= 2

    def test_empty_text(self) -> None:
        result = split_paragraphs("", heading=None, target_chars=100, overlap=0)
        assert len(result) == 1


class TestChunkSections:
    def test_single_section(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(heading="Intro", content="Hello world", section_type="header")],
            full_text="Hello world",
        )
        chunks = chunk_sections(doc, target_chars=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0]["section_heading"] == "Intro"
        assert chunks[0]["section_type"] == "header"

    def test_nested_sections(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[
                Section(
                    heading="Top",
                    content="Top content",
                    subsections=[Section(heading="Sub", content="Sub content")],
                )
            ],
            full_text="Top content Sub content",
        )
        chunks = chunk_sections(doc, target_chars=1000, overlap=0)
        assert len(chunks) == 2
        assert chunks[1]["section_heading"] == "Top > Sub"

    def test_large_section_splits(self) -> None:
        content = "Para one.\n\n" + "Para two.\n\n" + "Para three."
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(heading="Big", content=content)],
            full_text=content,
        )
        chunks = chunk_sections(doc, target_chars=20, overlap=0)
        assert len(chunks) >= 2


class TestChunkTranscript:
    def test_groups_turns(self, transcript_document: Document) -> None:
        chunks = chunk_transcript(transcript_document, target_chars=10000)
        assert len(chunks) >= 1
        assert chunks[0]["section_type"] == "transcript"
        assert "Alice" in chunks[0]["speakers"]

    def test_splits_on_target(self) -> None:
        turns = [
            TranscriptTurn(speaker="A", content="x" * 100, page=i)
            for i in range(10)
        ]
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.pdf"),
            transcript_turns=turns,
            full_text="".join(t.content for t in turns),
        )
        chunks = chunk_transcript(doc, target_chars=200)
        assert len(chunks) >= 2

    def test_page_range(self, transcript_document: Document) -> None:
        chunks = chunk_transcript(transcript_document, target_chars=10000)
        assert chunks[0]["page_start"] == 1
        assert chunks[0]["page_end"] == 2


class TestChunkFallback:
    def test_fallback(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            full_text="Para one.\n\nPara two.\n\nPara three.",
        )
        chunks = chunk_fallback(doc, target_chars=1000, overlap=0)
        assert len(chunks) == 1
        assert chunks[0]["section_type"] == "full_text"

    def test_empty_text(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            full_text="",
        )
        chunks = chunk_fallback(doc, target_chars=1000, overlap=0)
        assert chunks == []


class TestChunkDocument:
    def test_uses_sections(self, sample_document: Document) -> None:
        result = chunk_document(sample_document)
        assert result.chunk_count >= 1
        assert result.chunks[0].section_heading == "Test"

    def test_uses_transcript(self, transcript_document: Document) -> None:
        result = chunk_document(transcript_document)
        assert result.chunk_count >= 1
        assert result.chunks[0].section_type == "transcript"

    def test_fallback_no_sections(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            full_text="Just plain text here.",
        )
        result = chunk_document(doc)
        assert result.chunk_count >= 1

    def test_custom_config(self, sample_document: Document) -> None:
        config = ChunkConfig(target_tokens=10, overlap_chars=0)
        result = chunk_document(sample_document, config=config)
        assert result.chunk_count >= 1

    def test_token_estimate(self, sample_document: Document) -> None:
        result = chunk_document(sample_document)
        for chunk in result.chunks:
            assert chunk.token_estimate == len(chunk.text) // 4
