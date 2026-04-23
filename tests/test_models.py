"""Tests for distillcore.models."""

from distillcore.models import (
    ChunkedDocument,
    Document,
    DocumentChunk,
    DocumentMetadata,
    ExtractionResult,
    PageText,
    ProcessingResult,
    Section,
    TranscriptTurn,
    ValidationReport,
)


class TestPageText:
    def test_create(self) -> None:
        p = PageText(page_number=1, text="hello")
        assert p.page_number == 1
        assert p.text == "hello"


class TestExtractionResult:
    def test_defaults(self) -> None:
        r = ExtractionResult()
        assert r.pages == []
        assert r.full_text == ""
        assert r.page_count == 0
        assert r.format == "unknown"
        assert r.metadata == {}

    def test_with_pages(self) -> None:
        pages = [PageText(page_number=1, text="p1"), PageText(page_number=2, text="p2")]
        r = ExtractionResult(pages=pages, full_text="p1\n\np2", page_count=2, format="pdf")
        assert r.page_count == 2
        assert r.format == "pdf"


class TestSection:
    def test_nested(self) -> None:
        child = Section(heading="Sub", content="sub content")
        parent = Section(heading="Top", content="top content", subsections=[child])
        assert len(parent.subsections) == 1
        assert parent.subsections[0].heading == "Sub"

    def test_defaults(self) -> None:
        s = Section()
        assert s.heading is None
        assert s.section_type == "general"
        assert s.content == ""
        assert s.subsections == []


class TestTranscriptTurn:
    def test_create(self) -> None:
        t = TranscriptTurn(speaker="Alice", content="Hello")
        assert t.speaker == "Alice"
        assert t.role == "unknown"
        assert t.page is None


class TestDocumentMetadata:
    def test_defaults(self) -> None:
        m = DocumentMetadata(source_filename="test.pdf")
        assert m.document_type == "unknown"
        assert m.page_count == 0
        assert m.extra == {}

    def test_extra(self) -> None:
        m = DocumentMetadata(
            source_filename="brief.pdf",
            extra={"case_number": "2024-CV-001", "court": "Superior Court"},
        )
        assert m.extra["case_number"] == "2024-CV-001"


class TestDocument:
    def test_minimal(self) -> None:
        doc = Document(metadata=DocumentMetadata(source_filename="test.txt"))
        assert doc.sections == []
        assert doc.transcript_turns == []
        assert doc.full_text == ""


class TestDocumentChunk:
    def test_defaults(self) -> None:
        c = DocumentChunk(chunk_index=0, text="chunk text", token_estimate=3)
        assert c.topic is None
        assert c.key_concepts == []
        assert c.embedding is None

    def test_with_enrichment(self) -> None:
        c = DocumentChunk(
            chunk_index=0,
            text="chunk",
            token_estimate=1,
            topic="Introduction",
            key_concepts=["overview", "summary"],
            relevance="high",
        )
        assert c.topic == "Introduction"
        assert len(c.key_concepts) == 2


class TestChunkedDocument:
    def test_create(self) -> None:
        chunks = [DocumentChunk(chunk_index=0, text="a", token_estimate=1)]
        cd = ChunkedDocument(chunk_count=1, chunks=chunks)
        assert cd.chunk_count == 1


class TestValidationReport:
    def test_defaults(self) -> None:
        v = ValidationReport()
        assert v.passed is False
        assert v.warnings == []

    def test_passed(self) -> None:
        v = ValidationReport(
            structuring_coverage=0.98,
            chunking_coverage=0.99,
            end_to_end_coverage=0.97,
            passed=True,
        )
        assert v.passed is True


class TestProcessingResult:
    def test_create(self) -> None:
        doc = Document(metadata=DocumentMetadata(source_filename="test.txt"))
        result = ProcessingResult(
            document=doc,
            chunks=[],
            validation=ValidationReport(),
        )
        assert result.document.metadata.source_filename == "test.txt"
        assert result.chunks == []


class TestModelSerialization:
    def test_roundtrip(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.pdf", page_count=5),
            sections=[Section(heading="Intro", content="Hello world")],
            full_text="Hello world",
        )
        data = doc.model_dump()
        restored = Document.model_validate(data)
        assert restored.metadata.source_filename == "test.pdf"
        assert restored.sections[0].heading == "Intro"
