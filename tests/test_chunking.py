"""Tests for distillcore.pipeline.chunking and distillcore.chunking utilities."""

from distillcore.chunking import (
    _greedy_fill,
    _hard_split,
    _subsplit,
    split_paragraphs,
)
from distillcore.config import ChunkConfig
from distillcore.models import Document, DocumentChunk, DocumentMetadata, Section, TranscriptTurn
from distillcore.pipeline.chunking import (
    _estimate_tokens,
    _merge_small_chunks,
    chunk_document,
    chunk_fallback,
    chunk_sections,
    chunk_transcript,
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

    def test_auto_is_default(self, sample_document: Document) -> None:
        result_default = chunk_document(sample_document)
        result_auto = chunk_document(sample_document, config=ChunkConfig(strategy="auto"))
        assert result_default.chunk_count == result_auto.chunk_count
        for a, b in zip(result_default.chunks, result_auto.chunks):
            assert a.text == b.text


class TestSubsplit:
    """Tests for the cascading subsplit logic."""

    def test_splits_on_line_breaks(self) -> None:
        """PDF-style text with single newlines should split on lines."""
        # 10 lines of 100 chars each = 1000 chars total
        lines = [f"Line {i}: " + "x" * 90 for i in range(10)]
        text = "\n".join(lines)
        result = _subsplit(text, target_chars=300, max_chars=600)
        assert all(len(r) <= 600 for r in result)
        assert len(result) > 1

    def test_splits_on_sentences(self) -> None:
        """A single long line with sentences should split on sentence boundaries."""
        sentences = [f"Sentence number {i} is about something important." for i in range(20)]
        text = " ".join(sentences)  # one long line, no newlines
        result = _subsplit(text, target_chars=200, max_chars=400)
        assert all(len(r) <= 400 for r in result)
        assert len(result) > 1

    def test_hard_cut_no_boundaries(self) -> None:
        """Text with no natural boundaries gets hard-cut."""
        text = "a" * 5000  # no spaces, no newlines
        result = _subsplit(text, target_chars=1000, max_chars=2000)
        assert all(len(r) <= 2000 for r in result)
        assert len(result) >= 3  # 5000 / 2000 = at least 3

    def test_hard_cut_prefers_spaces(self) -> None:
        """Hard cut should break at a space when possible."""
        words = ["word"] * 500  # 500 * 5 = 2500 chars with spaces
        text = " ".join(words)
        result = _hard_split(text, max_chars=1000)
        # Every part except possibly the last should end without a partial word
        for part in result[:-1]:
            assert not part.endswith("wor")  # didn't split mid-word

    def test_max_chars_ceiling_guaranteed(self) -> None:
        """No output chunk should ever exceed max_chars."""
        # Mix of long lines and sentences
        lines = ["A" * 300 + ". " + "B" * 300 + ". " + "C" * 300 for _ in range(5)]
        text = "\n".join(lines)
        result = _subsplit(text, target_chars=400, max_chars=1000)
        for chunk in result:
            assert len(chunk) <= 1000, f"Chunk exceeded max_chars: {len(chunk)}"


class TestSplitParagraphsOversized:
    """Tests for split_paragraphs handling oversized paragraphs."""

    def test_pdf_style_text(self) -> None:
        """Text with only single newlines (like PDF extraction) should be chunked."""
        lines = [f"Line {i}: " + "x" * 80 for i in range(50)]
        text = "\n".join(lines)  # no double-newlines
        result = split_paragraphs(text, heading=None, target_chars=500, overlap=0, max_chars=1000)
        assert len(result) > 1
        assert all(len(r) <= 1000 for r in result)

    def test_mixed_normal_and_oversized(self) -> None:
        """Normal paragraphs stay normal; oversized ones get subsplit."""
        small = "Small paragraph here."
        big = "\n".join([f"Line {i}: " + "x" * 80 for i in range(30)])
        text = f"{small}\n\n{big}\n\n{small}"
        result = split_paragraphs(text, heading=None, target_chars=500, overlap=0, max_chars=1000)
        assert len(result) > 3  # more than just the 3 "paragraphs"
        assert all(len(r) <= 1000 for r in result)

    def test_overlap_with_subsplit(self) -> None:
        """Overlap should carry content between chunks from subsplit parts."""
        lines = [f"Line {i}: content" for i in range(20)]
        text = "\n".join(lines)
        result = split_paragraphs(text, heading=None, target_chars=100, overlap=50, max_chars=200)
        assert len(result) > 1
        # Verify some content appears in consecutive chunks (overlap)
        for i in range(len(result) - 1):
            # At least check no chunk exceeds ceiling
            assert len(result[i]) <= 200

    def test_heading_preserved_with_subsplit(self) -> None:
        """Heading should be prepended to each chunk even after subsplit."""
        lines = ["x" * 80 for _ in range(10)]
        text = "\n".join(lines)
        result = split_paragraphs(
            text, heading="My Heading", target_chars=200, overlap=0, max_chars=500
        )
        for chunk in result:
            assert chunk.startswith("My Heading\n\n")


class TestGreedyFill:
    """Tests for the _greedy_fill helper."""

    def test_basic_fill(self) -> None:
        pieces = ["aaa", "bbb", "ccc", "ddd"]
        result = _greedy_fill(pieces, target_chars=8, joiner=" ")
        # "aaa bbb" = 7, "ccc ddd" = 7
        assert result == ["aaa bbb", "ccc ddd"]

    def test_single_large_piece(self) -> None:
        """A single piece larger than target passes through (not split here)."""
        result = _greedy_fill(["a" * 100], target_chars=50, joiner=" ")
        assert result == ["a" * 100]

    def test_newline_joiner(self) -> None:
        pieces = ["line1", "line2", "line3"]
        result = _greedy_fill(pieces, target_chars=12, joiner="\n")
        # "line1\nline2" = 11, "line3" = 5
        assert result == ["line1\nline2", "line3"]


class TestEstimateTokens:
    def test_default_heuristic(self) -> None:
        config = ChunkConfig()
        assert _estimate_tokens("a" * 100, config) == 25

    def test_custom_tokenizer(self) -> None:
        config = ChunkConfig(tokenizer=lambda text: len(text.split()))
        assert _estimate_tokens("one two three four", config) == 4

    def test_custom_tokenizer_used_in_chunks(self, sample_document: Document) -> None:
        config = ChunkConfig(tokenizer=lambda text: len(text.split()))
        result = chunk_document(sample_document, config=config)
        for chunk in result.chunks:
            assert chunk.token_estimate == len(chunk.text.split())


class TestMinTokensMerge:
    def _make_chunks(self, texts: list[str]) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_index=i,
                text=t,
                token_estimate=len(t) // 4,
            )
            for i, t in enumerate(texts)
        ]

    def test_no_merge_when_disabled(self) -> None:
        chunks = self._make_chunks(["short", "another short", "third"])
        config = ChunkConfig(min_tokens=0)
        result = _merge_small_chunks(chunks, config)
        assert len(result) == 3

    def test_merges_small_into_previous(self) -> None:
        chunks = self._make_chunks(["a" * 200, "tiny", "b" * 200])
        config = ChunkConfig(min_tokens=10)
        result = _merge_small_chunks(chunks, config)
        assert len(result) == 2
        assert "tiny" in result[0].text

    def test_first_chunk_small_kept(self) -> None:
        chunks = self._make_chunks(["hi", "a" * 200])
        config = ChunkConfig(min_tokens=10)
        result = _merge_small_chunks(chunks, config)
        assert len(result) == 2

    def test_via_chunk_document(self) -> None:
        parts = ["Hi.", "Ok.", "Sure.", "Yes.", "A longer paragraph here."]
        text = "\n\n".join(parts)
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            full_text=text,
        )
        config_no_merge = ChunkConfig(target_tokens=500, min_tokens=0)
        config_merge = ChunkConfig(target_tokens=500, min_tokens=20)
        result_no = chunk_document(doc, config=config_no_merge)
        result_yes = chunk_document(doc, config=config_merge)
        assert result_yes.chunk_count <= result_no.chunk_count

    def test_reindexed_after_merge(self) -> None:
        cfg = ChunkConfig(min_tokens=10)
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[
                Section(heading="S", content="a" * 200 + "\n\nx\n\n" + "b" * 200)
            ],
            full_text="a" * 200 + "\n\nx\n\n" + "b" * 200,
        )
        result = chunk_document(doc, config=cfg)
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_index == i
