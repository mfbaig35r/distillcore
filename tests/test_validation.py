"""Tests for distillcore.validation."""

from distillcore.models import (
    ChunkedDocument,
    Document,
    DocumentChunk,
    DocumentMetadata,
    Section,
)
from distillcore.validation.checks import (
    validate_chunking,
    validate_end_to_end,
    validate_extraction,
    validate_structuring,
)
from distillcore.validation.coverage import (
    compute_coverage,
    find_missing_segments,
    normalize_text,
)


class TestNormalizeText:
    def test_collapses_whitespace(self) -> None:
        assert normalize_text("hello   world") == "hello world"

    def test_strips(self) -> None:
        assert normalize_text("  hello  ") == "hello"

    def test_empty(self) -> None:
        assert normalize_text("") == ""

    def test_unicode_normalization(self) -> None:
        # NFKC normalizes full-width chars
        assert normalize_text("\uff28ello") == "Hello"


class TestComputeCoverage:
    def test_identical(self) -> None:
        assert compute_coverage("hello world", "hello world") == 1.0

    def test_subset(self) -> None:
        cov = compute_coverage("hello world foo", "hello world")
        assert 0.5 < cov < 1.0

    def test_empty_original(self) -> None:
        assert compute_coverage("", "anything") == 1.0

    def test_no_overlap(self) -> None:
        cov = compute_coverage("alpha beta gamma", "xyz uvw")
        assert cov == 0.0

    def test_case_insensitive(self) -> None:
        assert compute_coverage("Hello World", "hello world") == 1.0

    def test_no_substring_false_positive(self) -> None:
        # "to" should NOT match inside "tomato", "be" should NOT match inside "beer"
        assert compute_coverage("to be", "tomato beer") == 0.0


class TestFindMissingSegments:
    def test_nothing_missing(self) -> None:
        text = "The quick brown fox jumped over the lazy dog. " * 3
        assert find_missing_segments(text, text) == []

    def test_finds_missing(self) -> None:
        original = (
            "This is the first sentence that is long enough to meet the threshold. "
            "This unique sentence about elephants dancing on clouds will not be found."
        )
        derived = (
            "This is the first sentence that is long enough to meet the threshold."
        )
        missing = find_missing_segments(original, derived, min_length=30)
        assert len(missing) >= 1
        assert any("elephants" in s for s in missing)

    def test_empty_original(self) -> None:
        assert find_missing_segments("", "something") == []

    def test_short_segments_ignored(self) -> None:
        assert find_missing_segments("Short. Also short.", "other", min_length=50) == []


class TestValidateExtraction:
    def test_lossless(self) -> None:
        pages = ["page one", "page two"]
        full = "page one\n\npage two"
        validate_extraction(full, pages)  # should not raise

    def test_lossy(self) -> None:
        try:
            validate_extraction("wrong", ["page one", "page two"])
            assert False, "Should have raised"
        except AssertionError:
            pass


class TestValidateStructuring:
    def test_high_coverage(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(heading="All", content="The complete text of the document")],
            full_text="The complete text of the document",
        )
        report = validate_structuring(doc, threshold=0.90)
        assert report.passed is True
        assert report.structuring_coverage >= 0.90

    def test_low_coverage(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(heading="Partial", content="only a bit")],
            full_text="The complete text of the document with many more words",
        )
        report = validate_structuring(doc, threshold=0.95)
        assert report.passed is False
        assert len(report.warnings) > 0


class TestValidateChunking:
    def test_good_chunking(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(content="hello world foo bar")],
            full_text="hello world foo bar",
        )
        chunked = ChunkedDocument(
            chunk_count=1,
            chunks=[DocumentChunk(chunk_index=0, text="hello world foo bar", token_estimate=4)],
        )
        report = validate_chunking(doc, chunked, threshold=0.90)
        assert report.passed is True

    def test_empty_chunks_warned(self) -> None:
        doc = Document(
            metadata=DocumentMetadata(source_filename="test.txt"),
            sections=[Section(content="text")],
            full_text="text",
        )
        chunked = ChunkedDocument(
            chunk_count=2,
            chunks=[
                DocumentChunk(chunk_index=0, text="text", token_estimate=1),
                DocumentChunk(chunk_index=1, text="", token_estimate=0),
            ],
        )
        report = validate_chunking(doc, chunked, threshold=0.0)
        assert any("Empty chunks" in w for w in report.warnings)


class TestValidateEndToEnd:
    def test_high_coverage(self) -> None:
        text = "The full document text with all words"
        chunked = ChunkedDocument(
            chunk_count=1,
            chunks=[DocumentChunk(chunk_index=0, text=text, token_estimate=7)],
        )
        report = validate_end_to_end(text, chunked, threshold=0.90)
        assert report.passed is True

    def test_low_coverage(self) -> None:
        chunked = ChunkedDocument(
            chunk_count=1,
            chunks=[DocumentChunk(chunk_index=0, text="tiny", token_estimate=1)],
        )
        report = validate_end_to_end(
            "A much longer document with many words", chunked, threshold=0.90
        )
        assert report.passed is False
