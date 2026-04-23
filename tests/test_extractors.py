"""Tests for distillcore.extractors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillcore.extractors import (
    _detect_format,
    extract,
    get_registered_formats,
    register_extractor,
)
from distillcore.extractors.text import TextExtractor
from distillcore.models import ExtractionResult, PageText


class TestTextExtractor:
    def test_extract_txt(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("Hello world\n\nSecond paragraph")
        extractor = TextExtractor()
        result = extractor.extract(f)
        assert result.format == "txt"
        assert result.page_count == 1
        assert "Hello world" in result.full_text
        assert result.pages[0].page_number == 1

    def test_extract_markdown(self, tmp_path: Path) -> None:
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nContent here")
        extractor = TextExtractor()
        result = extractor.extract(f)
        assert "# Title" in result.full_text

    def test_formats(self) -> None:
        assert "txt" in TextExtractor.formats
        assert "md" in TextExtractor.formats


class TestDetectFormat:
    def test_pdf(self) -> None:
        assert _detect_format(Path("file.pdf")) == "pdf"

    def test_txt(self) -> None:
        assert _detect_format(Path("file.txt")) == "txt"

    def test_no_extension(self) -> None:
        assert _detect_format(Path("README")) == "txt"

    def test_uppercase(self) -> None:
        assert _detect_format(Path("FILE.PDF")) == "pdf"


class TestExtractRegistry:
    def test_text_registered(self) -> None:
        assert "txt" in get_registered_formats()
        assert "md" in get_registered_formats()

    def test_extract_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("content")
        result = extract(f)
        assert result.full_text == "content"

    def test_extract_with_format_override(self, tmp_path: Path) -> None:
        f = tmp_path / "data.dat"
        f.write_text("plain text data")
        result = extract(f, format="txt")
        assert result.full_text == "plain text data"

    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(ValueError, match="No extractor registered"):
            extract(f)

    def test_custom_extractor(self, tmp_path: Path) -> None:
        class CsvExtractor:
            formats = ["csv"]

            def extract(self, source, config=None):
                return ExtractionResult(
                    pages=[PageText(page_number=1, text="csv data")],
                    full_text="csv data",
                    page_count=1,
                    format="csv",
                )

        register_extractor(CsvExtractor())
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        result = extract(f)
        assert result.format == "csv"

    def test_text_extractor_ignores_config(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("content")
        extractor = TextExtractor()
        result = extractor.extract(f, config={"arbitrary": "config"})
        assert result.full_text == "content"


class TestPdfExtractor:
    def test_extract_pdf(self, tmp_path: Path) -> None:
        """Test PDF extraction with mocked pdfplumber."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page one text"

        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [mock_page]

        with patch("distillcore.extractors.pdf.pdfplumber") as mock_plumber:
            mock_plumber.open.return_value = mock_pdf
            from distillcore.extractors.pdf import PdfExtractor

            extractor = PdfExtractor()
            result = extractor.extract(tmp_path / "test.pdf")

        assert result.format == "pdf"
        assert result.page_count == 1
        assert "Page one text" in result.full_text
