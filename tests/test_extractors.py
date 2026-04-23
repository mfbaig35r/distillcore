"""Tests for distillcore.extractors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from distillcore.extractors import (
    _detect_format,
    _validate_path,
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


class TestPathValidation:
    def test_unrestricted(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("data")
        result = _validate_path(f, allowed_dirs=None)
        assert result == f.resolve()

    def test_within_allowed_dir(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("data")
        result = _validate_path(f, allowed_dirs=[str(tmp_path)])
        assert result == f.resolve()

    def test_outside_allowed_dir(self, tmp_path: Path) -> None:
        with pytest.raises(PermissionError, match="Access denied"):
            _validate_path(Path("/etc/passwd"), allowed_dirs=[str(tmp_path)])

    def test_traversal_rejected(self, tmp_path: Path) -> None:
        evil = tmp_path / ".." / ".." / "etc" / "passwd"
        with pytest.raises(PermissionError, match="Access denied"):
            _validate_path(evil, allowed_dirs=[str(tmp_path)])

    def test_multiple_allowed_dirs(self, tmp_path: Path) -> None:
        subdir = tmp_path / "docs"
        subdir.mkdir()
        f = subdir / "test.txt"
        f.write_text("data")
        result = _validate_path(f, allowed_dirs=["/nonexistent", str(tmp_path)])
        assert result == f.resolve()


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


class TestDocxExtractor:
    def test_extract_paragraphs(self, tmp_path: Path) -> None:
        """Create a real .docx and extract it."""
        import docx

        doc = docx.Document()
        doc.core_properties.title = "Test Report"
        doc.core_properties.author = "Jane Doe"
        doc.add_paragraph("First paragraph.")
        doc.add_paragraph("Second paragraph.")
        f = tmp_path / "test.docx"
        doc.save(str(f))

        from distillcore.extractors.docx import DocxExtractor

        extractor = DocxExtractor()
        result = extractor.extract(f)

        assert result.format == "docx"
        assert result.page_count == 1
        assert "First paragraph." in result.full_text
        assert "Second paragraph." in result.full_text
        assert "\n\n" in result.full_text
        assert result.metadata["title"] == "Test Report"
        assert result.metadata["author"] == "Jane Doe"

    def test_extract_table(self, tmp_path: Path) -> None:
        """Tables should be extracted as tab-separated rows."""
        import docx

        doc = docx.Document()
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "A1"
        table.cell(0, 1).text = "B1"
        table.cell(1, 0).text = "A2"
        table.cell(1, 1).text = "B2"
        f = tmp_path / "table.docx"
        doc.save(str(f))

        from distillcore.extractors.docx import DocxExtractor

        result = DocxExtractor().extract(f)
        assert "A1\tB1" in result.full_text
        assert "A2\tB2" in result.full_text

    def test_registry_detection(self, tmp_path: Path) -> None:
        assert "docx" in get_registered_formats()

    def test_empty_doc(self, tmp_path: Path) -> None:
        import docx

        doc = docx.Document()
        f = tmp_path / "empty.docx"
        doc.save(str(f))

        from distillcore.extractors.docx import DocxExtractor

        result = DocxExtractor().extract(f)
        assert result.full_text == ""
        assert result.page_count == 1


class TestHtmlExtractor:
    def test_basic_extraction(self, tmp_path: Path) -> None:
        f = tmp_path / "page.html"
        f.write_text(
            "<html><head><title>My Page</title></head>"
            "<body><p>First paragraph.</p><p>Second paragraph.</p></body></html>"
        )
        from distillcore.extractors.html import HtmlExtractor

        result = HtmlExtractor().extract(f)
        assert result.format == "html"
        assert "First paragraph." in result.full_text
        assert "Second paragraph." in result.full_text
        assert result.metadata["title"] == "My Page"

    def test_strips_script_and_style(self, tmp_path: Path) -> None:
        f = tmp_path / "noisy.html"
        f.write_text(
            "<html><body>"
            "<script>var x = 1;</script>"
            "<style>body { color: red; }</style>"
            "<p>Real content.</p>"
            "</body></html>"
        )
        from distillcore.extractors.html import HtmlExtractor

        result = HtmlExtractor().extract(f)
        assert "var x" not in result.full_text
        assert "color: red" not in result.full_text
        assert "Real content." in result.full_text

    def test_strips_nav_footer(self, tmp_path: Path) -> None:
        f = tmp_path / "layout.html"
        f.write_text(
            "<html><body>"
            "<nav>Menu items</nav>"
            "<main><p>Main content.</p></main>"
            "<footer>Copyright 2026</footer>"
            "</body></html>"
        )
        from distillcore.extractors.html import HtmlExtractor

        result = HtmlExtractor().extract(f)
        assert "Menu items" not in result.full_text
        assert "Copyright" not in result.full_text
        assert "Main content." in result.full_text

    def test_extracts_author_meta(self, tmp_path: Path) -> None:
        f = tmp_path / "meta.html"
        f.write_text(
            '<html><head><meta name="author" content="Jane Doe"></head>'
            "<body><p>Content.</p></body></html>"
        )
        from distillcore.extractors.html import HtmlExtractor

        result = HtmlExtractor().extract(f)
        assert result.metadata["author"] == "Jane Doe"

    def test_empty_html(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.html"
        f.write_text("<html><body></body></html>")
        from distillcore.extractors.html import HtmlExtractor

        result = HtmlExtractor().extract(f)
        assert result.page_count == 1

    def test_registry_detection(self) -> None:
        formats = get_registered_formats()
        assert "html" in formats
        assert "htm" in formats


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
