"""PDF text extraction using pdfplumber, with vision OCR fallback for scanned pages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pdfplumber

from ..models import ExtractionResult, PageText

logger = logging.getLogger(__name__)

# Pages with fewer chars than this are considered empty (likely scanned)
_EMPTY_PAGE_THRESHOLD = 20


class PdfExtractor:
    """Extract text from PDF files using pdfplumber with OCR fallback."""

    formats = ["pdf"]

    def extract(
        self,
        source: Path | str,
        config: Any = None,
    ) -> ExtractionResult:
        """Extract text from a PDF page-by-page.

        Uses pdfplumber for text-based pages. If more than half the pages are
        empty (scanned images), falls back to vision OCR for those pages.

        Reads enable_ocr and api_key from config if provided.
        """
        enable_ocr = getattr(config, "enable_ocr", True) if config else True
        api_key = config.resolve_api_key() if config and hasattr(config, "resolve_api_key") else ""
        pdf_path = Path(source)
        pages: list[PageText] = []

        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(PageText(page_number=i, text=text))

        # Detect and OCR scanned pages
        empty_pages = [p for p in pages if len(p.text.strip()) < _EMPTY_PAGE_THRESHOLD]

        if (
            enable_ocr
            and len(empty_pages) > len(pages) * 0.5
            and len(pages) > 1
        ):
            logger.info(
                f"{pdf_path.name}: {len(empty_pages)}/{len(pages)} pages empty, "
                f"running vision OCR"
            )
            pages = _ocr_empty_pages(pdf_path, pages, api_key=api_key)

        full_text = "\n\n".join(p.text for p in pages)

        return ExtractionResult(
            pages=pages,
            full_text=full_text,
            page_count=len(pages),
            format="pdf",
        )


def _ocr_empty_pages(
    pdf_path: Path, pages: list[PageText], api_key: str = ""
) -> list[PageText]:
    """Run vision OCR on pages that pdfplumber couldn't extract text from."""
    from ..llm.ocr import ocr_pdf_pages

    empty_page_nums = [
        p.page_number for p in pages if len(p.text.strip()) < _EMPTY_PAGE_THRESHOLD
    ]

    if not empty_page_nums:
        return pages

    ocr_results = ocr_pdf_pages(pdf_path, empty_page_nums, api_key=api_key)

    updated: list[PageText] = []
    for page in pages:
        if page.page_number in ocr_results:
            updated.append(
                PageText(page_number=page.page_number, text=ocr_results[page.page_number])
            )
        else:
            updated.append(page)

    ocr_count = sum(1 for pn in empty_page_nums if pn in ocr_results)
    logger.info(f"OCR recovered {ocr_count} pages for {pdf_path.name}")
    return updated
