"""Plain text extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import ExtractionResult, PageText


class TextExtractor:
    """Extract text from plain text files (.txt, .md, .markdown)."""

    formats = ["txt", "text", "md", "markdown"]

    def extract(self, source: Path | str, config: Any = None) -> ExtractionResult:
        text = Path(source).read_text(encoding="utf-8", errors="replace")
        return ExtractionResult(
            pages=[PageText(page_number=1, text=text)],
            full_text=text,
            page_count=1,
            format="txt",
        )
