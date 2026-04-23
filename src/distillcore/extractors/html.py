"""HTML text extraction using BeautifulSoup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from ..models import ExtractionResult, PageText

# Tags that are noise for document processing
_STRIP_TAGS = ["script", "style", "nav", "footer", "header", "noscript"]


class HtmlExtractor:
    """Extract text from HTML files, stripping tags and preserving block structure."""

    formats = ["html", "htm"]

    def extract(self, source: Path | str, config: Any = None) -> ExtractionResult:
        """Extract text from an HTML file.

        Strips script/style/nav/footer/header tags, then extracts text
        with double-newline separation for block-level structure.
        """
        raw = Path(source).read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        # Remove noise tags
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        # Extract text with block-level separation
        text = soup.get_text(separator="\n\n")

        # Collapse excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        # Extract metadata
        metadata: dict[str, str] = {}
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta and author_meta.get("content"):
            metadata["author"] = str(author_meta["content"]).strip()

        return ExtractionResult(
            pages=[PageText(page_number=1, text=text)],
            full_text=text,
            page_count=1,
            format="html",
            metadata=metadata,
        )
