"""DOCX text extraction using python-docx."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import docx

from ..models import ExtractionResult, PageText

logger = logging.getLogger(__name__)


class DocxExtractor:
    """Extract text from .docx files with paragraph structure and table content."""

    formats = ["docx"]

    def extract(self, source: Path | str, config: Any = None) -> ExtractionResult:
        """Extract text from a DOCX file.

        Paragraphs are joined with double newlines to preserve block structure.
        Tables are extracted row-by-row with tab-separated cells.
        Core properties (title, author, created) are captured in metadata.
        """
        doc = docx.Document(str(source))

        parts: list[str] = []
        for element in doc.element.body:
            tag = element.tag.split("}")[-1]  # strip namespace
            if tag == "p":
                text = element.text or ""
                if text.strip():
                    parts.append(text.strip())
            elif tag == "tbl":
                table_text = _extract_table(element, doc)
                if table_text:
                    parts.append(table_text)

        full_text = "\n\n".join(parts)

        # Extract core properties
        metadata: dict[str, str] = {}
        props = doc.core_properties
        if props.title:
            metadata["title"] = props.title
        if props.author:
            metadata["author"] = props.author
        if props.subject:
            metadata["subject"] = props.subject
        if props.created:
            metadata["created"] = props.created.isoformat()

        return ExtractionResult(
            pages=[PageText(page_number=1, text=full_text)],
            full_text=full_text,
            page_count=1,
            format="docx",
            metadata=metadata,
        )


def _extract_table(tbl_element: Any, doc: docx.Document) -> str:
    """Extract text from a table element, row-by-row with tab separation."""
    from docx.table import Table

    try:
        table = Table(tbl_element, doc)
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("\t".join(cells))
        return "\n".join(rows)
    except Exception:
        return ""
