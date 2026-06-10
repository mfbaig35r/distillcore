"""CSV / delimited-text extractor (stdlib csv only, no extras needed)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..models import ExtractionResult, PageText

# Delimiters Sniffer is allowed to detect. Pipe is included for log-style data.
_SNIFF_DELIMITERS = ",\t|;"
# Bytes of file content used for delimiter detection.
_SNIFF_SIZE = 4096


class CsvExtractor:
    """Extract tabular text from CSV / TSV / pipe-separated files.

    Output format:
        - Header row preserved as the first line.
        - Cells joined by tab within a row (independent of source delimiter).
        - Rows joined by newline. One ``PageText`` (CSVs have no native pagination).

    Metadata populated:
        - ``columns`` — list of header column names
        - ``row_count`` — number of data rows (excluding header)
        - ``delimiter`` — detected source delimiter
    """

    formats = ["csv", "tsv"]

    def extract(self, source: Path | str, config: Any = None) -> ExtractionResult:
        path = Path(source)
        ext = path.suffix.lower().lstrip(".")
        raw = path.read_text(encoding="utf-8", errors="replace")

        if not raw.strip():
            return ExtractionResult(
                pages=[],
                full_text="",
                page_count=0,
                format=ext or "csv",
                metadata={"columns": [], "row_count": 0, "delimiter": ""},
            )

        delimiter = _detect_delimiter(raw, ext)

        reader = csv.reader(raw.splitlines(), delimiter=delimiter)
        rows = [row for row in reader if row]

        if not rows:
            return ExtractionResult(
                pages=[],
                full_text="",
                page_count=0,
                format=ext or "csv",
                metadata={"columns": [], "row_count": 0, "delimiter": delimiter},
            )

        header, *data_rows = rows
        # Tab-normalize so downstream chunking treats cells uniformly regardless
        # of whether the source was comma- or pipe-separated.
        text = "\n".join("\t".join(row) for row in rows)

        return ExtractionResult(
            pages=[PageText(page_number=1, text=text)],
            full_text=text,
            page_count=1,
            format=ext or "csv",
            metadata={
                "columns": header,
                "row_count": len(data_rows),
                "delimiter": delimiter,
            },
        )


def _detect_delimiter(raw: str, ext: str) -> str:
    """Detect delimiter via csv.Sniffer with a sensible fallback per extension."""
    sample = raw[:_SNIFF_SIZE]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=_SNIFF_DELIMITERS)
        return dialect.delimiter
    except csv.Error:
        # Sniffer fails on single-column files or odd quoting. Fall back to ext.
        return "\t" if ext == "tsv" else ","
