"""Extractor registry — auto-detects format and dispatches to the right extractor."""

from __future__ import annotations

from pathlib import Path

from ..models import ExtractionResult
from .base import Extractor

_registry: dict[str, Extractor] = {}


def register_extractor(extractor: Extractor) -> None:
    """Register an extractor for its declared formats."""
    for fmt in extractor.formats:
        _registry[fmt.lower()] = extractor


def get_registered_formats() -> list[str]:
    """Return list of registered format extensions."""
    return sorted(_registry.keys())


def extract(source: Path | str, format: str | None = None, **kwargs: object) -> ExtractionResult:
    """Extract text from a file. Auto-detects format from extension if not given."""
    source = Path(source)
    if format is None:
        format = _detect_format(source)
    ext = format.lower().lstrip(".")
    if ext not in _registry:
        available = ", ".join(get_registered_formats()) or "(none)"
        raise ValueError(
            f"No extractor registered for format '{ext}'. "
            f"Available: {available}. "
            f"Install extras: pip install distillcore[pdf]"
        )
    return _registry[ext].extract(source, **kwargs)


def _detect_format(path: Path) -> str:
    """Detect format from file extension."""
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "txt"


# Auto-register built-in extractors
from .text import TextExtractor  # noqa: E402

register_extractor(TextExtractor())

try:
    from .pdf import PdfExtractor  # noqa: E402

    register_extractor(PdfExtractor())
except ImportError:
    pass  # pdfplumber not installed
