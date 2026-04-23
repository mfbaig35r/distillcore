"""Extractor registry — auto-detects format and dispatches to the right extractor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def extract(
    source: Path | str, format: str | None = None, config: Any = None
) -> ExtractionResult:
    """Extract text from a file. Auto-detects format from extension if not given.

    If config.allowed_dirs is set, validates the path is within an allowed directory.
    """
    source = Path(source)
    allowed_dirs = getattr(config, "allowed_dirs", None) if config else None
    source = _validate_path(source, allowed_dirs)
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
    return _registry[ext].extract(source, config=config)


def _validate_path(source: Path, allowed_dirs: list[str] | None) -> Path:
    """Resolve path and check it's within allowed directories.

    Returns the resolved path. Raises PermissionError if outside allowed dirs.
    If allowed_dirs is None, all paths are permitted (library default).
    """
    resolved = source.resolve()
    if allowed_dirs is None:
        return resolved
    for allowed in allowed_dirs:
        allowed_path = Path(allowed).expanduser().resolve()
        try:
            resolved.relative_to(allowed_path)
            return resolved
        except ValueError:
            continue
    raise PermissionError(
        f"Access denied: {resolved} is not within allowed directories: {allowed_dirs}"
    )


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

try:
    from .docx import DocxExtractor  # noqa: E402

    register_extractor(DocxExtractor())
except ImportError:
    pass  # python-docx not installed

try:
    from .html import HtmlExtractor  # noqa: E402

    register_extractor(HtmlExtractor())
except ImportError:
    pass  # beautifulsoup4 not installed
