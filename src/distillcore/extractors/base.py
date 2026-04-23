"""Extractor protocol definition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..models import ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    """Protocol for document extractors."""

    formats: list[str]

    def extract(self, source: Path | str, config: Any = None) -> ExtractionResult: ...
