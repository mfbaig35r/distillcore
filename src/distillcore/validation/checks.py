"""Pipeline stage validation checks."""

from __future__ import annotations

from ..models import ChunkedDocument, Document, Section, ValidationReport
from .coverage import compute_coverage, find_missing_segments


def validate_extraction(full_text: str, pages_text: list[str]) -> None:
    """Assert the extraction join is lossless."""
    expected = "\n\n".join(pages_text)
    assert full_text == expected, "Extraction join is not lossless"


def _collect_section_content(sections: list[Section]) -> str:
    """Recursively collect all section content."""
    parts: list[str] = []
    for section in sections:
        if section.content:
            parts.append(section.content)
        if section.subsections:
            parts.append(_collect_section_content(section.subsections))
    return "\n".join(parts)


def validate_structuring(doc: Document, threshold: float = 0.95) -> ValidationReport:
    """Validate that structuring preserved the original text."""
    section_content = _collect_section_content(doc.sections)
    turn_content = "\n".join(t.content for t in doc.transcript_turns)
    structured_text = f"{section_content}\n{turn_content}".strip()

    coverage = compute_coverage(doc.full_text, structured_text)
    missing = find_missing_segments(doc.full_text, structured_text)

    warnings: list[str] = []
    if coverage < threshold:
        warnings.append(
            f"Structuring coverage {coverage:.1%} is below threshold {threshold:.1%}"
        )

    return ValidationReport(
        structuring_coverage=coverage,
        missing_segments=missing,
        warnings=warnings,
        passed=coverage >= threshold,
    )


def validate_chunking(
    doc: Document,
    chunked: ChunkedDocument,
    threshold: float = 0.98,
) -> ValidationReport:
    """Validate that chunking preserved structured content."""
    section_content = _collect_section_content(doc.sections)
    turn_content = "\n".join(t.content for t in doc.transcript_turns)
    structured_text = f"{section_content}\n{turn_content}".strip()

    chunk_text = "\n".join(c.text for c in chunked.chunks)
    coverage = compute_coverage(structured_text, chunk_text)

    warnings: list[str] = []
    if coverage < threshold:
        warnings.append(
            f"Chunking coverage {coverage:.1%} is below threshold {threshold:.1%}"
        )

    empty_indices = [c.chunk_index for c in chunked.chunks if not c.text.strip()]
    if empty_indices:
        warnings.append(f"Empty chunks found at indices: {empty_indices}")

    return ValidationReport(
        chunking_coverage=coverage,
        warnings=warnings,
        passed=coverage >= threshold,
    )


def validate_end_to_end(
    full_text: str,
    chunked: ChunkedDocument,
    threshold: float = 0.93,
) -> ValidationReport:
    """Validate end-to-end coverage from raw text to final chunks."""
    chunk_text = "\n".join(c.text for c in chunked.chunks)
    coverage = compute_coverage(full_text, chunk_text)

    warnings: list[str] = []
    if coverage < threshold:
        warnings.append(
            f"End-to-end coverage {coverage:.1%} is below threshold {threshold:.1%}"
        )

    return ValidationReport(
        end_to_end_coverage=coverage,
        warnings=warnings,
        passed=coverage >= threshold,
    )
