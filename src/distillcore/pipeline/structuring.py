"""Document structuring via LLM — breaks documents into hierarchical sections."""

from __future__ import annotations

import logging

from ..config import DistillConfig
from ..llm.client import get_client
from ..llm.json_repair import safe_parse
from ..models import Section, TranscriptTurn

logger = logging.getLogger(__name__)


def structure_document(
    full_text: str,
    document_type: str,
    filename: str,
    config: DistillConfig,
    pages_text: list[str] | None = None,
    is_transcript: bool = False,
) -> dict:
    """Send document text to LLM for structured decomposition.

    Falls back to empty sections on failure — never crashes the pipeline.
    """
    is_large = len(full_text) > config.large_doc_char_threshold

    try:
        if is_transcript and pages_text and config.domain.transcript_prompt:
            return _structure_transcript_chunked(pages_text, filename, config)
        if is_large and pages_text:
            return _structure_large_document(pages_text, document_type, filename, config)
        return _structure_single(full_text, document_type, filename, config)
    except Exception as e:
        logger.error(f"Structuring failed for {filename}: {e}")
        return {"sections": []}


def _structure_single(
    full_text: str, document_type: str, filename: str, config: DistillConfig
) -> dict:
    """Single-call structuring for non-transcript documents."""
    prompt = config.domain.structuring_prompt
    if not prompt:
        return {"sections": []}

    client = get_client(config.resolve_api_key())

    user_msg = (
        f"Document: {filename}\n"
        f"Type: {document_type}\n\n"
        f"--- FULL TEXT ---\n{full_text[:100_000]}"
    )

    response = client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        max_tokens=config.max_tokens,
        temperature=0,
    )

    raw = response.choices[0].message.content or "{}"
    return safe_parse(raw)


def _structure_large_document(
    pages_text: list[str],
    document_type: str,
    filename: str,
    config: DistillConfig,
) -> dict:
    """Process large documents in overlapping page windows."""
    prompt = config.domain.structuring_prompt
    if not prompt:
        return {"sections": []}

    client = get_client(config.resolve_api_key())
    window_size = config.llm_page_window_size
    overlap = config.llm_page_window_overlap

    all_sections: list[dict] = []

    step = max(1, window_size - overlap)
    for chunk_start in range(0, len(pages_text), step):
        chunk_end = min(chunk_start + window_size, len(pages_text))
        page_start = chunk_start + 1
        page_end = chunk_end

        chunk_text = "\n\n".join(
            f"--- PAGE {chunk_start + i + 1} ---\n{pages_text[chunk_start + i]}"
            for i in range(chunk_end - chunk_start)
        )

        user_msg = (
            f"Document: {filename}\n"
            f"Type: {document_type}\n"
            f"Pages: {page_start}-{page_end}\n\n"
            f"--- TEXT ---\n{chunk_text}"
        )

        logger.info(f"Structuring window pages {page_start}-{page_end}")

        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=config.max_tokens,
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        chunk_result = safe_parse(raw)
        all_sections.extend(chunk_result.get("sections", []))

        if chunk_end >= len(pages_text):
            break

    return {"sections": all_sections}


def _structure_transcript_chunked(
    pages_text: list[str],
    filename: str,
    config: DistillConfig,
) -> dict:
    """Process transcript in page chunks, then merge."""
    prompt = config.domain.transcript_prompt
    if not prompt:
        return {"sections": [], "transcript_turns": []}

    client = get_client(config.resolve_api_key())
    window_size = config.llm_page_window_size
    overlap = config.llm_page_window_overlap

    all_turns: list[dict] = []
    all_sections: list[dict] = []

    step = max(1, window_size - overlap)
    for chunk_start in range(0, len(pages_text), step):
        chunk_end = min(chunk_start + window_size, len(pages_text))
        page_start = chunk_start + 1
        page_end = chunk_end

        chunk_text = "\n\n".join(
            f"--- PAGE {chunk_start + i + 1} ---\n{pages_text[chunk_start + i]}"
            for i in range(chunk_end - chunk_start)
        )

        system = prompt.format(start=page_start, end=page_end)
        user_msg = f"Document: {filename}\n\n{chunk_text}"

        logger.info(f"Processing transcript chunk pages {page_start}-{page_end}")

        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=config.max_tokens,
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        chunk_result = safe_parse(raw)
        all_turns.extend(chunk_result.get("transcript_turns", []))
        all_sections.extend(chunk_result.get("sections", []))

        if chunk_end >= len(pages_text):
            break

    return {"transcript_turns": all_turns, "sections": all_sections}


def parse_structure_result(
    result: dict,
    pages_text: list[str] | None = None,
) -> tuple[list[Section], list[TranscriptTurn]]:
    """Parse the LLM JSON response into typed models.

    If pages_text is provided, section content is populated by slicing the
    original page text using page_range boundaries — no need for the LLM
    to reproduce content verbatim.
    """
    sections = [_parse_section(s) for s in result.get("sections", [])]

    if pages_text:
        for section in sections:
            _populate_section_content(section, pages_text)

    transcript_turns = []
    for t in result.get("transcript_turns", []):
        transcript_turns.append(
            TranscriptTurn(
                speaker=t.get("speaker", "Unknown"),
                role=t.get("role", "unknown"),
                content=t.get("content", ""),
                page=t.get("page"),
                line_start=t.get("line_start"),
            )
        )

    return sections, transcript_turns


def _populate_section_content(
    section: Section,
    pages_text: list[str],
) -> None:
    """Fill section.content by slicing pages_text using page boundaries.

    pages_text is 0-indexed; page_start/page_end are 1-indexed.
    Only overwrites content if it's empty (preserves any LLM-provided content
    for backward compatibility with custom presets).
    """
    if (
        not section.content
        and section.page_start is not None
        and section.page_end is not None
    ):
        start_idx = section.page_start - 1
        end_idx = section.page_end
        if 0 <= start_idx < len(pages_text) and end_idx <= len(pages_text):
            section.content = "\n\n".join(pages_text[start_idx:end_idx])
        else:
            logger.warning(
                f"Section '{section.heading}' page_range [{section.page_start}, "
                f"{section.page_end}] out of bounds (document has {len(pages_text)} pages)"
            )

    for sub in section.subsections:
        _populate_section_content(sub, pages_text)


def _parse_section(data: dict) -> Section:
    """Recursively parse a section dict into a Section model."""
    subsections = [_parse_section(s) for s in data.get("subsections", [])]

    page_start = None
    page_end = None
    page_range = data.get("page_range")
    if page_range and isinstance(page_range, (list, tuple)) and len(page_range) == 2:
        page_start = page_range[0]
        page_end = page_range[1]

    return Section(
        heading=data.get("heading"),
        section_type=data.get("section_type", "general"),
        content=data.get("content", ""),
        subsections=subsections,
        page_start=page_start,
        page_end=page_end,
    )
