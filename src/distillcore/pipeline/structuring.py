"""Document structuring via LLM — breaks documents into hierarchical sections."""

from __future__ import annotations

import logging

from ..config import DistillConfig
from ..llm.client import get_client
from ..llm.json_repair import safe_parse

# Re-export shared helpers for backward compatibility
from ._shared import (  # noqa: F401
    _populate_section_content,
    parse_structure_result,
)

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
        return {"sections": [], "_structuring_error": str(e)}


def _structure_single(
    full_text: str, document_type: str, filename: str, config: DistillConfig
) -> dict:
    """Single-call structuring for non-transcript documents."""
    prompt = config.domain.structuring_prompt
    if not prompt:
        return {
            "sections": [],
            "_structuring_error": "No structuring prompt configured for this domain",
        }

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
        return {
            "sections": [],
            "_structuring_error": "No structuring prompt configured for this domain",
        }

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
