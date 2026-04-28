"""Async document structuring via LLM — with parallel page windows."""

from __future__ import annotations

import asyncio
import logging

from ..config import DistillConfig
from ..llm.async_client import get_async_client
from ..llm.json_repair import safe_parse

logger = logging.getLogger(__name__)


async def structure_document_async(
    full_text: str,
    document_type: str,
    filename: str,
    config: DistillConfig,
    pages_text: list[str] | None = None,
    is_transcript: bool = False,
) -> dict:
    """Async version of structure_document.

    Large documents benefit from parallel page-window processing.
    Falls back to empty sections on failure.
    """
    is_large = len(full_text) > config.large_doc_char_threshold

    try:
        if is_transcript and pages_text and config.domain.transcript_prompt:
            return await _structure_transcript_async(pages_text, filename, config)
        if is_large and pages_text:
            return await _structure_large_document_async(
                pages_text, document_type, filename, config
            )
        return await _structure_single_async(full_text, document_type, filename, config)
    except Exception as e:
        logger.error(f"Structuring failed for {filename}: {e}")
        return {"sections": [], "_structuring_error": str(e)}


async def _structure_single_async(
    full_text: str, document_type: str, filename: str, config: DistillConfig
) -> dict:
    """Single-call structuring for non-transcript documents."""
    prompt = config.domain.structuring_prompt
    if not prompt:
        return {
            "sections": [],
            "_structuring_error": "No structuring prompt configured for this domain",
        }

    client = get_async_client(config.resolve_api_key())

    user_msg = (
        f"Document: {filename}\n"
        f"Type: {document_type}\n\n"
        f"--- FULL TEXT ---\n{full_text[:100_000]}"
    )

    response = await client.chat.completions.create(
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


async def _structure_large_document_async(
    pages_text: list[str],
    document_type: str,
    filename: str,
    config: DistillConfig,
) -> dict:
    """Process large documents with parallel page windows."""
    prompt = config.domain.structuring_prompt
    if not prompt:
        return {
            "sections": [],
            "_structuring_error": "No structuring prompt configured for this domain",
        }

    window_size = config.llm_page_window_size
    overlap = config.llm_page_window_overlap
    step = max(1, window_size - overlap)

    async def _process_window(chunk_start: int) -> dict:
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
        client = get_async_client(config.resolve_api_key())

        response = await client.chat.completions.create(
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

    # Build window starts
    starts = list(range(0, len(pages_text), step))
    # Parallel LLM calls
    window_results = await asyncio.gather(*[_process_window(s) for s in starts])

    all_sections: list[dict] = []
    for wr in window_results:
        all_sections.extend(wr.get("sections", []))

    return {"sections": all_sections}


async def _structure_transcript_async(
    pages_text: list[str],
    filename: str,
    config: DistillConfig,
) -> dict:
    """Process transcript with parallel page windows."""
    prompt = config.domain.transcript_prompt
    if not prompt:
        return {"sections": [], "transcript_turns": []}

    window_size = config.llm_page_window_size
    overlap = config.llm_page_window_overlap
    step = max(1, window_size - overlap)

    async def _process_window(chunk_start: int) -> dict:
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
        client = get_async_client(config.resolve_api_key())

        response = await client.chat.completions.create(
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
        return safe_parse(raw)

    starts = list(range(0, len(pages_text), step))
    window_results = await asyncio.gather(*[_process_window(s) for s in starts])

    all_turns: list[dict] = []
    all_sections: list[dict] = []
    for wr in window_results:
        all_turns.extend(wr.get("transcript_turns", []))
        all_sections.extend(wr.get("sections", []))

    return {"transcript_turns": all_turns, "sections": all_sections}
