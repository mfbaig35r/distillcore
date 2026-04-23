"""Section-aware document chunker.

Strategies:
  1. Transcripts  -> group consecutive turns (~target_tokens per chunk)
  2. Sectioned docs -> one chunk per section, split large sections on paragraphs
  3. Fallback       -> split full_text on paragraph boundaries
"""

from __future__ import annotations

import re

from ..config import ChunkConfig
from ..models import ChunkedDocument, Document, DocumentChunk


def chunk_document(
    doc: Document,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    """Chunk a Document using the best strategy for its type."""
    if config is None:
        config = ChunkConfig()

    target_chars = config.target_tokens * 4
    overlap = config.overlap_chars

    # For transcripts: only use turn-based chunking if the turns captured
    # enough of the full text.
    turn_chars = sum(len(t.content) for t in doc.transcript_turns) if doc.transcript_turns else 0
    full_chars = len(doc.full_text) if doc.full_text else 0
    turn_coverage = turn_chars / full_chars if full_chars > 0 else 0

    if doc.transcript_turns and turn_coverage > 0.5:
        raw_chunks = chunk_transcript(doc, target_chars)
    elif doc.sections:
        raw_chunks = chunk_sections(doc, target_chars, overlap)
    else:
        raw_chunks = chunk_fallback(doc, target_chars, overlap)

    chunks = [
        DocumentChunk(
            chunk_index=i,
            text=rc["text"],
            token_estimate=len(rc["text"]) // 4,
            section_type=rc.get("section_type"),
            section_heading=rc.get("section_heading"),
            page_start=rc.get("page_start"),
            page_end=rc.get("page_end"),
            speakers=rc.get("speakers"),
        )
        for i, rc in enumerate(raw_chunks)
    ]

    return ChunkedDocument(chunk_count=len(chunks), chunks=chunks)


# -- Transcript chunking ---


def chunk_transcript(doc: Document, target_chars: int) -> list[dict]:
    """Group consecutive transcript turns into chunks of ~target_chars."""
    turns = doc.transcript_turns
    chunks: list[dict] = []
    buf_turns: list = []
    buf_chars = 0

    for turn in turns:
        turn_text = f"{turn.speaker}: {turn.content}"
        turn_len = len(turn_text)

        if buf_turns and buf_chars + turn_len > target_chars:
            chunks.append(_finalize_transcript_chunk(buf_turns))
            buf_turns = []
            buf_chars = 0

        buf_turns.append(turn)
        buf_chars += turn_len

    if buf_turns:
        chunks.append(_finalize_transcript_chunk(buf_turns))

    return chunks


def _finalize_transcript_chunk(turns: list) -> dict:
    """Build a raw chunk dict from a group of transcript turns."""
    speakers = sorted(set(t.speaker for t in turns))
    pages = [t.page for t in turns if t.page is not None]
    page_lo = min(pages) if pages else None
    page_hi = max(pages) if pages else None

    header_parts = []
    if page_lo is not None:
        page_str = f"p.{page_lo}" if page_lo == page_hi else f"p.{page_lo}-{page_hi}"
        header_parts.append(f"Transcript {page_str}")
    speaker_summary = ", ".join(speakers[:3])
    if len(speakers) > 3:
        speaker_summary += f" +{len(speakers) - 3} others"
    header_parts.append(speaker_summary)
    header = " — ".join(header_parts)

    body = "\n\n".join(f"{t.speaker}: {t.content}" for t in turns)
    text = f"{header}\n\n{body}"

    return {
        "text": text,
        "section_type": "transcript",
        "section_heading": header,
        "page_start": page_lo,
        "page_end": page_hi,
        "speakers": speakers,
    }


# -- Section-based chunking ---


def chunk_sections(doc: Document, target_chars: int, overlap: int) -> list[dict]:
    """One chunk per section; split large sections on paragraph boundaries."""
    chunks: list[dict] = []
    for section in doc.sections:
        _chunk_one_section(
            section,
            parent_heading=None,
            chunks=chunks,
            target_chars=target_chars,
            overlap=overlap,
        )
    return chunks


def _chunk_one_section(
    section,
    parent_heading: str | None,
    chunks: list[dict],
    target_chars: int,
    overlap: int,
) -> None:
    """Recursively chunk a section and its subsections."""
    heading = section.heading
    display_heading = heading
    if parent_heading and heading:
        display_heading = f"{parent_heading} > {heading}"
    elif parent_heading:
        display_heading = parent_heading

    content = section.content.strip()
    if content:
        if len(content) <= target_chars:
            text = f"{display_heading}\n\n{content}" if display_heading else content
            chunks.append(
                {
                    "text": text,
                    "section_type": section.section_type,
                    "section_heading": display_heading,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                }
            )
        else:
            para_chunks = split_paragraphs(content, display_heading, target_chars, overlap)
            for pc in para_chunks:
                chunks.append(
                    {
                        "text": pc,
                        "section_type": section.section_type,
                        "section_heading": display_heading,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    }
                )

    for sub in section.subsections:
        _chunk_one_section(
            sub,
            parent_heading=display_heading,
            chunks=chunks,
            target_chars=target_chars,
            overlap=overlap,
        )


# -- Fallback text chunking ---


def chunk_fallback(doc: Document, target_chars: int, overlap: int) -> list[dict]:
    """Split full_text on paragraph boundaries when no sections exist."""
    text = doc.full_text.strip()
    if not text:
        return []

    para_chunks = split_paragraphs(text, heading=None, target_chars=target_chars, overlap=overlap)
    return [{"text": pc, "section_type": "full_text"} for pc in para_chunks]


# -- Shared paragraph splitter ---


def split_paragraphs(
    text: str,
    heading: str | None,
    target_chars: int,
    overlap: int,
) -> list[str]:
    """Split text on paragraph boundaries at ~target_chars with overlap."""
    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if buf and buf_len + len(para) > target_chars:
            chunk_text = "\n\n".join(buf)
            if heading:
                chunk_text = f"{heading}\n\n{chunk_text}"
            result.append(chunk_text)

            # Overlap: carry trailing text forward
            overlap_buf: list[str] = []
            overlap_len = 0
            for p in reversed(buf):
                if overlap_len + len(p) > overlap:
                    break
                overlap_buf.insert(0, p)
                overlap_len += len(p)
            buf = overlap_buf
            buf_len = overlap_len

        buf.append(para)
        buf_len += len(para)

    if buf:
        chunk_text = "\n\n".join(buf)
        if heading:
            chunk_text = f"{heading}\n\n{chunk_text}"
        result.append(chunk_text)

    return result if result else [f"{heading}\n\n{text}" if heading else text]
