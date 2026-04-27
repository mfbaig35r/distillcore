"""Section-aware document chunker for the pipeline.

Strategies (auto-selected based on document content):
  1. Transcripts  -> group consecutive turns (~target_tokens per chunk)
  2. Sectioned docs -> one chunk per section, split large sections on paragraphs
  3. Fallback       -> split full_text on paragraph boundaries

Named strategies ("paragraph", "sentence", "fixed", "llm") delegate to
the public chunk() API in distillcore.chunking.
"""

from __future__ import annotations

from ..chunking import estimate_tokens as _estimate_tokens_raw
from ..chunking import split_paragraphs
from ..config import ChunkConfig
from ..models import ChunkedDocument, Document, DocumentChunk

# -- Token estimation (wraps the shared helper with ChunkConfig) ---------------


def _estimate_tokens(text: str, config: ChunkConfig) -> int:
    """Estimate token count using config.tokenizer or len//4 fallback."""
    return _estimate_tokens_raw(text, config.tokenizer)


# -- Main entry point ----------------------------------------------------------


def chunk_document(
    doc: Document,
    config: ChunkConfig | None = None,
) -> ChunkedDocument:
    """Chunk a Document using the best strategy for its type."""
    if config is None:
        config = ChunkConfig()

    target_chars = config.target_tokens * 4
    max_chars = config.max_tokens * 4
    overlap = config.overlap_chars
    strategy = config.strategy

    if strategy == "auto":
        # Auto-selection: transcript > sections > fallback
        turn_chars = (
            sum(len(t.content) for t in doc.transcript_turns)
            if doc.transcript_turns
            else 0
        )
        full_chars = len(doc.full_text) if doc.full_text else 0
        turn_coverage = turn_chars / full_chars if full_chars > 0 else 0

        if doc.transcript_turns and turn_coverage > 0.5:
            raw_chunks = chunk_transcript(doc, target_chars)
        elif doc.sections:
            raw_chunks = chunk_sections(doc, target_chars, overlap, max_chars)
        else:
            raw_chunks = chunk_fallback(doc, target_chars, overlap, max_chars)
    else:
        # Named strategy: delegate to the public chunk() API
        from ..chunking import chunk

        text = doc.full_text or ""
        text_chunks = chunk(
            text,
            strategy=strategy,
            target_tokens=config.target_tokens,
            max_tokens=config.max_tokens,
            overlap_tokens=max(config.overlap_chars // 4, 1),
            tokenizer=config.tokenizer,
        )
        raw_chunks = [{"text": t, "section_type": "chunk"} for t in text_chunks]

    chunks = [
        DocumentChunk(
            chunk_index=i,
            text=rc["text"],
            token_estimate=_estimate_tokens(rc["text"], config),
            section_type=rc.get("section_type"),
            section_heading=rc.get("section_heading"),
            page_start=rc.get("page_start"),
            page_end=rc.get("page_end"),
            speakers=rc.get("speakers"),
        )
        for i, rc in enumerate(raw_chunks)
    ]

    # Merge small chunks
    if config.min_tokens > 0:
        chunks = _merge_small_chunks(chunks, config)
        for i, c in enumerate(chunks):
            c.chunk_index = i

    return ChunkedDocument(chunk_count=len(chunks), chunks=chunks)


# -- Small chunk merging -------------------------------------------------------


def _merge_small_chunks(
    chunks: list[DocumentChunk], config: ChunkConfig
) -> list[DocumentChunk]:
    """Merge chunks below min_tokens into their neighbors."""
    if not chunks or config.min_tokens <= 0:
        return chunks

    merged: list[DocumentChunk] = []
    for chunk in chunks:
        if chunk.token_estimate >= config.min_tokens or not merged:
            merged.append(chunk.model_copy())
        else:
            prev = merged[-1]
            prev.text = prev.text + "\n\n" + chunk.text
            prev.token_estimate = _estimate_tokens(prev.text, config)
            if chunk.page_end is not None:
                prev.page_end = chunk.page_end
            if chunk.speakers and prev.speakers:
                prev.speakers = sorted(set(prev.speakers + chunk.speakers))

    return merged


# -- Transcript chunking -------------------------------------------------------


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


# -- Section-based chunking ----------------------------------------------------


def chunk_sections(
    doc: Document, target_chars: int, overlap: int, max_chars: int | None = None
) -> list[dict]:
    """One chunk per section; split large sections on paragraph boundaries."""
    if max_chars is None:
        max_chars = target_chars * 2
    chunks: list[dict] = []
    for section in doc.sections:
        _chunk_one_section(
            section,
            parent_heading=None,
            chunks=chunks,
            target_chars=target_chars,
            overlap=overlap,
            max_chars=max_chars,
        )
    return chunks


def _chunk_one_section(
    section,
    parent_heading: str | None,
    chunks: list[dict],
    target_chars: int,
    overlap: int,
    max_chars: int | None = None,
) -> None:
    """Recursively chunk a section and its subsections."""
    if max_chars is None:
        max_chars = target_chars * 2
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
            para_chunks = split_paragraphs(
                content, display_heading, target_chars, overlap, max_chars
            )
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
            max_chars=max_chars,
        )


# -- Fallback text chunking ----------------------------------------------------


def chunk_fallback(
    doc: Document, target_chars: int, overlap: int, max_chars: int | None = None
) -> list[dict]:
    """Split full_text on paragraph boundaries when no sections exist."""
    if max_chars is None:
        max_chars = target_chars * 2
    text = doc.full_text.strip()
    if not text:
        return []

    para_chunks = split_paragraphs(
        text, heading=None, target_chars=target_chars, overlap=overlap, max_chars=max_chars
    )
    return [{"text": pc, "section_type": "full_text"} for pc in para_chunks]
