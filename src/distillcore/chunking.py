"""Standalone chunking API for distillcore.

Usage::

    from distillcore import chunk

    chunks = chunk("your text here")
    chunks = chunk(text, strategy="sentence", target_tokens=300)
    chunks = chunk(text, strategy="llm", api_key="sk-...")

Strategies:
  - "paragraph" (default): split on paragraph boundaries, subsplit oversized blocks
  - "sentence": split on sentence boundaries, group into target-sized chunks
  - "fixed": pure sliding window at word boundaries
  - "llm": LLM-driven semantic chunking (requires API key)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable

logger = logging.getLogger(__name__)

# -- Sentence boundary regex ---------------------------------------------------

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# -- Token estimation ----------------------------------------------------------

_DEFAULT_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    """Estimate token count for a text string.

    Args:
        text: The text to estimate tokens for.
        tokenizer: Optional callable that counts tokens. If None, uses len // 4.
    """
    if tokenizer is not None:
        return tokenizer(text)
    return len(text) // _DEFAULT_CHARS_PER_TOKEN


def _tokens_to_chars(tokens: int) -> int:
    """Convert token count to approximate character count."""
    return tokens * _DEFAULT_CHARS_PER_TOKEN


# -- Public API ----------------------------------------------------------------

_STRATEGIES = ("paragraph", "sentence", "fixed", "llm")


def chunk(
    text: str,
    *,
    strategy: str = "paragraph",
    target_tokens: int = 500,
    max_tokens: int = 1000,
    overlap_tokens: int = 50,
    min_tokens: int = 0,
    tokenizer: Callable[[str], int] | None = None,
    # LLM-specific
    api_key: str = "",
    model: str = "gpt-4o",
) -> list[str]:
    """Split text into chunks using the specified strategy.

    Args:
        text: The text to chunk.
        strategy: Chunking strategy. One of "paragraph", "sentence", "fixed", "llm".
        target_tokens: Target chunk size in tokens. Default 500.
        max_tokens: Maximum chunk size in tokens. Default 1000.
        overlap_tokens: Token overlap between consecutive chunks. Default 50.
        min_tokens: Minimum tokens per chunk; smaller chunks merge into neighbors.
            Default 0 (disabled).
        tokenizer: Optional callable that counts tokens in a string. If None, uses
            len(text) // 4 as an approximation.
        api_key: OpenAI API key (required for strategy="llm"). Falls back to
            OPENAI_API_KEY env var.
        model: OpenAI model name for LLM chunking. Default "gpt-4o".

    Returns:
        List of text chunks.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if not text or not text.strip():
        return []

    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from: {', '.join(_STRATEGIES)}"
        )

    target_chars = _tokens_to_chars(target_tokens)
    max_chars = _tokens_to_chars(max_tokens)
    overlap_chars = _tokens_to_chars(overlap_tokens)

    if strategy == "paragraph":
        result = _chunk_paragraph(text, target_chars, max_chars, overlap_chars)
    elif strategy == "sentence":
        result = _chunk_sentence(text, target_chars, max_chars)
    elif strategy == "fixed":
        result = _chunk_fixed(text, max_chars, overlap_chars)
    elif strategy == "llm":
        result = _chunk_llm(text, target_tokens, max_tokens, api_key, model)

    # Merge small chunks
    if min_tokens > 0 and len(result) > 1:
        result = _merge_small(result, min_tokens, tokenizer)

    return result


async def achunk(
    text: str,
    *,
    strategy: str = "paragraph",
    target_tokens: int = 500,
    max_tokens: int = 1000,
    overlap_tokens: int = 50,
    min_tokens: int = 0,
    tokenizer: Callable[[str], int] | None = None,
    api_key: str = "",
    model: str = "gpt-4o",
) -> list[str]:
    """Async version of chunk(). Only strategy="llm" is truly async."""
    if not text or not text.strip():
        return []

    if strategy not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from: {', '.join(_STRATEGIES)}"
        )

    if strategy == "llm":
        result = await _achunk_llm(text, target_tokens, max_tokens, api_key, model)
    else:
        # Non-LLM strategies are synchronous — just delegate
        result = chunk(
            text,
            strategy=strategy,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

    if min_tokens > 0 and len(result) > 1:
        result = _merge_small(result, min_tokens, tokenizer)

    return result


# -- Strategy: paragraph -------------------------------------------------------


def _chunk_paragraph(
    text: str, target_chars: int, max_chars: int, overlap_chars: int
) -> list[str]:
    """Split on paragraph boundaries, subsplit oversized blocks."""
    return split_paragraphs(
        text, heading=None, target_chars=target_chars,
        overlap=overlap_chars, max_chars=max_chars,
    )


# -- Strategy: sentence --------------------------------------------------------


def _chunk_sentence(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Split on sentence boundaries, group into target-sized chunks."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    filled = _greedy_fill(sentences, target_chars, joiner=" ")

    # Enforce max_chars ceiling
    result: list[str] = []
    for c in filled:
        if len(c) <= max_chars:
            result.append(c)
        else:
            result.extend(_hard_split(c, max_chars))

    return result


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundary detection."""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


# -- Strategy: fixed -----------------------------------------------------------


def _chunk_fixed(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Pure sliding window at word boundaries."""
    if len(text) <= max_chars:
        return [text]

    result: list[str] = []
    pos = 0
    while pos < len(text):
        end = pos + max_chars
        if end >= len(text):
            result.append(text[pos:].strip())
            break

        # Prefer breaking at a space in the latter half
        space_idx = text.rfind(" ", pos + max_chars // 2, end)
        if space_idx > pos:
            end = space_idx

        result.append(text[pos:end].strip())

        # Advance by (chunk_size - overlap), ensuring forward progress
        advance = max(end - pos - overlap_chars, 1)
        pos += advance

    return [c for c in result if c]


# -- Strategy: LLM -------------------------------------------------------------

_LLM_CHUNK_PROMPT = """\
You are a document chunking assistant. Given numbered sentences, group them \
into coherent chunks of roughly {target_tokens} tokens each.

Rules:
- Group consecutive sentences that share a topic or logical unit.
- Prefer boundaries where the topic, subject, or rhetorical purpose shifts.
- Every sentence must belong to exactly one group.
- Groups must be contiguous (no gaps, no overlaps).
- The union of all ranges must cover sentences 0 through {last_idx}.

Return JSON only, no markdown fences:
{{"chunks": [{{"start": 0, "end": 4, "topic": "brief label"}}, ...]}}"""

_LLM_WINDOW_SENTENCES = 300
_LLM_OVERLAP_SENTENCES = 30


def _resolve_api_key(api_key: str) -> str:
    """Resolve API key from argument or environment, raising if missing."""
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "API key required for strategy='llm'. Pass api_key= or set OPENAI_API_KEY."
        )
    return key


def _chunk_llm(
    text: str, target_tokens: int, max_tokens: int, api_key: str, model: str
) -> list[str]:
    """LLM-driven chunking: pre-split into sentences, ask LLM for groupings."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    if len(sentences) <= 3:
        return [text]

    try:
        key = _resolve_api_key(api_key)
        groups = _llm_get_groups(sentences, target_tokens, key, model)
        return _reassemble(sentences, groups)
    except Exception:
        logger.warning("LLM chunking failed, falling back to paragraph strategy")
        return _chunk_paragraph(
            text,
            _tokens_to_chars(target_tokens),
            _tokens_to_chars(max_tokens),
            _tokens_to_chars(50),
        )


async def _achunk_llm(
    text: str, target_tokens: int, max_tokens: int, api_key: str, model: str
) -> list[str]:
    """Async LLM-driven chunking."""
    sentences = _split_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    if len(sentences) <= 3:
        return [text]

    try:
        key = _resolve_api_key(api_key)
        groups = await _allm_get_groups(sentences, target_tokens, key, model)
        return _reassemble(sentences, groups)
    except Exception:
        logger.warning("Async LLM chunking failed, falling back to paragraph strategy")
        return _chunk_paragraph(
            text,
            _tokens_to_chars(target_tokens),
            _tokens_to_chars(max_tokens),
            _tokens_to_chars(50),
        )


def _build_llm_messages(
    sentences: list[str], target_tokens: int
) -> tuple[str, str]:
    """Build system and user messages for LLM chunking."""
    last_idx = len(sentences) - 1
    system = _LLM_CHUNK_PROMPT.format(target_tokens=target_tokens, last_idx=last_idx)
    user = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return system, user


def _llm_get_groups(
    sentences: list[str], target_tokens: int, api_key: str, model: str
) -> list[dict]:
    """Call LLM to get sentence groupings. Handles windowing for large docs."""
    from .llm.json_repair import safe_parse

    if len(sentences) <= _LLM_WINDOW_SENTENCES:
        system, user = _build_llm_messages(sentences, target_tokens)
        return _llm_call_sync(system, user, api_key, model, safe_parse)

    return _llm_windowed(
        sentences, target_tokens, api_key, model, _llm_call_sync, safe_parse
    )


async def _allm_get_groups(
    sentences: list[str], target_tokens: int, api_key: str, model: str
) -> list[dict]:
    """Async version of _llm_get_groups."""
    from .llm.json_repair import safe_parse

    if len(sentences) <= _LLM_WINDOW_SENTENCES:
        system, user = _build_llm_messages(sentences, target_tokens)
        return await _llm_call_async(system, user, api_key, model, safe_parse)

    # Windowed processing (sequential async calls)
    all_groups: list[dict] = []
    step = _LLM_WINDOW_SENTENCES - _LLM_OVERLAP_SENTENCES
    for win_start in range(0, len(sentences), step):
        win_end = min(win_start + _LLM_WINDOW_SENTENCES, len(sentences))
        window = sentences[win_start:win_end]
        system, user = _build_llm_messages(window, target_tokens)
        groups = await _llm_call_async(system, user, api_key, model, safe_parse)
        for g in groups:
            g["start"] += win_start
            g["end"] += win_start
        all_groups.extend(groups)
        if win_end >= len(sentences):
            break

    return _resolve_overlaps(all_groups, len(sentences))


def _llm_call_sync(
    system: str,
    user: str,
    api_key: str,
    model: str,
    parse_json: Callable[[str], dict],
) -> list[dict]:
    """Make a sync LLM call and parse the response."""
    from .llm.client import get_client

    client = get_client(api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    result = parse_json(raw)
    return _validate_groups(result.get("chunks", []))


async def _llm_call_async(
    system: str,
    user: str,
    api_key: str,
    model: str,
    parse_json: Callable[[str], dict],
) -> list[dict]:
    """Make an async LLM call and parse the response."""
    from .llm.async_client import get_async_client

    client = get_async_client(api_key)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content or "{}"
    result = parse_json(raw)
    return _validate_groups(result.get("chunks", []))


def _llm_windowed(
    sentences: list[str],
    target_tokens: int,
    api_key: str,
    model: str,
    call_fn: Callable[..., list[dict]],
    parse_json: Callable[[str], dict],
) -> list[dict]:
    """Process sentences in overlapping windows for large documents."""
    all_groups: list[dict] = []
    step = _LLM_WINDOW_SENTENCES - _LLM_OVERLAP_SENTENCES

    for win_start in range(0, len(sentences), step):
        win_end = min(win_start + _LLM_WINDOW_SENTENCES, len(sentences))
        window = sentences[win_start:win_end]
        system, user = _build_llm_messages(window, target_tokens)
        groups = call_fn(system, user, api_key, model, parse_json)
        for g in groups:
            g["start"] += win_start
            g["end"] += win_start
        all_groups.extend(groups)
        if win_end >= len(sentences):
            break

    return _resolve_overlaps(all_groups, len(sentences))


def _validate_groups(groups: list) -> list[dict]:
    """Validate and normalize LLM-returned groups."""
    validated: list[dict] = []
    dropped = 0
    for g in groups:
        if isinstance(g, dict) and "start" in g and "end" in g:
            validated.append({
                "start": int(g["start"]),
                "end": int(g["end"]),
                "topic": g.get("topic", ""),
            })
        else:
            dropped += 1
    if dropped:
        logger.warning("Dropped %d invalid groups from LLM output", dropped)
    return validated


def _resolve_overlaps(groups: list[dict], total: int) -> list[dict]:
    """Resolve overlapping groups from windowed processing.

    Sorts by start index and merges overlapping ranges.
    """
    if not groups:
        return groups

    groups.sort(key=lambda g: g["start"])

    resolved: list[dict] = [groups[0]]
    for g in groups[1:]:
        prev = resolved[-1]
        if g["start"] > prev["end"]:
            resolved.append(g)
        elif g["end"] > prev["end"]:
            prev["end"] = g["end"]

    return resolved


def _reassemble(sentences: list[str], groups: list[dict]) -> list[str]:
    """Reassemble text chunks from sentence groups."""
    if not groups:
        return [" ".join(sentences)]

    chunks: list[str] = []
    covered: set[int] = set()

    for g in groups:
        start = max(0, g["start"])
        end = min(len(sentences) - 1, g["end"])
        chunk_sentences = sentences[start:end + 1]
        if chunk_sentences:
            chunks.append(" ".join(chunk_sentences))
            covered.update(range(start, end + 1))

    # Pick up any uncovered sentences as additional chunks
    uncovered = [i for i in range(len(sentences)) if i not in covered]
    if uncovered:
        buf: list[int] = [uncovered[0]]
        for idx in uncovered[1:]:
            if idx == buf[-1] + 1:
                buf.append(idx)
            else:
                chunks.append(" ".join(sentences[buf[0]:buf[-1] + 1]))
                buf = [idx]
        chunks.append(" ".join(sentences[buf[0]:buf[-1] + 1]))

    return chunks if chunks else [" ".join(sentences)]


# -- Small chunk merging -------------------------------------------------------


def _merge_small(
    chunks: list[str],
    min_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> list[str]:
    """Merge chunks below min_tokens into their neighbors."""
    merged: list[str] = []
    for c in chunks:
        est = estimate_tokens(c, tokenizer)
        if est >= min_tokens or not merged:
            merged.append(c)
        else:
            merged[-1] = merged[-1] + "\n\n" + c

    # Handle trailing small chunk
    if len(merged) > 1 and estimate_tokens(merged[-1], tokenizer) < min_tokens:
        merged[-2] = merged[-2] + "\n\n" + merged.pop()

    return merged


# -- Shared utilities (also used by pipeline/chunking.py) ----------------------


def split_paragraphs(
    text: str,
    heading: str | None,
    target_chars: int,
    overlap: int,
    max_chars: int | None = None,
) -> list[str]:
    """Split text on paragraph boundaries at ~target_chars with overlap.

    Oversized paragraphs (e.g. PDF pages with only single-newline breaks) are
    subsplit using cascading strategies: line breaks -> sentences -> hard cut.
    No chunk will exceed max_chars.
    """
    if max_chars is None:
        max_chars = target_chars * 2

    paragraphs = re.split(r"\n{2,}", text)
    result: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > target_chars:
            sub_parts = _subsplit(para, target_chars, max_chars)
        else:
            sub_parts = [para]

        for part in sub_parts:
            if buf and buf_len + len(part) > target_chars:
                chunk_text = "\n\n".join(buf)
                if heading:
                    chunk_text = f"{heading}\n\n{chunk_text}"
                result.append(chunk_text)

                overlap_buf: list[str] = []
                overlap_len = 0
                for p in reversed(buf):
                    if overlap_len + len(p) > overlap:
                        break
                    overlap_buf.insert(0, p)
                    overlap_len += len(p)
                buf = overlap_buf
                buf_len = overlap_len

            buf.append(part)
            buf_len += len(part)

    if buf:
        chunk_text = "\n\n".join(buf)
        if heading:
            chunk_text = f"{heading}\n\n{chunk_text}"
        result.append(chunk_text)

    return result if result else [f"{heading}\n\n{text}" if heading else text]


def _subsplit(text: str, target_chars: int, max_chars: int) -> list[str]:
    """Split an oversized text block using cascading strategies.

    Level 1: line breaks
    Level 2: sentence boundaries
    Level 3: hard cut at word boundary
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    filled = _greedy_fill(lines, target_chars, joiner="\n")

    result: list[str] = []
    for c in filled:
        if len(c) <= max_chars:
            result.append(c)
        else:
            sentences = _SENTENCE_BOUNDARY.split(c)
            sent_filled = _greedy_fill(sentences, target_chars, joiner=" ")

            for sc in sent_filled:
                if len(sc) <= max_chars:
                    result.append(sc)
                else:
                    result.extend(_hard_split(sc, max_chars))

    return result


def _greedy_fill(pieces: list[str], target_chars: int, joiner: str) -> list[str]:
    """Accumulate pieces into chunks up to target_chars, joined with joiner."""
    result: list[str] = []
    buf: list[str] = []
    buf_len = 0
    joiner_len = len(joiner)

    for p in pieces:
        new_len = buf_len + (joiner_len if buf else 0) + len(p)
        if buf and new_len > target_chars:
            result.append(joiner.join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len = new_len

    if buf:
        result.append(joiner.join(buf))

    return result if result else [joiner.join(pieces)]


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split text at max_chars, preferring word boundaries."""
    parts: list[str] = []
    while len(text) > max_chars:
        cut = max_chars
        space_idx = text.rfind(" ", max_chars // 2, cut)
        if space_idx > 0:
            cut = space_idx
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts
