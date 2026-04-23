"""JSON parse and repair utilities for truncated LLM output."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def safe_parse(raw: str) -> dict:
    """Parse JSON with fallback for truncated output."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("JSON parse failed, attempting repair")
        patched = try_fix_truncated_json(raw)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            logger.error("JSON repair failed, returning empty structure")
            return {}


def try_fix_truncated_json(raw: str) -> str:
    """Best-effort fix for JSON truncated by token limit."""
    raw = raw.rstrip()
    if raw.count('"') % 2 == 1:
        raw += '"'

    opens = 0
    open_sq = 0
    for ch in raw:
        if ch == "{":
            opens += 1
        elif ch == "}":
            opens -= 1
        elif ch == "[":
            open_sq += 1
        elif ch == "]":
            open_sq -= 1

    raw += "]" * open_sq + "}" * opens
    return raw
