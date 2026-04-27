"""Async LLM-based document classification."""

from __future__ import annotations

import logging

from ..config import DistillConfig
from ..llm.async_client import get_async_client
from ..llm.json_repair import safe_parse
from ..models import DocumentMetadata
from ._shared import (
    build_classification_user_msg,
    build_default_metadata,
    fallback_metadata,
    sanitize_classification_output,
)

logger = logging.getLogger(__name__)


async def classify_document_async(
    filename: str,
    pages_text: list[str],
    page_count: int,
    config: DistillConfig,
) -> DocumentMetadata:
    """Async version of classify_document.

    Falls back to unknown values on failure — never crashes the pipeline.
    """
    prompt = config.domain.classification_prompt
    if not prompt:
        return fallback_metadata(filename, page_count)

    user_msg = build_classification_user_msg(filename, pages_text)

    try:
        result = await _call_llm_async(user_msg, prompt, config)
        result = sanitize_classification_output(result)
    except Exception as e:
        logger.error(f"Classification failed for {filename}: {e}")
        return fallback_metadata(filename, page_count)

    parser = config.domain.parse_classification
    if parser:
        try:
            return parser(result, filename, page_count)
        except Exception as e:
            logger.error(f"Classification parsing failed for {filename}: {e}")
            return fallback_metadata(filename, page_count)

    return build_default_metadata(result, filename, page_count)


async def _call_llm_async(
    user_msg: str, prompt: str, config: DistillConfig, retry: bool = True
) -> dict:
    """Call LLM for classification. Retries once on failure."""
    client = get_async_client(config.resolve_api_key())

    try:
        response = await client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        result = safe_parse(raw)
        if result:
            return result
        raise ValueError("Empty parse result")
    except Exception:
        if retry:
            logger.warning("Classification retry")
            return await _call_llm_async(user_msg, prompt, config, retry=False)
        raise
