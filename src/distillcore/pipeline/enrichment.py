"""LLM-based chunk enrichment with topic, key_concepts, and relevance."""

from __future__ import annotations

import logging

from ..config import DistillConfig
from ..llm.client import get_client
from ..llm.json_repair import safe_parse
from ..models import DocumentChunk
from ._shared import apply_enrichments, build_chunk_summaries, truncate_enrichment_msg

logger = logging.getLogger(__name__)


def enrich_chunks(
    chunks: list[DocumentChunk],
    document_type: str,
    config: DistillConfig,
) -> list[DocumentChunk]:
    """Call LLM to enrich chunks with topic, key_concepts, and relevance."""
    prompt = config.domain.enrichment_prompt
    if not prompt:
        return chunks

    chunk_summaries = build_chunk_summaries(chunks)
    user_msg = truncate_enrichment_msg(chunk_summaries, document_type, len(chunks))

    try:
        client = get_client(config.resolve_api_key())
        response = client.chat.completions.create(
            model=config.openai_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = response.choices[0].message.content or "{}"
        result = safe_parse(raw)
        enriched_count = apply_enrichments(chunks, result)
        logger.info(f"Enriched {enriched_count}/{len(chunks)} chunks")

    except Exception as e:
        logger.error(f"LLM enrichment failed, returning unenriched chunks: {e}")

    return chunks
