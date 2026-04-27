"""Generic (domain-neutral) preset for distillcore."""

from __future__ import annotations

from ..config import DomainConfig
from ..models import DocumentMetadata

GENERIC_CLASSIFICATION_PROMPT = """\
You are a document analyst. Given the first two pages of a document and its \
filename, extract metadata as JSON.

Extract:
- document_type: classify the document (e.g., "report", "article", "memo", \
"letter", "contract", "manual", "presentation", "spreadsheet", "form", \
"correspondence", "policy", "proposal", "minutes", "invoice", "receipt", \
or any other appropriate document type)
- document_title: the actual title as it appears in the document
- author: the author or creator if present
- date: the document date in YYYY-MM-DD format if present, null otherwise
- summary: a one-sentence summary of the document's purpose

Return ONLY valid JSON matching this schema, no markdown fences:
{
  "document_type": "...",
  "document_title": "...",
  "author": "..." | null,
  "date": "YYYY-MM-DD" | null,
  "summary": "..."
}"""

GENERIC_STRUCTURING_PROMPT = """\
You are a document analyst. Given the full text of a document and its metadata, \
identify its hierarchical section structure and return section boundaries as JSON.

Rules:
- Break the document into hierarchical sections identified by headings and page ranges.
- Each section has: heading (string|null), section_type (string), \
page_range ([start, end]), subsections (array of section objects).
- section_type values: "title", "header", "body", "conclusion", "appendix", \
"table_of_contents", "references", "signature", "general"
- page_range is REQUIRED for every section. Use 1-based page numbers matching \
the --- PAGE N --- markers in the input.
- Do NOT include section content text. Return boundaries only.
- Return ONLY valid JSON matching this schema, no markdown fences.

Output schema:
{
  "sections": [{"heading": str|null, "section_type": str, \
"page_range": [int,int], "subsections": [...]}]
}"""

GENERIC_ENRICHMENT_PROMPT = """\
You are a document analyst. Given a list of text chunks from a document, \
enrich each chunk with metadata for semantic search and retrieval.

For each chunk, provide:
- topic: A short descriptive label (e.g., "Executive Summary", "Budget Analysis", \
"Technical Requirements")
- key_concepts: 3-8 key concepts, entities, or themes in the chunk
- relevance: "high", "medium", or "low" — how central is this chunk to the \
core content of the document?
  - high: directly addresses key findings, decisions, core content
  - medium: supporting context, background, methodology
  - low: boilerplate, headers, formatting artifacts, signatures

Return ONLY valid JSON matching this schema, no markdown fences:
{
  "enrichments": [
    {"chunk_index": 0, "topic": "...", "key_concepts": ["..."], \
"relevance": "high|medium|low"},
    ...
  ]
}"""


def _parse_generic_classification(
    result: dict, filename: str, page_count: int
) -> DocumentMetadata:
    """Parse generic classification LLM result into DocumentMetadata."""
    extra = {}
    if result.get("author"):
        extra["author"] = result["author"]
    if result.get("date"):
        extra["date"] = result["date"]
    if result.get("summary"):
        extra["summary"] = result["summary"]

    return DocumentMetadata(
        source_filename=filename,
        document_title=result.get("document_title"),
        document_type=result.get("document_type", "unknown"),
        page_count=page_count,
        extra=extra,
    )


GENERIC_PRESET = DomainConfig(
    name="generic",
    classification_prompt=GENERIC_CLASSIFICATION_PROMPT,
    structuring_prompt=GENERIC_STRUCTURING_PROMPT,
    enrichment_prompt=GENERIC_ENRICHMENT_PROMPT,
    parse_classification=_parse_generic_classification,
)
