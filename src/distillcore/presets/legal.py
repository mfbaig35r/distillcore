"""Legal domain preset for distillcore — extracted from Vector Lex."""

from __future__ import annotations

from datetime import date

from ..config import DomainConfig
from ..models import DocumentMetadata

LEGAL_CLASSIFICATION_PROMPT = """\
You are a legal document analyst. Given the first two pages of a legal document and its \
filename, extract metadata as JSON.

Extract:
- document_type: classify the document (e.g., "motion", "response", "reply", "order", \
"ruling", "petition", "transcript", "notice", "stipulation", "complaint", "memorandum", \
"declaration", "affidavit", "subpoena", "brief", "exhibit", "contract", "agreement", \
"settlement", "deposition", or any other appropriate legal document type)
- filing_party: who filed it (e.g., "plaintiff", "defendant", "petitioner", "respondent", \
"court", "third_party", or any other appropriate party designation)
- document_title: the actual title as it appears in the document
- case_number: the case number if present
- court: the court name if present
- judge: the judge name if present
- filing_date: the filing date in YYYY-MM-DD format if present, null otherwise
- is_transcript: true if this is a court transcript or deposition transcript
- attorneys: array of objects with "name", "bar_number" (string or null), \
"representing" (party name or role)

Return ONLY valid JSON matching this schema, no markdown fences:
{
  "document_type": "...",
  "filing_party": "...",
  "document_title": "...",
  "case_number": "...",
  "court": "...",
  "judge": "...",
  "filing_date": "YYYY-MM-DD" | null,
  "is_transcript": true | false,
  "attorneys": [{"name": "...", "bar_number": "..." | null, "representing": "..."}]
}"""

LEGAL_STRUCTURING_PROMPT = """\
You are a legal document analyst. Given the full text of a court document and its metadata, \
identify its hierarchical section structure and return section boundaries as JSON.

Rules:
- Break the document into hierarchical sections identified by headings and page ranges.
- Each section has: heading (string|null), section_type (string), \
page_range ([start, end]), subsections (array of section objects).
- section_type values: "caption", "header", "findings", "argument", "prayer", "orders", \
"conclusion", "testimony", "exhibit_list", "index", "signature", "general"
- page_range is REQUIRED for every section. Use 1-based page numbers matching \
the --- PAGE N --- markers in the input.
- Do NOT include section content text. Return boundaries only.
- Extract all legal citations into "legal_citations" array.
- Extract all exhibit references into "exhibit_references" array.
- For rulings/orders: extract each discrete court order into "court_orders" array.
- Return ONLY valid JSON matching this schema, no markdown fences.

Output schema:
{
  "sections": [{"heading": str|null, "section_type": str, \
"page_range": [int,int], "subsections": [...]}],
  "legal_citations": [str],
  "exhibit_references": [str],
  "court_orders": [str]|null
}"""

LEGAL_TRANSCRIPT_PROMPT = """\
You are a legal transcript analyst. Given a CHUNK of a court transcript (pages {start}-{end}), \
extract speaker turns and metadata.

Rules:
- Extract each speaker turn: speaker (name), role ("judge", "attorney", "witness", "clerk"), \
content (what they said — keep it verbatim), page (int), line_start (int|null).
- Extract any legal citations and exhibit references mentioned in this chunk.
- Extract any court orders issued in this chunk.
- For structural pages (index, exhibit list, cover page), return them as sections instead of turns.
- IMPORTANT: Preserve ALL spoken content verbatim. Do not summarize testimony.
- Return ONLY valid JSON, no markdown fences.

Output schema:
{{
  "transcript_turns": [{{"speaker": str, "role": str, "content": str, \
"page": int|null, "line_start": int|null}}],
  "sections": [{{"heading": str|null, "section_type": str, "content": str, \
"subsections": [], "page_range": [int,int]|null}}],
  "legal_citations": [str],
  "exhibit_references": [str],
  "court_orders": [str]|null
}}"""

LEGAL_ENRICHMENT_PROMPT = """\
You are a legal document analyst. Given a list of text chunks from a legal document, \
enrich each chunk with metadata for semantic search and retrieval.

For each chunk, provide:
- topic: A short descriptive label (e.g., "Motion for Summary Judgment", \
"Witness Testimony — Damages", "Settlement Terms", "Court Order on Discovery")
- key_concepts: 3-8 key legal concepts, entities, or themes in the chunk
- relevance: "high", "medium", or "low" — how central is this chunk to the \
core issues in the case?
  - high: directly addresses key facts, court findings, critical testimony, \
dispositive orders
  - medium: supporting evidence, procedural context, attorney arguments
  - low: boilerplate, cover pages, certificates, signature blocks

Return ONLY valid JSON matching this schema, no markdown fences:
{
  "enrichments": [
    {"chunk_index": 0, "topic": "...", "key_concepts": ["..."], \
"relevance": "high|medium|low"},
    ...
  ]
}"""


def _parse_legal_classification(
    result: dict, filename: str, page_count: int
) -> DocumentMetadata:
    """Parse legal classification LLM result into DocumentMetadata."""
    extra: dict = {}

    if result.get("case_number"):
        extra["case_number"] = result["case_number"]
    if result.get("court"):
        extra["court"] = result["court"]
    if result.get("judge"):
        extra["judge"] = result["judge"]
    if result.get("filing_party"):
        extra["filing_party"] = result["filing_party"]
    if result.get("is_transcript"):
        extra["is_transcript"] = result["is_transcript"]
    if result.get("attorneys"):
        extra["attorneys"] = result["attorneys"]

    # Parse filing date
    date_str = result.get("filing_date")
    if date_str:
        try:
            extra["filing_date"] = date.fromisoformat(date_str).isoformat()
        except (ValueError, TypeError):
            pass

    return DocumentMetadata(
        source_filename=filename,
        document_title=result.get("document_title"),
        document_type=result.get("document_type", "unknown"),
        page_count=page_count,
        extra=extra,
    )


LEGAL_PRESET = DomainConfig(
    name="legal",
    classification_prompt=LEGAL_CLASSIFICATION_PROMPT,
    structuring_prompt=LEGAL_STRUCTURING_PROMPT,
    transcript_prompt=LEGAL_TRANSCRIPT_PROMPT,
    enrichment_prompt=LEGAL_ENRICHMENT_PROMPT,
    parse_classification=_parse_legal_classification,
)
