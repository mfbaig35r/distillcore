"""Pydantic models for the distillcore pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int
    text: str


class ExtractionResult(BaseModel):
    pages: list[PageText] = Field(default_factory=list)
    full_text: str = ""
    page_count: int = 0
    format: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Section(BaseModel):
    heading: str | None = None
    section_type: str = "general"
    content: str = ""
    subsections: list[Section] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None


class TranscriptTurn(BaseModel):
    speaker: str
    role: str = "unknown"
    content: str
    page: int | None = None
    line_start: int | None = None


class DocumentMetadata(BaseModel):
    source_filename: str
    document_title: str | None = None
    document_type: str = "unknown"
    page_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    metadata: DocumentMetadata
    sections: list[Section] = Field(default_factory=list)
    transcript_turns: list[TranscriptTurn] = Field(default_factory=list)
    full_text: str = ""


class DocumentChunk(BaseModel):
    chunk_index: int
    text: str
    token_estimate: int
    section_type: str | None = None
    section_heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    speakers: list[str] | None = None
    # LLM-enriched fields
    topic: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    relevance: str | None = None
    # Embedding (set after embed stage)
    embedding: list[float] | None = None


class ChunkedDocument(BaseModel):
    chunk_count: int
    chunks: list[DocumentChunk]


class ValidationReport(BaseModel):
    structuring_coverage: float = 0.0
    chunking_coverage: float = 0.0
    end_to_end_coverage: float = 0.0
    missing_segments: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    passed: bool = False


class ProcessingResult(BaseModel):
    document: Document
    chunks: list[DocumentChunk]
    validation: ValidationReport


class BatchResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[ProcessingResult]
