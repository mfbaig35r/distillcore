"""Shared test fixtures."""

import pytest

from distillcore.config import DistillConfig
from distillcore.models import (
    Document,
    DocumentMetadata,
    ExtractionResult,
    PageText,
    Section,
    TranscriptTurn,
)


@pytest.fixture
def sample_text() -> str:
    return "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."


@pytest.fixture
def sample_extraction(sample_text: str) -> ExtractionResult:
    return ExtractionResult(
        pages=[PageText(page_number=1, text=sample_text)],
        full_text=sample_text,
        page_count=1,
        format="txt",
    )


@pytest.fixture
def sample_document(sample_text: str) -> Document:
    return Document(
        metadata=DocumentMetadata(source_filename="test.txt", page_count=1),
        sections=[Section(heading="Test", content=sample_text)],
        full_text=sample_text,
    )


@pytest.fixture
def transcript_document() -> Document:
    turns = [
        TranscriptTurn(speaker="Alice", role="attorney", content="First question?", page=1),
        TranscriptTurn(speaker="Bob", role="witness", content="First answer.", page=1),
        TranscriptTurn(speaker="Alice", role="attorney", content="Second question?", page=2),
        TranscriptTurn(speaker="Bob", role="witness", content="Second answer.", page=2),
    ]
    full = "\n\n".join(f"{t.speaker}: {t.content}" for t in turns)
    return Document(
        metadata=DocumentMetadata(source_filename="transcript.pdf", page_count=2),
        transcript_turns=turns,
        full_text=full,
    )


@pytest.fixture
def default_config() -> DistillConfig:
    return DistillConfig(openai_api_key="test-key")
