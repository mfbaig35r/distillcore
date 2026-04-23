"""Tests for distillcore.presets."""

import pytest

from distillcore.config import DomainConfig
from distillcore.presets import list_presets, load_preset, register_preset


class TestPresets:
    def test_generic_registered(self) -> None:
        preset = load_preset("generic")
        assert preset.name == "generic"
        assert preset.classification_prompt != ""
        assert preset.parse_classification is not None

    def test_legal_registered(self) -> None:
        preset = load_preset("legal")
        assert preset.name == "legal"
        assert "legal" in preset.classification_prompt.lower()
        assert preset.transcript_prompt != ""

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            load_preset("nonexistent")

    def test_list_presets(self) -> None:
        presets = list_presets()
        assert "generic" in presets
        assert "legal" in presets

    def test_custom_preset(self) -> None:
        custom = DomainConfig(name="medical", classification_prompt="Medical analyst")
        register_preset("medical", custom)
        loaded = load_preset("medical")
        assert loaded.name == "medical"


class TestGenericParser:
    def test_parse_classification(self) -> None:
        preset = load_preset("generic")
        result = {
            "document_type": "report",
            "document_title": "Q4 Summary",
            "author": "Jane Doe",
            "date": "2024-01-15",
            "summary": "Quarterly report.",
        }
        meta = preset.parse_classification(result, "report.pdf", 10)
        assert meta.document_type == "report"
        assert meta.document_title == "Q4 Summary"
        assert meta.extra["author"] == "Jane Doe"


class TestLegalParser:
    def test_parse_classification(self) -> None:
        preset = load_preset("legal")
        result = {
            "document_type": "motion",
            "document_title": "Motion for Summary Judgment",
            "case_number": "2024-CV-001",
            "court": "Superior Court",
            "judge": "Smith",
            "filing_party": "plaintiff",
            "filing_date": "2024-03-15",
            "is_transcript": False,
            "attorneys": [{"name": "Atty A", "bar_number": "123", "representing": "Plaintiff"}],
        }
        meta = preset.parse_classification(result, "motion.pdf", 5)
        assert meta.document_type == "motion"
        assert meta.extra["case_number"] == "2024-CV-001"
        assert meta.extra["filing_date"] == "2024-03-15"
        assert len(meta.extra["attorneys"]) == 1

    def test_parse_with_bad_date(self) -> None:
        preset = load_preset("legal")
        result = {
            "document_type": "order",
            "filing_date": "not-a-date",
        }
        meta = preset.parse_classification(result, "order.pdf", 2)
        assert "filing_date" not in meta.extra
