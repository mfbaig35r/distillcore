"""Tests for distillcore.config."""

import os
from unittest.mock import patch

from distillcore.config import ChunkConfig, DistillConfig, DomainConfig, EmbeddingConfig


class TestChunkConfig:
    def test_defaults(self) -> None:
        c = ChunkConfig()
        assert c.target_tokens == 500
        assert c.overlap_chars == 200
        assert c.max_tokens == 1000

    def test_override(self) -> None:
        c = ChunkConfig(target_tokens=1000, overlap_chars=100)
        assert c.target_tokens == 1000
        assert c.overlap_chars == 100


class TestEmbeddingConfig:
    def test_defaults(self) -> None:
        c = EmbeddingConfig()
        assert c.model == "text-embedding-3-small"
        assert c.embed_fn is None

    def test_custom_embed_fn(self) -> None:
        def my_embed(texts: list[str]) -> list[list[float]]:
            return [[0.0] * 3 for _ in texts]

        c = EmbeddingConfig(embed_fn=my_embed)
        assert c.embed_fn is not None
        result = c.embed_fn(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 3


class TestDomainConfig:
    def test_defaults(self) -> None:
        d = DomainConfig()
        assert d.name == "generic"
        assert d.classification_prompt == ""
        assert d.parse_classification is None

    def test_custom_prompts(self) -> None:
        d = DomainConfig(
            name="medical",
            classification_prompt="You are a medical document analyst.",
            enrichment_prompt="Tag medical concepts.",
        )
        assert d.name == "medical"
        assert "medical" in d.classification_prompt


class TestDistillConfig:
    def test_defaults(self) -> None:
        c = DistillConfig()
        assert c.openai_model == "gpt-4o"
        assert c.enrich_chunks is True
        assert c.enable_ocr is True
        assert isinstance(c.chunk, ChunkConfig)
        assert isinstance(c.embedding, EmbeddingConfig)
        assert isinstance(c.domain, DomainConfig)

    def test_resolve_api_key_from_field(self) -> None:
        c = DistillConfig(openai_api_key="sk-test")
        assert c.resolve_api_key() == "sk-test"

    def test_resolve_api_key_from_env(self) -> None:
        c = DistillConfig()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            assert c.resolve_api_key() == "sk-env"

    def test_resolve_api_key_empty(self) -> None:
        c = DistillConfig()
        with patch.dict(os.environ, {}, clear=True):
            assert c.resolve_api_key() == ""

    def test_nested_override(self) -> None:
        c = DistillConfig(chunk=ChunkConfig(target_tokens=1000))
        assert c.chunk.target_tokens == 1000
        assert c.embedding.model == "text-embedding-3-small"
