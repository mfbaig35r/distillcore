"""Tests for distillcore.embedding providers."""

import json
from unittest.mock import MagicMock, patch

from distillcore.embedding.ollama import ollama_embedder
from distillcore.embedding.openai import openai_embedder


class TestOpenaiEmbedder:
    def test_returns_callable(self) -> None:
        fn = openai_embedder(model="text-embedding-3-small", api_key="test")
        assert callable(fn)

    def test_calls_openai(self) -> None:
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]

        with patch("distillcore.embedding.openai.get_client") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_resp
            fn = openai_embedder(model="text-embedding-3-small", api_key="sk-test")
            result = fn(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        mock_client.return_value.embeddings.create.assert_called_once_with(
            input=["hello", "world"], model="text-embedding-3-small"
        )


class TestOllamaEmbedder:
    def test_returns_callable(self) -> None:
        fn = ollama_embedder(model="nomic-embed-text")
        assert callable(fn)

    def test_calls_ollama_api(self) -> None:
        response_data = {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("distillcore.embedding.ollama.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = mock_resp
            fn = ollama_embedder(model="nomic-embed-text")
            result = fn(["hello", "world"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

        call_args = mock_urlopen.call_args[0][0]
        body = json.loads(call_args.data)
        assert body["model"] == "nomic-embed-text"
        assert body["input"] == ["hello", "world"]

    def test_custom_base_url(self) -> None:
        response_data = {"embeddings": [[0.1]]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("distillcore.embedding.ollama.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = mock_resp
            fn = ollama_embedder(base_url="http://remote:8080")
            fn(["test"])

        call_args = mock_urlopen.call_args[0][0]
        assert "http://remote:8080/api/embed" in call_args.full_url


class TestLocalEmbedder:
    def test_loads_model_and_embeds(self) -> None:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = MagicMock(
            tolist=MagicMock(return_value=[[0.1, 0.2]])
        )
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("distillcore.embedding.local.SentenceTransformer", mock_cls):
            from distillcore.embedding.local import local_embedder

            fn = local_embedder(model="all-MiniLM-L6-v2", device="cpu")
            result = fn(["hello"])

        mock_cls.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")
        mock_instance.encode.assert_called_once()
        assert result == [[0.1, 0.2]]

    def test_not_installed_raises(self) -> None:
        with patch("distillcore.embedding.local.SentenceTransformer", None):
            from distillcore.embedding.local import local_embedder

            try:
                local_embedder()
                assert False, "Should have raised"
            except ImportError as e:
                assert "distillcore[local]" in str(e)


class TestCohereEmbedder:
    def test_creates_client_and_embeds(self) -> None:
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.embeddings.float_ = [[0.1, 0.2], [0.3, 0.4]]
        mock_client.embed.return_value = mock_resp
        mock_cohere.ClientV2.return_value = mock_client

        with patch("distillcore.embedding.cohere.cohere", mock_cohere):
            from distillcore.embedding.cohere import cohere_embedder

            fn = cohere_embedder(model="embed-english-v3.0", api_key="test-key")
            result = fn(["hello", "world"])

        assert len(result) == 2
        mock_client.embed.assert_called_once_with(
            texts=["hello", "world"],
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )

    def test_custom_input_type(self) -> None:
        mock_cohere = MagicMock()
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.embeddings.float_ = [[0.1]]
        mock_client.embed.return_value = mock_resp
        mock_cohere.ClientV2.return_value = mock_client

        with patch("distillcore.embedding.cohere.cohere", mock_cohere):
            from distillcore.embedding.cohere import cohere_embedder

            fn = cohere_embedder(input_type="search_query")
            fn(["query"])

        _, kwargs = mock_client.embed.call_args
        assert kwargs["input_type"] == "search_query"

    def test_not_installed_raises(self) -> None:
        with patch("distillcore.embedding.cohere.cohere", None):
            from distillcore.embedding.cohere import cohere_embedder

            try:
                cohere_embedder()
                assert False, "Should have raised"
            except ImportError as e:
                assert "distillcore[cohere]" in str(e)
