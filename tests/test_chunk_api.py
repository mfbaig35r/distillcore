"""Tests for distillcore.chunking — the public chunk() API."""

from unittest.mock import MagicMock, patch

import pytest

from distillcore.chunking import achunk, chunk


class TestChunkParagraph:
    """Default strategy: paragraph splitting."""

    def test_basic(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunk(text)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(c, str) for c in result)

    def test_splits_large_text(self) -> None:
        text = "\n\n".join([f"Paragraph {i}. " + "x" * 300 for i in range(10)])
        result = chunk(text, target_tokens=200)
        assert len(result) > 1

    def test_empty_text(self) -> None:
        assert chunk("") == []
        assert chunk("   ") == []

    def test_single_paragraph(self) -> None:
        result = chunk("Just one paragraph here.")
        assert len(result) == 1
        assert result[0] == "Just one paragraph here."

    def test_overlap(self) -> None:
        text = "\n\n".join([f"Paragraph {i}." for i in range(20)])
        result = chunk(text, target_tokens=10, overlap_tokens=5)
        assert len(result) > 1


class TestChunkSentence:
    """Sentence-based splitting."""

    def test_basic(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth here."
        result = chunk(text, strategy="sentence")
        assert len(result) >= 1

    def test_groups_sentences(self) -> None:
        sentences = [f"Sentence number {i} is about something." for i in range(20)]
        text = " ".join(sentences)
        result = chunk(text, strategy="sentence", target_tokens=50)
        assert len(result) > 1

    def test_single_sentence(self) -> None:
        result = chunk("Just one sentence.", strategy="sentence")
        assert len(result) == 1


class TestChunkFixed:
    """Fixed-size sliding window."""

    def test_basic(self) -> None:
        text = "word " * 500  # ~2500 chars
        result = chunk(text, strategy="fixed", max_tokens=100)
        assert len(result) > 1

    def test_small_text(self) -> None:
        result = chunk("tiny text", strategy="fixed")
        assert len(result) == 1
        assert result[0] == "tiny text"

    def test_overlap(self) -> None:
        text = "word " * 500
        result_no_overlap = chunk(text, strategy="fixed", max_tokens=100, overlap_tokens=0)
        result_overlap = chunk(text, strategy="fixed", max_tokens=100, overlap_tokens=25)
        # With overlap, we get more chunks (some content repeated)
        assert len(result_overlap) >= len(result_no_overlap)

    def test_respects_max_tokens(self) -> None:
        text = "word " * 1000
        max_tokens = 50
        max_chars = max_tokens * 4
        result = chunk(text, strategy="fixed", max_tokens=max_tokens)
        for c in result:
            assert len(c) <= max_chars + 10  # small tolerance for word boundaries


class TestChunkLLM:
    """LLM-driven chunking with mocked API calls."""

    def _mock_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    def test_basic_llm_chunking(self) -> None:
        text = (
            "The economy grew last quarter. GDP increased by 3%. "
            "Unemployment fell. Meanwhile, in tech news. "
            "AI advances continue. New models released."
        )
        llm_response = '{"chunks": [{"start": 0, "end": 2, "topic": "economy"}, '
        llm_response += '{"start": 3, "end": 5, "topic": "tech"}]}'

        with patch("distillcore.llm.client.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                self._mock_response(llm_response)
            )
            mock_get.return_value = mock_client

            result = chunk(text, strategy="llm", api_key="test-key")
            assert len(result) == 2
            assert "economy" in result[0].lower() or "GDP" in result[0]

    def test_llm_fallback_on_error(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        with patch("distillcore.llm.client.get_client") as mock_get:
            mock_get.side_effect = Exception("API error")

            result = chunk(text, strategy="llm", api_key="test-key")
            # Should fall back to paragraph strategy
            assert len(result) >= 1

    def test_llm_prompt_structure(self) -> None:
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        llm_response = '{"chunks": [{"start": 0, "end": 4, "topic": "all"}]}'

        with patch("distillcore.llm.client.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = (
                self._mock_response(llm_response)
            )
            mock_get.return_value = mock_client

            chunk(text, strategy="llm", api_key="test-key", target_tokens=300)

            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert "300" in messages[0]["content"]  # target_tokens in prompt
            assert messages[1]["role"] == "user"
            assert "[0]" in messages[1]["content"]  # numbered sentences
            assert call_kwargs["temperature"] == 0
            assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_few_sentences_returns_whole(self) -> None:
        """Text with <=3 sentences should return as-is without LLM call."""
        text = "One sentence. Two sentences."

        with patch("distillcore.llm.client.get_client") as mock_get:
            result = chunk(text, strategy="llm", api_key="test-key")
            mock_get.assert_not_called()
            assert len(result) == 1
            assert result[0] == text


class TestAchunkLLM:
    """Async LLM chunking."""

    def _mock_response(self, content: str) -> MagicMock:
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    @pytest.mark.asyncio
    async def test_async_llm_chunking(self) -> None:
        from unittest.mock import AsyncMock

        text = (
            "The economy grew. GDP increased. Unemployment fell. "
            "In tech news. AI advances. New models released."
        )
        llm_response = '{"chunks": [{"start": 0, "end": 2, "topic": "econ"}, '
        llm_response += '{"start": 3, "end": 5, "topic": "tech"}]}'

        with patch("distillcore.llm.async_client.get_async_client") as mock_get:
            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(
                return_value=self._mock_response(llm_response)
            )
            mock_get.return_value = mock_client

            result = await achunk(text, strategy="llm", api_key="test-key")
            assert len(result) == 2

    @pytest.mark.asyncio
    async def test_async_non_llm(self) -> None:
        text = "Para one.\n\nPara two."
        result = await achunk(text, strategy="paragraph")
        assert len(result) >= 1


class TestMinTokens:
    def test_merge_disabled(self) -> None:
        text = "\n\n".join(["Hi.", "Ok.", "Sure.", "A longer paragraph."])
        result_no = chunk(text, min_tokens=0)
        result_yes = chunk(text, min_tokens=50)
        assert len(result_yes) <= len(result_no)

    def test_merge_combines_small(self) -> None:
        text = "\n\n".join(["x" * 10, "y" * 10, "z" * 400])
        result = chunk(text, min_tokens=20, target_tokens=500)
        # First two tiny chunks should merge
        assert any("x" in c and "y" in c for c in result)


class TestCustomTokenizer:
    def test_word_tokenizer(self) -> None:
        # With a word tokenizer, target_tokens=5 means ~5 words per chunk
        text = " ".join([f"word{i}" for i in range(50)])
        result = chunk(
            text,
            strategy="sentence",
            target_tokens=10,
            tokenizer=lambda t: len(t.split()),
        )
        assert len(result) >= 1


class TestUnknownStrategy:
    def test_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            chunk("text", strategy="nonexistent")

    @pytest.mark.asyncio
    async def test_raises_async(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            await achunk("text", strategy="nonexistent")
