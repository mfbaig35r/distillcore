"""Tests for distillcore.llm.json_repair."""

from distillcore.llm.json_repair import safe_parse, try_fix_truncated_json


class TestSafeParse:
    def test_valid_json(self) -> None:
        assert safe_parse('{"key": "value"}') == {"key": "value"}

    def test_empty_object(self) -> None:
        assert safe_parse("{}") == {}

    def test_invalid_returns_empty(self) -> None:
        assert safe_parse("not json at all") == {}

    def test_truncated_repairs(self) -> None:
        # Truncated JSON that can be repaired
        result = safe_parse('{"sections": [{"heading": "test"')
        assert isinstance(result, dict)

    def test_empty_string(self) -> None:
        assert safe_parse("") == {}


class TestTryFixTruncatedJson:
    def test_unbalanced_braces(self) -> None:
        fixed = try_fix_truncated_json('{"a": {"b": 1}')
        assert fixed.endswith("}")

    def test_unbalanced_brackets(self) -> None:
        fixed = try_fix_truncated_json('{"items": [1, 2')
        assert "]" in fixed
        assert fixed.endswith("}")

    def test_odd_quotes(self) -> None:
        fixed = try_fix_truncated_json('{"key": "value')
        assert fixed.count('"') % 2 == 0

    def test_already_valid(self) -> None:
        result = try_fix_truncated_json('{"ok": true}')
        assert result == '{"ok": true}'

    def test_nested_truncation(self) -> None:
        raw = '{"a": [1, 2'
        fixed = try_fix_truncated_json(raw)
        import json

        parsed = json.loads(fixed)
        assert "a" in parsed
