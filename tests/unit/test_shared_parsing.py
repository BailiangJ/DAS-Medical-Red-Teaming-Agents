import pytest

from med_red_team.shared.parsing import (
    extract_json_object,
    parse_bool,
    parse_json_dict,
    parse_typed_json,
)


def test_extracts_json_from_fenced_or_surrounding_text():
    assert parse_json_dict('```json\n{"ok": true}\n```') == {"ok": True}
    assert parse_json_dict('Result: {"text": "brace } inside", "n": 2} done') == {
        "text": "brace } inside",
        "n": 2,
    }


def test_strict_boolean_parsing():
    assert parse_bool(True) is True
    assert parse_bool("false") is False
    with pytest.raises(ValueError):
        parse_bool("yes")
    with pytest.raises(ValueError):
        parse_bool(1)


def test_typed_validation_remains_caller_owned():
    result = parse_typed_json('{"value": 3}', lambda data: data["value"] * 2)
    assert result == 6


def test_rejects_missing_or_unterminated_object():
    with pytest.raises(ValueError):
        extract_json_object("no object")
    with pytest.raises(ValueError):
        extract_json_object('{"broken": true')
