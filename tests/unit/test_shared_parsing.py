import pytest

from med_red_team.shared.parsing import (
    StructuredOutputError,
    extract_json_object,
    parse_bool,
    parse_finite_float,
    parse_int,
    parse_json_dict,
    parse_string,
    parse_typed_json,
)
from med_red_team.utils import parse_json_response


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
    with pytest.raises(StructuredOutputError):
        extract_json_object("no object")
    with pytest.raises(StructuredOutputError):
        extract_json_object('{"broken": true')


def test_strict_primitive_parsers_reject_python_coercions():
    assert parse_string("answer") == "answer"
    assert parse_int(3) == 3
    assert parse_finite_float(3) == 3.0

    with pytest.raises(StructuredOutputError):
        parse_string(["answer"])
    with pytest.raises(StructuredOutputError):
        parse_int(True)
    with pytest.raises(StructuredOutputError):
        parse_finite_float(False)
    with pytest.raises(StructuredOutputError):
        parse_finite_float(float("nan"))
    with pytest.raises(StructuredOutputError):
        parse_finite_float(float("inf"))


def test_json_field_validation_does_not_coerce_malformed_values():
    with pytest.raises(StructuredOutputError):
        parse_json_response(
            '{"modified_prompt": ["not", "a", "string"]}',
            expected_fields={"modified_prompt": str},
        )
    with pytest.raises(StructuredOutputError):
        parse_json_response(
            '{"violation": true}',
            expected_fields={"violation": float},
        )
