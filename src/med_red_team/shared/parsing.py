"""Shared mechanics for parsing structured LLM output."""

import json
import math
import re
from typing import Any, Callable, TypeVar


ParsedT = TypeVar("ParsedT")


class StructuredOutputError(ValueError):
    """Raised when model output cannot satisfy the requested structure."""


def strip_markdown_fences(text: str) -> str:
    """Remove one surrounding Markdown code fence when present."""
    value = text.strip()
    match = re.fullmatch(r"```(?:[A-Za-z0-9_+-]+)?\s*\n?(.*?)\n?```", value, re.DOTALL)
    return match.group(1).strip() if match else value


def extract_json_object(text: str) -> str:
    """Extract the first balanced JSON object, respecting quoted strings."""
    value = strip_markdown_fences(text)
    start = value.find("{")
    if start < 0:
        raise StructuredOutputError("No JSON object found in model response")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start:index + 1]
    raise StructuredOutputError("Unterminated JSON object in model response")


def parse_json_dict(text: str) -> dict[str, Any]:
    try:
        value = json.loads(extract_json_object(text))
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"Invalid JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise StructuredOutputError(
            f"Expected a JSON object, got {type(value).__name__}"
        )
    return value


def parse_bool(value: Any) -> bool:
    """Parse only real booleans or explicit true/false strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise StructuredOutputError(
        f"Expected boolean true/false, got {value!r}"
    )


def parse_string(value: Any, *, non_empty: bool = False) -> str:
    """Require a real string rather than coercing arbitrary JSON values."""
    if not isinstance(value, str):
        raise StructuredOutputError(f"Expected string, got {value!r}")
    if non_empty and not value.strip():
        raise StructuredOutputError("Expected a non-empty string")
    return value


def parse_int(value: Any) -> int:
    """Require a JSON integer while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise StructuredOutputError(f"Expected integer, got {value!r}")
    return value


def parse_finite_float(value: Any) -> float:
    """Require a finite real number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StructuredOutputError(f"Expected finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise StructuredOutputError(f"Expected finite number, got {value!r}")
    return result


def parse_typed_json(text: str, validator: Callable[[dict[str, Any]], ParsedT]) -> ParsedT:
    """Parse shared JSON mechanics, then delegate domain validation locally."""
    return validator(parse_json_dict(text))
