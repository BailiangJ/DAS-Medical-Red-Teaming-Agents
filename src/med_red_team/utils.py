"""Small framework utilities retained where multiple axes genuinely reuse them."""

from collections.abc import Callable
from typing import Any, TypeVar

from tqdm import tqdm

from med_red_team.shared.io import atomic_write_json
from med_red_team.shared.parsing import parse_bool, parse_json_dict, strip_markdown_fences


ResultT = TypeVar("ResultT")


def clean_markdown_response(response_text: str) -> str:
    return strip_markdown_fences(response_text)


def normalize_bool(value: Any) -> bool:
    return parse_bool(value)


def _convert_value(value: Any, expected_type: type) -> Any:
    if expected_type is bool:
        return parse_bool(value)
    if isinstance(value, expected_type):
        return value
    return expected_type(value)


def parse_json_response(
    response_text: str,
    expected_fields: dict[str, type] | None = None,
    fallback: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Parse an LLM JSON object and optionally validate selected fields."""
    try:
        data = parse_json_dict(response_text)
        if expected_fields is None:
            return data

        result: dict[str, Any] = {}
        for field, expected_type in expected_fields.items():
            if field not in data or data[field] is None:
                if fallback is not None and field in fallback:
                    result[field] = fallback[field]
                    continue
                raise ValueError(f"Missing required field: {field}")
            try:
                result[field] = _convert_value(data[field], expected_type)
            except (TypeError, ValueError) as exc:
                if fallback is not None and field in fallback:
                    result[field] = fallback[field]
                else:
                    raise ValueError(
                        f"Invalid value for {field!r}: expected {expected_type.__name__}"
                    ) from exc
        return result
    except (TypeError, ValueError) as exc:
        if verbose:
            print(f"[WARNING] Failed to parse structured response: {exc}")
        if fallback is not None:
            return dict(fallback)
        raise


def parse_attacker_response(
    response_text: str,
    fallback_prompt: str,
    verbose: bool = False,
) -> tuple[str, bool]:
    fallback = {"modified_prompt": fallback_prompt, "is_modified": False}
    result = parse_json_response(
        response_text,
        expected_fields={"modified_prompt": str, "is_modified": bool},
        fallback=fallback,
        verbose=verbose,
    )
    modified_prompt = result["modified_prompt"].strip()
    if not modified_prompt:
        return fallback_prompt, False
    return modified_prompt, result["is_modified"]


def parse_grader_response(
    response_text: str,
    verbose: bool = False,
) -> tuple[float, str]:
    try:
        result = parse_json_response(
            response_text,
            expected_fields={"violation": float, "reason": str},
            verbose=verbose,
        )
        violation = result["violation"]
        reason = result["reason"].strip() or "No reason provided by grader."
        if violation not in {0.0, 0.5, 1.0}:
            return 0.5, f"Invalid score ({violation}). {reason}"
        return violation, reason
    except (TypeError, ValueError) as exc:
        return 0.5, f"Parse error: unable to determine violation status ({exc})"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def evaluate_items_fail_fast(
    items: list[Any],
    evaluate_fn: Callable[[Any, int], ResultT],
    description: str = "Processing",
    progress_callback: Callable[[list[ResultT]], None] | None = None,
    save_every: int = 10,
    verbose: bool = True,
    use_tqdm: bool = True,
    stop_fn: Callable[[int, list[ResultT]], bool] | None = None,
) -> list[ResultT]:
    """Evaluate sequentially, checkpoint completed results, and re-raise errors."""
    results: list[ResultT] = []
    iterator = enumerate(tqdm(items, desc=description, disable=not use_tqdm))
    try:
        for index, item in iterator:
            if stop_fn and stop_fn(index, results):
                break
            try:
                results.append(evaluate_fn(item, index))
                if progress_callback and len(results) % save_every == 0:
                    progress_callback(results)
            except Exception:
                if verbose:
                    print(
                        f"\n[ERROR] Evaluation failed at item {index + 1}; "
                        f"checkpointing {len(results)} completed result(s)."
                    )
                if progress_callback:
                    progress_callback(results)
                raise
    except KeyboardInterrupt:
        if progress_callback:
            progress_callback(results)
        raise
    return results


def update_attack_metadata(
    test_case: Any,
    strategy_name: str,
    details: dict[str, Any],
) -> None:
    """Accumulate strategy metadata without defining axis-specific semantics."""
    test_case.metadata.setdefault("attacks_applied", []).append(strategy_name)
    test_case.metadata.setdefault("attack_details", {})[strategy_name] = details
    if "modified_prompt" in details:
        test_case.metadata["modified_prompt"] = details["modified_prompt"]
    if "manipulation_failed" in details:
        test_case.metadata["manipulation_failed"] = details["manipulation_failed"]
    for control_field in ("failure_category", "reason", "status"):
        if control_field in details:
            test_case.metadata[control_field] = details[control_field]


__all__ = [
    "atomic_write_json",
    "clean_markdown_response",
    "evaluate_items_fail_fast",
    "normalize_bool",
    "parse_attacker_response",
    "parse_grader_response",
    "parse_json_response",
    "truncate_text",
    "update_attack_metadata",
]
