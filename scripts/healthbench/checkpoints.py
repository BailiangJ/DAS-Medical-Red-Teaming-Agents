"""HealthBench-local checkpoint and resume semantics."""

from collections.abc import Iterable
from typing import Any, TypeVar

from med_red_team.healthbench.data import (
    HEALTHBENCH_SCHEMA_VERSION,
    _healthbench_generation_config,
    _healthbench_model_id,
    _healthbench_testee_system_prompt,
    healthbench_attack_strategy_config,
    healthbench_attack_strategy_order,
    healthbench_dataset_effective_sha,
    healthbench_filter_single_turn,
    healthbench_result_completeness_error,
    healthbench_source_results_sha,
)
from med_red_team.shared.checkpoints import merge_unique, validate_resume_metadata


ResultT = TypeVar("ResultT")
_UNSET = object()


def healthbench_case_id(item: Any) -> str:
    """Return the stable HealthBench case identity from a dict or result object."""
    case_id = item.get("test_case_id") if isinstance(item, dict) else getattr(item, "test_case_id", None)
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"HealthBench item is missing test_case_id: {item!r}")
    return case_id


def set_healthbench_sample_number(item: Any, sample_number: int) -> None:
    """Write sample_number on a dict or result object."""
    if isinstance(item, dict):
        item["sample_number"] = sample_number
    else:
        item.sample_number = sample_number


def merge_healthbench_results(
    existing: Iterable[ResultT],
    current: Iterable[ResultT],
) -> list[ResultT]:
    """Merge by case ID and keep sample numbers cumulative on resume."""
    combined = merge_unique(
        existing,
        current,
        key=healthbench_case_id,
    )
    for sample_number, result in enumerate(combined, start=1):
        set_healthbench_sample_number(result, sample_number)
    return combined


def validate_healthbench_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    phase: str,
    expected_testee_model: Any = _UNSET,
    expected_testee_config: Any = _UNSET,
    expected_testee_system_prompt: Any = _UNSET,
    expected_grader_model: Any = _UNSET,
    expected_grader_config: Any = _UNSET,
    expected_dataset_effective_sha256: Any = _UNSET,
    expected_source_results_sha256: Any = _UNSET,
    expected_config_sample_limit: Any = _UNSET,
    expected_dataset_sample_limit: Any = _UNSET,
    expected_filter_single_turn: Any = _UNSET,
    expected_attack_strategies: Any = _UNSET,
    expected_attack_strategy_config: Any = _UNSET,
    expected_extra: dict[str, Any] | None = None,
) -> None:
    """Require a checkpoint to match one resolved HealthBench stage contract."""
    if not isinstance(metadata, dict):
        raise ValueError("HealthBench checkpoint metadata must be a JSON object")

    validate_resume_metadata(
        {
            "schema_version": HEALTHBENCH_SCHEMA_VERSION,
            "axis": "healthbench",
            "phase": phase,
        },
        metadata,
    )

    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("HealthBench checkpoint metadata is missing config")
    dataset = metadata.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("HealthBench checkpoint metadata is missing dataset")

    checks = (
        ("testee model", _healthbench_model_id(metadata, "testee"), expected_testee_model),
        (
            "testee generation config",
            _healthbench_generation_config(metadata, "testee"),
            expected_testee_config,
        ),
        (
            "testee system prompt",
            _healthbench_testee_system_prompt(metadata),
            expected_testee_system_prompt,
        ),
        ("grader model", _healthbench_model_id(metadata, "grader"), expected_grader_model),
        (
            "grader generation config",
            _healthbench_generation_config(metadata, "grader"),
            expected_grader_config,
        ),
        (
            "dataset content",
            healthbench_dataset_effective_sha(metadata),
            expected_dataset_effective_sha256,
        ),
        (
            "source artifact content",
            healthbench_source_results_sha(metadata),
            expected_source_results_sha256,
        ),
        ("stage sample limit", config.get("max_samples"), expected_config_sample_limit),
        (
            "source population sample limit",
            dataset.get("sample_limit"),
            expected_dataset_sample_limit,
        ),
        (
            "single-turn filter",
            healthbench_filter_single_turn(metadata),
            expected_filter_single_turn,
        ),
        (
            "attack strategies",
            healthbench_attack_strategy_order(metadata),
            expected_attack_strategies,
        ),
        (
            "attack strategy config",
            healthbench_attack_strategy_config(metadata),
            expected_attack_strategy_config,
        ),
    )

    mismatches = [
        f"{label}: checkpoint has {actual!r}, expected {expected!r}"
        for label, actual, expected in checks
        if expected is not _UNSET and actual != expected
    ]
    for key, expected in (expected_extra or {}).items():
        actual = metadata.get(key)
        if actual != expected:
            mismatches.append(
                f"{key}: checkpoint has {actual!r}, expected {expected!r}"
            )

    if mismatches:
        raise ValueError(
            "Checkpoint is incompatible with this run (" + "; ".join(mismatches) + ")"
        )


def validate_healthbench_resume_rows(
    items: Iterable[Any],
    *,
    allowed_case_ids: Iterable[str],
) -> set[str]:
    """Reject duplicate, out-of-population, or internally incomplete rows in a
    resumed HealthBench stage."""
    allowed = set(allowed_case_ids)
    seen: set[str] = set()
    duplicates: set[str] = set()
    out_of_scope: set[str] = set()
    incomplete: dict[str, str] = {}

    for item in items:
        case_id = healthbench_case_id(item)
        if case_id in seen:
            duplicates.add(case_id)
        seen.add(case_id)
        if case_id not in allowed:
            out_of_scope.add(case_id)
        completeness_error = healthbench_result_completeness_error(item)
        if completeness_error is not None:
            incomplete[case_id] = completeness_error

    if duplicates:
        raise ValueError(
            "HealthBench checkpoint contains duplicate test_case_id values: "
            + ", ".join(sorted(duplicates))
        )
    if out_of_scope:
        raise ValueError(
            "HealthBench checkpoint contains rows outside the current source population: "
            + ", ".join(sorted(out_of_scope))
        )
    if incomplete:
        details = "; ".join(
            f"{case_id} ({message})" for case_id, message in sorted(incomplete.items())
        )
        raise ValueError(
            "HealthBench checkpoint contains rows with incomplete or inconsistent "
            f"grading data: {details}"
        )
    return seen
