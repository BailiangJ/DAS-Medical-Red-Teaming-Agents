"""Public HealthBench evaluation API.

The axis modules reference one another (for example, the pipeline uses the
rubric data and grader), so these exports are resolved lazily.  This keeps
``import med_red_team.healthbench`` provider-free and avoids import-order
cycles while keeping the package-level API focused on user-facing components.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Attack strategies.
    "AdjustImpossibleMeasurementStrategy": ("attacker", "AdjustImpossibleMeasurementStrategy"),
    "CognitiveBiasStrategy": ("attacker", "CognitiveBiasStrategy"),
    "DistractionSentenceStrategy": ("attacker", "DistractionSentenceStrategy"),
    # Configuration.
    "HealthBenchConfig": ("config", "HealthBenchConfig"),
    # Data structures.
    "RubricItem": ("data", "RubricItem"),
    "RubricGradeResult": ("data", "RubricGradeResult"),
    "HealthBenchTestCase": ("data", "HealthBenchTestCase"),
    "AttackResult": ("data", "AttackResult"),
    "HealthBenchRobustnessResult": ("data", "HealthBenchRobustnessResult"),
    "HealthBenchRobustnessSummary": ("data", "HealthBenchRobustnessSummary"),
    # Grading and pipeline.
    "RubricGrader": ("grader", "RubricGrader"),
    "HealthBenchConversationAttack": ("pipeline", "HealthBenchConversationAttack"),
    "HealthBenchRobustnessPipeline": ("pipeline", "HealthBenchRobustnessPipeline"),
    # Provider-free dataset and metric utilities.
    "load_healthbench_jsonl": ("utils", "load_healthbench_jsonl"),
    "filter_single_turn_cases": ("utils", "filter_single_turn_cases"),
    "calculate_score": ("utils", "calculate_score"),
    "calculate_tag_scores": ("utils", "calculate_tag_scores"),
    "compute_bootstrap_std": ("utils", "compute_bootstrap_std"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from its axis module on first access."""
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    symbol = getattr(import_module(f"{__name__}.{module_name}"), symbol_name)
    globals()[name] = symbol
    return symbol


def __dir__() -> list[str]:
    """Include lazy public exports in interactive discovery tools."""
    return sorted(set(globals()) | set(__all__))
