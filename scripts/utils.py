"""Runner-facing helpers that do not own axis semantics."""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from med_red_team.shared.config_loading import load_config
from med_red_team.shared.io import read_json
from med_red_team.shared.io import validate_output_path as _validate_output_path


def load_config_from_py(path: str, expected_type: type):
    return load_config(path, expected_type)


def generate_output_path(
    log_dir: str,
    testee_model: str,
    mode: str | None,
    strategies: list[str] | None = None,
    max_samples: int | None = None,
    timestamp: str | None = None,
    suffix: str = "",
) -> Path:
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [testee_model.replace("/", "_").replace(":", "_")]
    if mode:
        parts.append(mode)
    if strategies:
        parts.append("_".join(strategies))
    parts.append("all" if max_samples is None else str(max_samples))
    parts.append(timestamp)
    output_dir = Path(log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / ("_".join(parts) + suffix + ".json")


def validate_output_path(
    output_path: Path,
    quiet: bool = False,
    overwrite: bool = False,
) -> bool:
    try:
        _validate_output_path(output_path, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        if not quiet:
            print(f"[ERROR] Invalid output path {output_path}: {exc}")
        return False
    if not quiet:
        print(f"  Output will be saved to: {output_path}")
    return True


def generate_and_validate_output_path(**kwargs) -> Path | None:
    quiet = kwargs.pop("quiet", False)
    path = generate_output_path(**kwargs)
    return path if validate_output_path(path, quiet=quiet) else None


def create_attack_strategies_from_config(
    config,
    model_pool,
    strategy_getter=None,
    *,
    verbose: bool | None = None,
) -> list:
    if strategy_getter is None:
        from med_red_team.attacker_registry import get_strategy

        strategy_getter = get_strategy

    configured_strategies = (
        config.resolved_attacker_strategies()
        if hasattr(config, "resolved_attacker_strategies")
        else config.attacker_strategies
    )
    strategies = []
    for name, configured_kwargs in configured_strategies.items():
        strategy_kwargs = dict(configured_kwargs)
        if hasattr(config, "attacker_model"):
            strategy_kwargs.setdefault("model_id", config.attacker_model)
        if hasattr(config, "attacker_config"):
            strategy_kwargs.setdefault("config", config.attacker_config)
        if verbose is not None and name in {
            "race_socioeconomic_label",
            "language_manipulation",
            "emotion_manipulation",
            "cognitive_bias",
        }:
            strategy_kwargs.setdefault("verbose", verbose)
        strategies.append(
            strategy_getter(
                strategy_name=name,
                model_pool=model_pool,
                **strategy_kwargs,
            )
        )
    return strategies


def load_results_from_json(
    file_path: Path,
    data_key: str = "results",
    quiet: bool = False,
    allow_missing: bool = False,
    reconstruct_fn: Callable[[dict[str, Any]], Any] | None = None,
):
    path = Path(file_path)
    if not path.exists():
        if allow_missing:
            return [], {}
        raise FileNotFoundError(f"Results file not found: {path}")

    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Result file must contain a JSON object: {path}")
    if data_key not in data:
        raise ValueError(f"Result file is missing required key {data_key!r}: {path}")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"Result metadata must be a JSON object: {path}")
    items = data[data_key]
    if not isinstance(items, list):
        raise ValueError(f"Result key {data_key!r} must contain a list: {path}")
    results = [reconstruct_fn(item) for item in items] if reconstruct_fn else items
    if not quiet:
        print(f"  Loaded {len(results)} {data_key} from: {path}")
    return results, metadata


def validate_resume(
    resume_metadata: dict,
    validations: dict[str, tuple[Any, str]],
    quiet: bool = False,
):
    errors = []
    for metadata_key, (expected_value, label) in validations.items():
        if expected_value is None:
            continue
        actual_value = resume_metadata.get(metadata_key)
        if actual_value != expected_value:
            errors.append(
                f"{label} mismatch: resume has {actual_value!r}, expected {expected_value!r}"
            )
    if errors:
        return False, "; ".join(errors)
    if not quiet:
        print("  Resume validation passed")
    return True, "Settings match"


def override_attacker_strategies(
    config,
    args_strategies: list[str] | None,
    args_attacker_model: str | None,
    default_config,
    quiet: bool = False,
):
    if not args_strategies and not args_attacker_model:
        return None

    fallback = getattr(default_config, "strategy_catalog", default_config.attacker_strategies)
    selected = args_strategies or list(config.attacker_strategies)
    strategies = {}
    for name in selected:
        if name in config.attacker_strategies:
            strategy_kwargs = dict(config.attacker_strategies[name])
        elif name in fallback:
            strategy_kwargs = dict(fallback[name])
            if not quiet:
                print(f"[Config] Using default strategy config for: {name}")
        else:
            raise ValueError(
                f"Unknown strategy {name!r}; available strategies: {sorted(fallback)}"
            )
        if args_attacker_model:
            strategy_kwargs["model_id"] = args_attacker_model
        strategies[name] = strategy_kwargs
    return strategies


def reconstruct_healthbench_result(data: dict):
    from med_red_team.healthbench.data import HealthBenchRobustnessResult

    return HealthBenchRobustnessResult.from_dict(data)
