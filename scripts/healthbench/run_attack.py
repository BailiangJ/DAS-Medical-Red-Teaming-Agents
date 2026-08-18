#!/usr/bin/env python3
"""
HealthBench Attack Evaluation
==============================

Apply attack strategies to HealthBench test cases to test model robustness.
Configure exactly one attack strategy per run; run each strategy separately.

Usage:
    # Basic usage with config file (recommended)
    python -m scripts.healthbench.run_attack \
        --config configs/examples/healthbench/impossible_measurement.py \
        --dataset test/healthbench_dataset/consensus.jsonl

    # With baseline results (for distraction attack)
    python -m scripts.healthbench.run_attack \
        --config configs/examples/healthbench/distraction.py \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --baseline-results logs/gpt-4o_baseline_20250119_143022.json

    # Override config with command-line args
    python -m scripts.healthbench.run_attack \
        --config configs/examples/healthbench/impossible_measurement.py \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --testee-model gpt-4o \
        --max-samples 50

Example:
    >>> python -m scripts.healthbench.run_attack \
    ...     --config configs/examples/healthbench/impossible_measurement.py \
    ...     --dataset test/healthbench_dataset/consensus.jsonl

    Results saved to: logs/o3_impossible_measurement_2_20250127_150322.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from med_red_team.testee import Testee
from med_red_team.model_pool import ModelPool
from med_red_team.healthbench.grader import RubricGrader
from med_red_team.healthbench.pipeline import HealthBenchRobustnessPipeline
from med_red_team.healthbench.utils import load_healthbench_jsonl
from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.healthbench.attacker import (
    DistractionSentenceStrategy,
    AdjustImpossibleMeasurementStrategy,
    CognitiveBiasStrategy,
)
from med_red_team.healthbench.data import (
    HealthBenchRobustnessResult,
    build_healthbench_metadata,
    healthbench_attack_strategy_config,
    healthbench_attack_strategy_order,
    healthbench_dataset_effective_sha,
    healthbench_effective_dataset_identity,
    healthbench_source_artifact_identity,
    healthbench_source_results_sha,
    healthbench_dataset_path,
    healthbench_filter_single_turn,
    _healthbench_generation_config,
    _healthbench_model_id,
    _healthbench_testee_system_prompt,
    validate_cached_baseline_result,
    validate_healthbench_artifact_metadata,
)
from med_red_team.models import GenerationConfig
from med_red_team.shared.checkpoints import checkpoint_path, remove_checkpoint
from med_red_team.shared.config_loading import resolve_sample_limit
from scripts.healthbench.checkpoints import (
    merge_healthbench_results,
    validate_healthbench_resume_rows,
)
from scripts.utils import (
    load_config_from_py,
    generate_and_validate_output_path,
    load_results_from_json,
    reconstruct_healthbench_result
)


_UNSET = object()


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HealthBench attack evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Each run must configure exactly one attack strategy. Run separate commands for
Cognitive Bait, Narrative Distraction, and Physiological Impossibility.

Examples:
  # Run with config file (recommended)
  python -m scripts.healthbench.run_attack \\
      --config configs/examples/healthbench/impossible_measurement.py \\
      --dataset test/healthbench_dataset/consensus.jsonl

  # With baseline results for distraction attack
  python -m scripts.healthbench.run_attack \\
      --config configs/examples/healthbench/distraction.py \\
      --dataset test/healthbench_dataset/consensus.jsonl \\
      --baseline-results logs/healthbench/gpt-4o/baseline_20250119_143022.json

  # Override config settings
  python -m scripts.healthbench.run_attack \\
      --config configs/examples/healthbench/impossible_measurement.py \\
      --dataset test/healthbench_dataset/consensus.jsonl \\
      --testee-model gpt-4o \\
      --max-samples 50
        """
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config .py file (e.g., configs/examples/healthbench/impossible_measurement.py)"
    )

    # Data settings
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HealthBench dataset (JSONL file) (overrides config)"
    )

    # Baseline caching (optional, needed for distraction attack)
    parser.add_argument(
        "--baseline-results",
        type=str,
        help="Path to baseline results JSON (optional, for caching baseline)"
    )

    # Model settings (override config)
    parser.add_argument(
        "--testee-model",
        type=str,
        default=None,
        help="Model ID for the testee (overrides config)"
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help="Model ID for rubric grading (overrides config)"
    )

    # Evaluation settings (override config)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test cases to attack (overrides config)"
    )
    parser.add_argument(
        "--filter-single-turn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to filter to single-turn conversations (overrides config)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)"
    )

    # Display settings
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    # Resume settings
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing results file to resume from"
    )

    return parser.parse_args()


# ==============================================================================
# Configuration Setup
# ==============================================================================

def create_config_from_args(args: argparse.Namespace) -> HealthBenchConfig:
    """
    Create HealthBenchConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        HealthBenchConfig instance
    """
    # Load config file
    if not args.quiet:
        print(f"[Config] Loading from: {args.config}")

    # Load the explicit CONFIG object from the selected preset
    config = load_config_from_py(args.config, HealthBenchConfig)

    # Override with command-line arguments
    overrides = {
        "max_samples": resolve_sample_limit(args.max_samples, config.max_samples),
    }

    # Data settings
    if args.dataset is not None:
        overrides["dataset_path"] = args.dataset

    # Model settings
    if args.testee_model is not None:
        overrides["testee_model"] = args.testee_model
    if args.grader_model is not None:
        overrides["grader_model"] = args.grader_model

    # Evaluation settings
    if args.filter_single_turn is not None:
        overrides["filter_single_turn"] = args.filter_single_turn

    # Output settings
    if args.output_dir is not None:
        overrides["log_dir"] = args.output_dir

    # Create new config with overrides
    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = HealthBenchConfig.from_dict(config_dict)

    return config


def _attack_dataset_metadata(config: HealthBenchConfig, dataset_identity: dict | None = None) -> dict:
    metadata = {
        "path": config.dataset_path,
        "sample_limit": config.max_samples,
        "filter_single_turn": config.filter_single_turn,
    }
    if dataset_identity:
        metadata.update({
            "canonical_path": dataset_identity.get("canonical_path"),
            "file_sha256": dataset_identity.get("file_sha256"),
            "effective_rows": dataset_identity.get("effective_rows"),
            "effective_sha256": dataset_identity.get("effective_sha256"),
        })
    return metadata



def _attack_source_metadata(
    config: HealthBenchConfig,
    baseline_results_file: str | None,
    dataset_identity: dict | None = None,
    baseline_identity: dict | None = None,
) -> dict:
    metadata = {
        "dataset_path": config.dataset_path,
        "workflow": "run_attack",
        "baseline_results_file": str(baseline_results_file) if baseline_results_file else None,
    }
    if dataset_identity:
        metadata.update({
            "dataset_file_sha256": dataset_identity.get("file_sha256"),
            "dataset_effective_sha256": dataset_identity.get("effective_sha256"),
        })
    if baseline_identity:
        metadata.update(baseline_identity)
        if baseline_identity.get("source_results_sha256") is not None:
            metadata["baseline_results_sha256"] = baseline_identity["source_results_sha256"]
    return metadata



def _make_metadata(
    config: HealthBenchConfig,
    *,
    is_partial: bool,
    baseline_results_file: str | None,
    total_cases_evaluated: int | None = None,
    dataset_identity: dict | None = None,
    baseline_identity: dict | None = None,
) -> dict:
    extra = {
        "total_cases_evaluated": total_cases_evaluated,
        "bootstrap_seed": 0,
        "bootstrap_clips_mean_to_unit_interval": True,
    }
    return build_healthbench_metadata(
        phase="attack",
        config=config,
        is_partial=is_partial,
        dataset=_attack_dataset_metadata(config, dataset_identity),
        source=_attack_source_metadata(config, baseline_results_file, dataset_identity, baseline_identity),
        attack_strategies=list(config.attacker_strategies.keys()),
        baseline_results_file=baseline_results_file,
        extra=extra,
    )



def load_baseline_results(
    baseline_path: str,
    quiet: bool = False,
    *,
    expected_testee_model: str | None = None,
    expected_testee_config: dict | None = None,
    expected_testee_system_prompt: str | None = None,
    expected_grader_model: str | None = None,
    expected_grader_config: dict | None = None,
):
    """Load baseline results from JSON file for baseline-dependent attacks."""
    if not baseline_path:
        return None, None

    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline results file not found: {baseline_path}")

    if not quiet:
        print(f"[Loading Baseline Results]")
        print(f"  From: {baseline_path}")

    results, metadata = load_results_from_json(
        baseline_path,
        data_key="results",
        quiet=quiet,
        reconstruct_fn=reconstruct_healthbench_result
    )
    validate_healthbench_artifact_metadata(
        metadata,
        phase="baseline",
        label=f"HealthBench baseline input {baseline_path}",
        expected_testee_model=expected_testee_model,
        expected_testee_config=expected_testee_config,
        expected_testee_system_prompt=expected_testee_system_prompt,
        expected_grader_model=expected_grader_model,
        expected_grader_config=expected_grader_config,
    )

    return results, metadata


def validate_resume_file(
    resume_path: Path,
    testee_model: str,
    grader_model: str,
    attack_strategies: List[str],
    attacker_strategy_config: Dict[str, Dict[str, Any]],
    dataset_path: str,
    filter_single_turn: bool,
    dataset_effective_sha256: str | None = None,
    baseline_results_sha256: str | None = None,
    *,
    testee_config: dict | None = None,
    testee_system_prompt: str | None = None,
    grader_config: dict | None = None,
    max_samples: Any = _UNSET,
    require_baseline_sha: bool = False,
) -> tuple:
    """
    Validate resume file settings match current settings.

    Args:
        resume_path: Path to resume file
        testee_model: Current testee model ID
        grader_model: Current grader model ID
        attack_strategies: Current attack strategy names in execution order
        attacker_strategy_config: Serialized configuration for each attack strategy
        dataset_path: Current HealthBench dataset path
        filter_single_turn: Current single-turn filtering setting

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        with open(resume_path, 'r') as f:
            data = json.load(f)

        metadata = data.get('metadata', {})
        errors = []
        file_testee_model = _healthbench_model_id(metadata, "testee")
        file_grader_model = _healthbench_model_id(metadata, "grader")
        file_filter_single_turn = healthbench_filter_single_turn(metadata)

        if metadata.get('axis') not in {None, 'healthbench'}:
            errors.append(f"Axis mismatch: file has '{metadata.get('axis')}', expected 'healthbench'")
        if metadata.get('phase') not in {None, 'attack'}:
            errors.append(f"Phase mismatch: file has '{metadata.get('phase')}', expected 'attack'")

        if file_testee_model != testee_model:
            errors.append(f"Testee model mismatch: file has '{file_testee_model}', current is '{testee_model}'")
        if file_grader_model != grader_model:
            errors.append(f"Grader model mismatch: file has '{file_grader_model}', current is '{grader_model}'")

        file_testee_config = _healthbench_generation_config(metadata, "testee")
        if testee_config is not None and file_testee_config != testee_config:
            errors.append(
                f"Testee config mismatch: file has {file_testee_config!r}, current is {testee_config!r}"
            )
        file_testee_system_prompt = _healthbench_testee_system_prompt(metadata)
        if testee_system_prompt is not None and file_testee_system_prompt != testee_system_prompt:
            errors.append(
                "Testee system prompt mismatch: "
                f"file has {file_testee_system_prompt!r}, current is {testee_system_prompt!r}"
            )
        file_grader_config = _healthbench_generation_config(metadata, "grader")
        if grader_config is not None and file_grader_config != grader_config:
            errors.append(
                f"Grader config mismatch: file has {file_grader_config!r}, current is {grader_config!r}"
            )
        file_config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        if max_samples is not _UNSET and file_config.get("max_samples") != max_samples:
            errors.append(
                "Sample limit mismatch: "
                f"file has {file_config.get('max_samples')!r}, current is {max_samples!r}"
            )

        file_strategies = healthbench_attack_strategy_order(metadata)
        if file_strategies != attack_strategies:
            errors.append(f"Attack strategies mismatch: file has {file_strategies}, current is {attack_strategies}")

        file_attacker_config = healthbench_attack_strategy_config(metadata)
        if file_attacker_config != attacker_strategy_config:
            errors.append(
                f"Attack strategy config mismatch: file has {file_attacker_config}, "
                f"current is {attacker_strategy_config}"
            )

        file_dataset_sha = healthbench_dataset_effective_sha(metadata)
        if file_dataset_sha:
            if dataset_effective_sha256 and file_dataset_sha != dataset_effective_sha256:
                errors.append(
                    "Dataset content mismatch: "
                    f"file has {file_dataset_sha}, current is {dataset_effective_sha256}"
                )
        else:
            file_dataset = healthbench_dataset_path(metadata)
            if not file_dataset:
                errors.append("Resume file is missing dataset metadata")
            elif Path(file_dataset).resolve() != Path(dataset_path).resolve():
                errors.append(f"Dataset mismatch: file has '{file_dataset}', current is '{dataset_path}'")

        file_baseline_sha = healthbench_source_results_sha(metadata)
        if require_baseline_sha and baseline_results_sha256 and not file_baseline_sha:
            errors.append("Resume file is missing baseline results content identity")
        if baseline_results_sha256 and file_baseline_sha and file_baseline_sha != baseline_results_sha256:
            errors.append(
                "Baseline results content mismatch: "
                f"file has {file_baseline_sha}, current is {baseline_results_sha256}"
            )

        if file_filter_single_turn != filter_single_turn:
            errors.append(
                f"Single-turn filter mismatch: file has {file_filter_single_turn}, "
                f"current is {filter_single_turn}"
            )

        if errors:
            return False, "; ".join(errors)
        return True, "Settings match"

    except Exception as e:
        return False, f"Error reading file: {e}"


def create_attack_strategies(
    config: HealthBenchConfig,
    model_pool: ModelPool
) -> list:
    """Create attack strategy instances from config."""
    strategies = []

    for strategy_name, strategy_kwargs in config.attacker_strategies.items():
        if strategy_name == "distraction":
            strategy = DistractionSentenceStrategy(
                model_pool=model_pool,
                **strategy_kwargs
            )
            strategies.append(strategy)
        elif strategy_name == "impossible_measurement":
            strategy = AdjustImpossibleMeasurementStrategy(
                model_pool=model_pool,
                **strategy_kwargs
            )
            strategies.append(strategy)
        elif strategy_name == "cognitive_bias":
            strategy = CognitiveBiasStrategy(
                model_pool=model_pool,
                **strategy_kwargs
            )
            strategies.append(strategy)
        else:
            print(f"[WARNING] Unknown strategy: {strategy_name}")

    return strategies


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run attack HealthBench evaluation."""
    # Parse arguments
    args = parse_args()

    # Create config
    config = create_config_from_args(args)
    baseline_results_file = getattr(args, "baseline_results", None)
    requested_max_samples = config.max_samples
    run_max_samples = requested_max_samples

    # Validate dataset is specified
    if config.dataset_path is None:
        print("[ERROR] --dataset is required")
        sys.exit(1)

    if len(config.attacker_strategies) != 1:
        print(
            "[ERROR] Exactly one HealthBench attack strategy must be configured "
            "per run. Run each strategy separately."
        )
        return 1

    strategy_name = next(iter(config.attacker_strategies))
    baseline_required = strategy_name in {"distraction", "cognitive_bias"}
    if not baseline_required:
        baseline_results_file = None
    if baseline_required and args.resume and not baseline_results_file:
        print(
            "[ERROR] Resuming baseline-dependent HealthBench attacks requires "
            "--baseline-results so the attack stage stays bound to explicit baseline inputs."
        )
        return 1

    dataset_identity = None
    baseline_identity = None
    all_test_cases = []
    effective_test_cases = []

    # Handle resume
    existing_results = []
    processed_ids = set()
    resume_path = None
    resume_metadata = {}

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {args.resume}")
            sys.exit(1)

        # Validate settings
        initial_baseline_identity = (
            healthbench_source_artifact_identity(baseline_results_file)
            if baseline_results_file else None
        )
        is_valid, msg = validate_resume_file(
            resume_path,
            config.testee_model,
            config.grader_model,
            list(config.attacker_strategies.keys()),
            config.to_dict().get("attacker_strategies", {}),
            config.dataset_path,
            config.filter_single_turn,
            baseline_results_sha256=(
                initial_baseline_identity.get("source_results_sha256")
                if initial_baseline_identity else None
            ),
            testee_config=config.to_dict().get("testee_config"),
            testee_system_prompt=config.testee_system_prompt,
            grader_config=config.to_dict().get("grader_config"),
            max_samples=requested_max_samples,
            require_baseline_sha=baseline_required,
        )
        if not is_valid:
            print(f"[ERROR] Cannot resume - {msg}")
            sys.exit(1)

        if not args.quiet:
            print(f"\n[Resuming from existing results]")

        existing_results, resume_metadata = load_results_from_json(
            resume_path,
            reconstruct_fn=reconstruct_healthbench_result,
            quiet=args.quiet
        )
        if existing_results:
            processed_ids = {r.test_case_id for r in existing_results}
            if not args.quiet:
                print(f"  Found {len(existing_results)} existing results")

    if existing_results and requested_max_samples is not None:
        run_max_samples = max(0, requested_max_samples - len(existing_results))
        if not args.quiet:
            print(
                f"  Will process up to {run_max_samples} new cases "
                f"(max_samples={requested_max_samples}, already done={len(existing_results)})"
            )

    if requested_max_samples == 0 and resume_path is None:
        if not args.quiet:
            print("Requested max_samples is 0; nothing to do.")
        return 0

    # Load dataset before starting new work so resume compatibility can use content identity.
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}")
        sys.exit(1)

    all_test_cases = load_healthbench_jsonl(str(dataset_path))
    effective_test_cases = all_test_cases
    if config.filter_single_turn:
        from med_red_team.healthbench.utils import filter_single_turn_cases
        effective_test_cases = filter_single_turn_cases(effective_test_cases)
    if requested_max_samples is not None:
        effective_test_cases = effective_test_cases[:requested_max_samples]
    dataset_identity = healthbench_effective_dataset_identity(
        effective_test_cases,
        path=config.dataset_path,
        sample_limit=requested_max_samples,
        filter_single_turn=config.filter_single_turn,
    )
    baseline_identity = (
        healthbench_source_artifact_identity(baseline_results_file)
        if baseline_results_file else None
    )

    if resume_path:
        is_valid, msg = validate_resume_file(
            resume_path,
            config.testee_model,
            config.grader_model,
            list(config.attacker_strategies.keys()),
            config.to_dict().get("attacker_strategies", {}),
            config.dataset_path,
            config.filter_single_turn,
            dataset_identity.get("effective_sha256"),
            baseline_identity.get("source_results_sha256") if baseline_identity else None,
            testee_config=config.to_dict().get("testee_config"),
            testee_system_prompt=config.testee_system_prompt,
            grader_config=config.to_dict().get("grader_config"),
            max_samples=requested_max_samples,
            require_baseline_sha=baseline_required,
        )
        if not is_valid:
            print(f"[ERROR] Cannot resume - {msg}")
            sys.exit(1)
        try:
            processed_ids = validate_healthbench_resume_rows(
                existing_results,
                allowed_case_ids=(case.prompt_id for case in effective_test_cases),
            )
        except ValueError as exc:
            print(f"[ERROR] Cannot resume - {exc}")
            sys.exit(1)

    # Load baseline results if provided (optional for impossible_measurement)
    baseline_results = None
    baseline_metadata = None
    if baseline_results_file:
        try:
            baseline_results, baseline_metadata = load_baseline_results(
                baseline_results_file,
                quiet=args.quiet,
                expected_testee_model=config.testee_model,
                expected_testee_config=config.to_dict().get("testee_config"),
                expected_testee_system_prompt=config.testee_system_prompt,
                expected_grader_model=config.grader_model,
                expected_grader_config=config.to_dict().get("grader_config"),
            )
        except Exception as exc:
            print(f"[ERROR] {exc}")
            return 1

        # Validate testee_model matches baseline
        if baseline_metadata and baseline_results:
            baseline_testee_model = _healthbench_model_id(baseline_metadata, "testee")
            if baseline_testee_model and baseline_testee_model != config.testee_model:
                print(f"\n[ERROR] Testee model mismatch!")
                print(f"  Baseline results were generated with: {baseline_testee_model}")
                print(f"  Current testee model is: {config.testee_model}")
                print(f"\n  Attack evaluation requires testing the same model that was used in baseline.")
                print(f"  Please use --testee-model {baseline_testee_model} or omit --testee-model to use baseline model.")
                sys.exit(1)

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 80)
        print("HEALTHBENCH ATTACK EVALUATION")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Dataset: {config.dataset_path}")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Grader model: {config.grader_model}")
        attack_str = ', '.join([f"{name}[{cfg.get('model_id', 'unknown')}]" for name, cfg in config.attacker_strategies.items()])
        print(f"  Attack strategies: {attack_str}")
        print(f"  Max samples: {'All' if requested_max_samples is None else requested_max_samples}")
        print(f"  Filter single-turn: {config.filter_single_turn}")
        print(f"  Using cached baseline: {baseline_results is not None}")

    # Use the already loaded effective dataset for evaluation
    if not args.quiet:
        print(f"\n[Loading Dataset]")
        print(f"  Loaded {len(all_test_cases)} test cases from {dataset_path.name}")
        print(f"  Effective cases after filter/sample limit: {len(effective_test_cases)}")

    test_cases = effective_test_cases

    # Filter out already processed cases if resuming
    if processed_ids:
        unprocessed_cases = [tc for tc in test_cases if tc.prompt_id not in processed_ids]
        if not args.quiet:
            print(f"  {len(unprocessed_cases)} cases remaining to process (resuming)")
        test_cases = unprocessed_cases

    if baseline_results is not None:
        dataset_lookup = {test_case.prompt_id: test_case for test_case in test_cases}
        baseline_ids = {result.test_case_id for result in baseline_results}
        if baseline_required:
            missing_baselines = [
                test_case.prompt_id
                for test_case in test_cases
                if test_case.prompt_id not in baseline_ids
            ]
            if missing_baselines:
                preview = ", ".join(missing_baselines[:5])
                print(
                    "[ERROR] Cached baseline results do not cover all requested HealthBench cases "
                    f"(examples: {preview})"
                )
                return 1
        for baseline_result in baseline_results:
            current_case = dataset_lookup.get(baseline_result.test_case_id)
            if current_case is not None:
                validate_cached_baseline_result(current_case, baseline_result)

    if not test_cases:
        if existing_results and resume_metadata.get("is_partial", False):
            final_path = resume_path
            if str(final_path).endswith(".inprogress"):
                final_path = Path(str(final_path)[:-len(".inprogress")])
            pipeline = HealthBenchRobustnessPipeline(
                testee=None,
                grader=None,
                attack_strategies=[],
                verbose=False,
            )
            summary = pipeline._compute_summary(existing_results, baseline_only=False)
            pipeline.save_results(
                results=existing_results,
                summary=summary,
                output_path=str(final_path),
                metadata=_make_metadata(
                    config,
                    is_partial=False,
                    baseline_results_file=baseline_results_file,
                    total_cases_evaluated=len(existing_results),
                    dataset_identity=dataset_identity,
                    baseline_identity=baseline_identity,
                ),
            )
            remove_checkpoint(final_path)
            if not args.quiet:
                print(f"  Finalized completed checkpoint: {final_path}")
        elif not args.quiet:
            print("No HealthBench attack cases remain to evaluate.")
        return 0

    # Initialize model pool
    model_pool = ModelPool()

    # Create testee
    testee = Testee(
        model_id=config.testee_model,
        model_pool=model_pool,
        system_prompt=config.testee_system_prompt,
        config=config.testee_config
    )

    # Create grader
    grader = RubricGrader(
        model_id=config.grader_model,
        model_pool=model_pool,
        config=config.grader_config,
        verbose=args.verbose
    )

    # Create attack strategies from config
    if not args.quiet:
        print(f"\n[Creating Attack Strategies]")

    attack_strategies = create_attack_strategies(config, model_pool)

    if len(attack_strategies) == 0:
        print("[ERROR] No attack strategies defined in config")
        sys.exit(1)

    for strategy in attack_strategies:
        if not args.quiet:
            print(f"  - {strategy.name}")

    # Create pipeline
    pipeline = HealthBenchRobustnessPipeline(
        testee=testee,
        grader=grader,
        attack_strategies=attack_strategies,
        verbose=not args.quiet
    )

    # Generate and validate output path before evaluation
    # When resuming, create new file with combined count
    expected_new_results = run_max_samples if run_max_samples is not None else len(test_cases)
    total_expected = len(existing_results) + expected_new_results
    output_path = generate_and_validate_output_path(
        log_dir=config.log_dir,
        testee_model=config.testee_model,
        mode=None,  # Skip mode for healthbench attack format
        strategies=list(config.attacker_strategies.keys()),
        max_samples=total_expected if existing_results else requested_max_samples,
        quiet=args.quiet
    )

    if output_path is None:
        print("[ERROR] Cannot write to output directory. Exiting.")
        return 1

    if resume_path and existing_results:
        if not args.quiet:
            print(f"\n[Resume Mode]")
            print(f"  Resuming from: {resume_path.name}")
            print(f"  Saving combined results to NEW file: {output_path.name}")

    def progress_callback(current_results):
        combined = merge_healthbench_results(existing_results, current_results)
        pipeline.save_results(
            results=combined,
            summary=pipeline._compute_summary(combined, baseline_only=False),
            output_path=str(checkpoint_path(output_path)),
            metadata=_make_metadata(
                config,
                is_partial=True,
                baseline_results_file=baseline_results_file,
                total_cases_evaluated=len(combined),
                dataset_identity=dataset_identity,
                baseline_identity=baseline_identity,
            ),
        )

    # Run attack evaluation (with periodic checkpointing)
    if not args.quiet:
        print(f"\n[Running Attack Evaluation]")

    try:
        new_results, summary = pipeline.run_attack(
            test_cases=test_cases,
            max_samples=None,
            filter_single_turn=False,
            baseline_results=baseline_results,
            progress_callback=progress_callback,
            save_every=10,
            baseline_metadata=baseline_metadata,
        )

        # Combine with existing results if resuming
        if existing_results:
            for sample_number, result in enumerate(
                new_results, start=len(existing_results) + 1
            ):
                result.sample_number = sample_number
        results = existing_results + new_results if existing_results else new_results

        # Recompute summary with all results
        if existing_results:
            summary = pipeline._compute_summary(results, baseline_only=False)
    except Exception as e:
        print(f"\n[ERROR] Attack evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        # Partial results saved to .inprogress checkpoint for recovery
        partial_file = output_path.parent / (output_path.name + ".inprogress")
        if partial_file.exists():
            print(f"Partial results were saved to: {partial_file}")
        sys.exit(1)

    # Save final results
    if not args.quiet:
        print(f"\n[Saving Results]")

    pipeline.save_results(
        results=results,
        summary=summary,
        output_path=str(output_path),
        metadata=_make_metadata(
            config,
            is_partial=False,
            baseline_results_file=baseline_results_file,
            total_cases_evaluated=len(results),
            dataset_identity=dataset_identity,
            baseline_identity=baseline_identity,
        ),
    )
    remove_checkpoint(output_path)

    # Print summary
    if not args.quiet:
        print(f"\n{'='*80}")
        print("ATTACK EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {output_path}")
        if baseline_results_file:
            print(f"Baseline results: {baseline_results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
