#!/usr/bin/env python3
"""
HealthBench Baseline Evaluation
================================

Evaluate model performance on HealthBench dataset without attacks.

This script runs baseline evaluation to measure how well a model performs
on HealthBench rubric-based evaluation. Results can be cached and reused
for attack testing with run_attack.py.

Usage:
    # Basic usage with defaults
    python -m scripts.healthbench.run_baseline \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --testee-model gpt-4o

    # Specify all parameters
    python -m scripts.healthbench.run_baseline \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --testee-model gpt-4o \
        --grader-model gemini-2.5-flash \
        --max-samples 100 \
        --filter-single-turn \
        --output-dir logs/healthbench

Example:
    >>> python -m scripts.healthbench.run_baseline \
    ...     --dataset test/healthbench_dataset/consensus.jsonl \
    ...     --testee-model gpt-4o \
    ...     --max-samples 50

    Results saved to: logs/healthbench/gpt-4o/baseline_20250119_143022.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, List, Optional, Set, Tuple

from med_red_team.testee import Testee
from med_red_team.model_pool import ModelPool
from med_red_team.healthbench.grader import RubricGrader
from med_red_team.healthbench.pipeline import HealthBenchRobustnessPipeline
from med_red_team.healthbench.utils import load_healthbench_jsonl
from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.healthbench.data import (
    HealthBenchRobustnessResult,
    build_healthbench_metadata,
    healthbench_dataset_effective_sha,
    healthbench_effective_dataset_identity,
    healthbench_dataset_path,
    healthbench_filter_single_turn,
    _healthbench_generation_config,
    _healthbench_model_id,
    _healthbench_testee_system_prompt,
)
from med_red_team.models import GenerationConfig

# Import baseline config
from configs.examples.healthbench.baseline import CONFIG as BASELINE_CONFIG
from med_red_team.shared.checkpoints import checkpoint_path, remove_checkpoint
from med_red_team.shared.config_loading import clone_config, resolve_sample_limit
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


def _generation_config_dict(value: Any) -> dict | None:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return None


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HealthBench baseline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline config
  python -m scripts.healthbench.run_baseline \\
      --dataset test/healthbench_dataset/consensus.jsonl

  # Override config with command-line args
  python -m scripts.healthbench.run_baseline \\
      --dataset test/healthbench_dataset/consensus.jsonl \\
      --testee-model gpt-4o \\
      --grader-model gemini-2.5-flash \\
      --max-samples 100

  # Load custom config file
  python -m scripts.healthbench.run_baseline \\
      --config configs/my_config.py \\
      --dataset test/healthbench_dataset/consensus.jsonl
        """
    )

    # Configuration file (optional - defaults to baseline config)
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (default: configs/examples/healthbench/baseline.py)"
    )

    # Data settings
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to HealthBench dataset (JSONL file) (overrides config)"
    )

    # Model settings
    parser.add_argument(
        "--testee-model",
        type=str,
        default=None,
        help="Model ID for the testee (e.g., gpt-4o) (overrides config)"
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help="Model ID for rubric grading (overrides config)"
    )

    # Evaluation settings
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of test cases to evaluate (overrides config)"
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
    # Load from config file if specified, otherwise use BASELINE_CONFIG
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, HealthBenchConfig)
    else:
        # Start with an independent baseline config.
        if not args.quiet:
            print(f"[Config] Using baseline config")
        config = clone_config(BASELINE_CONFIG)

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


def _baseline_dataset_metadata(config: HealthBenchConfig, dataset_identity: dict | None = None) -> dict:
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



def _baseline_source_metadata(config: HealthBenchConfig, dataset_identity: dict | None = None) -> dict:
    metadata = {
        "dataset_path": config.dataset_path,
        "workflow": "run_baseline",
    }
    if dataset_identity:
        metadata.update({
            "dataset_path": config.dataset_path,
            "dataset_file_sha256": dataset_identity.get("file_sha256"),
            "dataset_effective_sha256": dataset_identity.get("effective_sha256"),
        })
    return metadata



def _make_metadata(
    config: HealthBenchConfig,
    *,
    is_partial: bool,
    total_cases_evaluated: int | None = None,
    dataset_identity: dict | None = None,
) -> dict:
    extra = {
        "total_cases_evaluated": total_cases_evaluated,
        "bootstrap_seed": 0,
        "bootstrap_clips_mean_to_unit_interval": True,
    }
    return build_healthbench_metadata(
        phase="baseline",
        config=config,
        is_partial=is_partial,
        dataset=_baseline_dataset_metadata(config, dataset_identity),
        source=_baseline_source_metadata(config, dataset_identity),
        extra=extra,
    )



def validate_resume_file(
    resume_path: Path,
    testee_model: str,
    grader_model: str,
    dataset_path: str,
    filter_single_turn: bool,
    dataset_effective_sha256: str | None = None,
    *,
    max_samples: Any = _UNSET,
    testee_config: dict | None = None,
    testee_system_prompt: str | None = None,
    grader_config: dict | None = None,
) -> Tuple[bool, str]:
    """
    Validate resume file settings match current settings.

    Args:
        resume_path: Path to resume file
        testee_model: Current testee model ID
        grader_model: Current grader model ID
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
        if metadata.get('phase') not in {None, 'baseline'}:
            errors.append(f"Phase mismatch: file has '{metadata.get('phase')}', expected 'baseline'")

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
        if (
            testee_system_prompt is not None
            and file_testee_system_prompt != testee_system_prompt
        ):
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


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run baseline HealthBench evaluation."""
    # Parse arguments
    args = parse_args()

    # Create config
    config = create_config_from_args(args)
    requested_max_samples = config.max_samples
    run_max_samples = requested_max_samples

    # Validate dataset is specified
    if config.dataset_path is None:
        print("[ERROR] --dataset is required")
        sys.exit(1)

    dataset_identity = None
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

        # Validate settings/content identity
        is_valid, msg = validate_resume_file(
            resume_path,
            config.testee_model,
            config.grader_model,
            config.dataset_path,
            config.filter_single_turn,
            max_samples=requested_max_samples,
            testee_config=_generation_config_dict(
                getattr(config, "testee_config", None)
            ),
            testee_system_prompt=getattr(config, "testee_system_prompt", None),
            grader_config=_generation_config_dict(
                getattr(config, "grader_config", None)
            ),
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

    if resume_path:
        is_valid, msg = validate_resume_file(
            resume_path,
            config.testee_model,
            config.grader_model,
            config.dataset_path,
            config.filter_single_turn,
            dataset_identity.get("effective_sha256"),
            max_samples=requested_max_samples,
            testee_config=_generation_config_dict(
                getattr(config, "testee_config", None)
            ),
            testee_system_prompt=getattr(config, "testee_system_prompt", None),
            grader_config=_generation_config_dict(
                getattr(config, "grader_config", None)
            ),
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

    # Print configuration
    if not args.quiet:
        print("=" * 80)
        print("HEALTHBENCH BASELINE EVALUATION")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Dataset: {config.dataset_path}")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Grader model: {config.grader_model}")
        print(f"  Max samples: {'All' if requested_max_samples is None else requested_max_samples}")
        print(f"  Filter single-turn: {config.filter_single_turn}")
        if resume_path:
            print(f"  Resuming from: {resume_path.name}")

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
            summary = pipeline._compute_summary(existing_results, baseline_only=True)
            pipeline.save_results(
                results=existing_results,
                summary=summary,
                output_path=str(final_path),
                metadata=_make_metadata(
                    config,
                    is_partial=False,
                    total_cases_evaluated=len(existing_results),
                    dataset_identity=dataset_identity,
                ),
            )
            remove_checkpoint(final_path)
            if not args.quiet:
                print(f"  Finalized completed checkpoint: {final_path}")
        elif not args.quiet:
            print("No HealthBench baseline cases remain to evaluate.")
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

    # Create pipeline (no attack strategies for baseline)
    pipeline = HealthBenchRobustnessPipeline(
        testee=testee,
        grader=grader,
        attack_strategies=[],  # Empty list for baseline-only
        verbose=not args.quiet
    )

    # Generate and validate output path before evaluation
    # When resuming, include both existing and expected new results in count
    expected_new_results = run_max_samples if run_max_samples is not None else len(test_cases)
    total_expected = len(existing_results) + expected_new_results
    output_path = generate_and_validate_output_path(
        log_dir=config.log_dir,
        testee_model=config.testee_model,
        mode="baseline",
        strategies=None,
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
            summary=pipeline._compute_summary(combined, baseline_only=True),
            output_path=str(checkpoint_path(output_path)),
            metadata=_make_metadata(
                config,
                is_partial=True,
                total_cases_evaluated=len(combined),
                dataset_identity=dataset_identity,
            ),
        )

    # Run baseline evaluation with error handling
    if not args.quiet:
        print(f"\n[Running Baseline Evaluation]")

    try:
        new_results, summary = pipeline.run_baseline(
            test_cases=test_cases,
            max_samples=None,
            filter_single_turn=False,
            progress_callback=progress_callback,
            save_every=10
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
            summary = pipeline._compute_summary(results, baseline_only=True)

    except KeyboardInterrupt:
        print(f"\n\n[WARNING] Evaluation interrupted by user")
        # Results are already saved by progress_callback
        sys.exit(1)

    except Exception as e:
        print(f"\n[ERROR] Unexpected error during evaluation: {e}")
        import traceback
        traceback.print_exc()
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
            total_cases_evaluated=len(results),
            dataset_identity=dataset_identity,
        ),
    )
    remove_checkpoint(output_path)

    # Print summary
    if not args.quiet:
        print(f"\n{'='*80}")
        print("BASELINE EVALUATION COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {output_path}")
        print(f"\nTo run attack evaluation using this baseline:")
        print(f"  python -m scripts.healthbench.run_attack \\")
        print(f"      --baseline-results {output_path} \\")
        print(f"      --dataset {config.dataset_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
