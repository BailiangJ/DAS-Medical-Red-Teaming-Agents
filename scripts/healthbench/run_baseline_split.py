#!/usr/bin/env python3
"""
HealthBench Baseline Evaluation - Split Mode
=============================================

Split the baseline evaluation process into separate collection and grading stages:
1. Collect: Get responses from testee model (no grading)
2. Grade: Grade saved testee responses (with resume support)

This allows for:
- Decoupling response collection from grading
- Using different graders on same testee responses
- Resuming grading if interrupted

Usage:
    # Stage 1: Collect responses only
    python -m scripts.healthbench.run_baseline_split \
        --mode collect \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --testee-model gpt-4o \
        --output responses.json

    # Stage 1 with resume: Resume collection if interrupted
    python -m scripts.healthbench.run_baseline_split \
        --mode collect \
        --dataset test/healthbench_dataset/consensus.jsonl \
        --testee-model gpt-4o \
        --resume responses.json \
        --output responses_continued.json

    # Stage 2: Grade collected responses
    python -m scripts.healthbench.run_baseline_split \
        --mode grade \
        --testee-responses responses.json \
        --grader-model gemini-2.5-flash \
        --output graded_results.json

    # Stage 2 with resume: Resume grading with previous progress
    python -m scripts.healthbench.run_baseline_split \
        --mode grade \
        --testee-responses responses.json \
        --grader-model gemini-2.5-flash \
        --resume graded_results.json \
        --output graded_results_continued.json

Example:
    >>> # First collect responses
    >>> python -m scripts.healthbench.run_baseline_split \
    ...     --mode collect \
    ...     --dataset test/healthbench_dataset/consensus.jsonl \
    ...     --testee-model gpt-4o

    >>> # Then grade them
    >>> python -m scripts.healthbench.run_baseline_split \
    ...     --mode grade \
    ...     --testee-responses logs/healthbench/gpt-4o/collect_20250119_143022.json \
    ...     --grader-model gemini-2.5-flash
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import List, Optional, Set, Tuple

from med_red_team.testee import Testee
from med_red_team.model_pool import ModelPool
from med_red_team.healthbench.grader import RubricGrader
from med_red_team.healthbench.pipeline import HealthBenchRobustnessPipeline
from med_red_team.healthbench.utils import load_healthbench_jsonl, calculate_score, filter_single_turn_cases
from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.healthbench.data import (
    HealthBenchRobustnessResult,
    RubricItem,
    RubricGradeResult,
    build_healthbench_metadata,
    healthbench_effective_dataset_identity,
    healthbench_source_artifact_identity,
    healthbench_dataset_path,
    _healthbench_generation_config,
    _healthbench_model_id,
    _healthbench_testee_system_prompt,
    validate_healthbench_artifact_metadata,
)
from med_red_team.shared.checkpoints import (
    checkpoint_path,
    remove_checkpoint,
)
from med_red_team.shared.config_loading import clone_config, resolve_sample_limit
from med_red_team.shared.io import atomic_write_result_envelope
from med_red_team.utils import evaluate_items_fail_fast

# Import baseline config
from configs.examples.healthbench.baseline import CONFIG as BASELINE_CONFIG
from scripts.healthbench.checkpoints import (
    merge_healthbench_results,
    validate_healthbench_checkpoint_metadata,
    validate_healthbench_resume_rows,
)
from scripts.utils import (
    load_config_from_py,
    generate_output_path,
    load_results_from_json,
    validate_output_path,
    reconstruct_healthbench_result
)


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run HealthBench baseline evaluation in split mode (collect or grade)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect testee responses
  python -m scripts.healthbench.run_baseline_split \\
      --mode collect \\
      --dataset test/healthbench_dataset/consensus.jsonl \\
      --testee-model gpt-4o

  # Resume collection from partial results
  python -m scripts.healthbench.run_baseline_split \\
      --mode collect \\
      --dataset test/healthbench_dataset/consensus.jsonl \\
      --testee-model gpt-4o \\
      --resume logs/healthbench/gpt-4o/collect_*.json

  # Grade collected responses
  python -m scripts.healthbench.run_baseline_split \\
      --mode grade \\
      --testee-responses logs/healthbench/gpt-4o/collect_*.json \\
      --grader-model gemini-2.5-flash

  # Resume grading from partial results
  python -m scripts.healthbench.run_baseline_split \\
      --mode grade \\
      --testee-responses logs/healthbench/gpt-4o/collect_*.json \\
      --grader-model gemini-2.5-flash \\
      --resume logs/healthbench/gpt-4o/grade_*.json
        """
    )

    # Mode selection (required)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["collect", "grade"],
        help="Mode: 'collect' to get testee responses, 'grade' to grade saved responses"
    )

    # Configuration file (optional)
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
        help="Model ID for the testee (required for collect mode) (overrides config)"
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help="Model ID for rubric grading (required for grade mode) (overrides config)"
    )

    # Mode-specific inputs
    parser.add_argument(
        "--testee-responses",
        type=str,
        default=None,
        help="Path to saved testee responses JSON (required for grade mode)"
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
        help="Path to existing results file to resume from (works for both collect and grade modes)"
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


def _collect_metadata(
    config: HealthBenchConfig,
    *,
    is_partial: bool,
    total_collected: int,
    dataset_identity: dict | None = None,
) -> dict:
    dataset_metadata = {
        "path": config.dataset_path,
        "sample_limit": config.max_samples,
        "filter_single_turn": config.filter_single_turn,
    }
    source_metadata = {
        "dataset_path": config.dataset_path,
        "workflow": "run_baseline_split_collect",
    }
    if dataset_identity:
        dataset_metadata.update({
            "canonical_path": dataset_identity.get("canonical_path"),
            "file_sha256": dataset_identity.get("file_sha256"),
            "effective_rows": dataset_identity.get("effective_rows"),
            "effective_sha256": dataset_identity.get("effective_sha256"),
        })
        source_metadata.update({
            "dataset_file_sha256": dataset_identity.get("file_sha256"),
            "dataset_effective_sha256": dataset_identity.get("effective_sha256"),
        })
    return build_healthbench_metadata(
        phase="baseline_collect",
        config=config,
        is_partial=is_partial,
        dataset=dataset_metadata,
        source=source_metadata,
        extra={
            "total_collected": total_collected,
            "bootstrap_seed": 0,
            "bootstrap_clips_mean_to_unit_interval": True,
        },
    )



def _grade_metadata(
    config: HealthBenchConfig,
    testee_metadata: dict,
    responses_path: Path,
    *,
    is_partial: bool,
    total_cases_evaluated: int,
) -> dict:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    source_config = testee_metadata.get("config") if isinstance(testee_metadata.get("config"), dict) else {}
    dataset_info = testee_metadata.get("dataset") if isinstance(testee_metadata.get("dataset"), dict) else {}
    models = testee_metadata.get("models") if isinstance(testee_metadata.get("models"), dict) else {}
    testee_data = models.get("testee") if isinstance(models.get("testee"), dict) else {}
    config_dict["dataset_path"] = (
        dataset_info.get("path")
        or source_config.get("dataset_path")
        or testee_metadata.get("dataset_path")
        or config_dict.get("dataset_path")
    )
    config_dict["filter_single_turn"] = (
        dataset_info.get("filter_single_turn")
        if dataset_info.get("filter_single_turn") is not None
        else source_config.get("filter_single_turn")
        if source_config.get("filter_single_turn") is not None
        else testee_metadata.get("filter_single_turn", config_dict.get("filter_single_turn"))
    )
    config_dict["testee_model"] = (
        testee_data.get("model_id")
        or source_config.get("testee_model")
        or testee_metadata.get("testee_model")
        or config_dict.get("testee_model")
    )
    config_dict["testee_config"] = (
        testee_data.get("generation_config")
        if testee_data.get("generation_config") is not None
        else source_config.get("testee_config", testee_metadata.get("testee_config", config_dict.get("testee_config")))
    )
    config_dict["testee_system_prompt"] = (
        testee_data.get("system_prompt")
        if testee_data.get("system_prompt") is not None
        else source_config.get("testee_system_prompt", testee_metadata.get("testee_system_prompt", config_dict.get("testee_system_prompt")))
    )
    responses_identity = healthbench_source_artifact_identity(
        responses_path,
        key_prefix="testee_responses",
    )
    return build_healthbench_metadata(
        phase="baseline",
        config=config_dict,
        is_partial=is_partial,
        dataset={
            "path": config_dict.get("dataset_path"),
            "sample_limit": dataset_info.get("sample_limit"),
            "filter_single_turn": config_dict.get("filter_single_turn"),
            "effective_sha256": dataset_info.get("effective_sha256"),
            "file_sha256": dataset_info.get("file_sha256"),
        },
        source={
            "source_results_file": str(responses_path),
            **responses_identity,
            "source_results_sha256": responses_identity.get("testee_responses_sha256"),
            "source_schema_version": testee_metadata.get("schema_version"),
            "source_axis": testee_metadata.get("axis"),
            "source_phase": testee_metadata.get("phase"),
            "source_models": testee_metadata.get("models", {}),
            "source_dataset": testee_metadata.get("dataset", {}),
            "workflow": "run_baseline_split_grade",
        },
        extra={
            "testee_responses_file": str(responses_path),
            "total_cases_evaluated": total_cases_evaluated,
            "bootstrap_seed": 0,
            "bootstrap_clips_mean_to_unit_interval": True,
        },
    )



def _responses_summary(total_items: int) -> dict:
    return {
        "total_cases": total_items,
        "total_items": total_items,
    }



# ==============================================================================
# Collect Mode
# ==============================================================================

def run_collect_mode(args: argparse.Namespace, config: HealthBenchConfig) -> int:
    """
    Run collect mode: Get testee responses without grading.

    Args:
        args: Command-line arguments
        config: HealthBench configuration

    Returns:
        Exit code
    """
    # Validate required settings
    if config.testee_model is None:
        print("[ERROR] --testee-model is required for collect mode")
        return 1

    if config.dataset_path is None:
        print("[ERROR] --dataset is required")
        return 1

    # Load dataset before resume validation so compatibility can use content identity.
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}")
        return 1

    all_test_cases = load_healthbench_jsonl(str(dataset_path))
    effective_test_cases = all_test_cases
    if config.filter_single_turn:
        effective_test_cases = filter_single_turn_cases(effective_test_cases)
    if config.max_samples is not None:
        effective_test_cases = effective_test_cases[:config.max_samples]
    dataset_identity = healthbench_effective_dataset_identity(
        effective_test_cases,
        path=config.dataset_path,
        sample_limit=config.max_samples,
        filter_single_turn=config.filter_single_turn,
    )

    # Handle resume
    existing_responses = []
    processed_ids = set()
    resume_path = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {args.resume}")
            return 1

        if not args.quiet:
            print(f"\n[Loading Resume File]")

        existing_responses, resume_metadata = load_results_from_json(
            resume_path,
            data_key="responses",
            quiet=args.quiet
        )

        try:
            validate_healthbench_checkpoint_metadata(
                resume_metadata,
                phase="baseline_collect",
                expected_testee_model=config.testee_model,
                expected_testee_config=config.testee_config.to_dict(),
                expected_testee_system_prompt=config.testee_system_prompt,
                expected_dataset_effective_sha256=dataset_identity.get("effective_sha256"),
                expected_config_sample_limit=config.max_samples,
                expected_dataset_sample_limit=config.max_samples,
                expected_filter_single_turn=config.filter_single_turn,
            )
            processed_ids = validate_healthbench_resume_rows(
                existing_responses,
                allowed_case_ids=(case.prompt_id for case in effective_test_cases),
            )
        except ValueError as exc:
            print(f"[ERROR] Cannot resume - {exc}")
            return 1
        if not args.quiet:
            print(f"  Found {len(existing_responses)} existing responses")

    # Print configuration
    if not args.quiet:
        print("=" * 80)
        print("HEALTHBENCH BASELINE - COLLECT MODE")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Dataset: {config.dataset_path}")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Max samples: {'All' if config.max_samples is None else config.max_samples}")
        print(f"  Filter single-turn: {config.filter_single_turn}")
        if resume_path:
            print(f"  Resuming from: {resume_path.name}")

    # Use the already loaded effective dataset for collection
    if not args.quiet:
        print(f"\n[Loading Dataset]")

    test_cases = effective_test_cases

    # Filter out already processed cases if resuming
    if processed_ids:
        unprocessed_cases = [tc for tc in test_cases if tc.prompt_id not in processed_ids]
        if not args.quiet:
            print(f"  {len(unprocessed_cases)} cases remaining to collect (resuming)")
        test_cases = unprocessed_cases

    if not args.quiet:
        total_after_resume = len(existing_responses) + len(test_cases)
        print(f"  Loaded {len(test_cases)} test cases to collect from {dataset_path.name}")
        if existing_responses:
            print(f"  Total after resume: {total_after_resume} responses")

    # Generate output path
    # When resuming, include both existing and expected new responses in count
    total_expected = len(existing_responses) + len(test_cases)
    output_path = generate_output_path(
        log_dir=config.log_dir,
        testee_model=config.testee_model,
        mode="collect",
        strategies=None,
        max_samples=total_expected if existing_responses else config.max_samples
    )

    # Validate output path is writable before starting collection
    if not validate_output_path(output_path, quiet=args.quiet):
        return 1

    if resume_path and existing_responses:
        if not args.quiet:
            print(f"\n[Resume Mode]")
            print(f"  Resuming from: {resume_path.name}")
            print(f"  Saving combined responses to: {output_path.name}")

    # Define collection function
    def collect_response(test_case, idx: int) -> dict:
        """Collect testee response for a single test case."""
        conv_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in test_case.conversation
        ])

        response = testee.answer(conv_text)
        completion = response.final_answer

        return {
            "test_case_id": test_case.prompt_id,
            "sample_number": idx + 1,
            "conversation": test_case.conversation,
            "rubrics": [r.to_dict() for r in test_case.rubrics],
            "completion": completion,
            "parse_error": response.metadata.get("parse_error"),
        }

    def save_responses(responses: List[dict], *, is_partial: bool):
        """Save responses to a stable result envelope."""
        all_responses = merge_healthbench_results(existing_responses, responses)
        target_path = checkpoint_path(output_path) if is_partial else output_path
        atomic_write_result_envelope(
            target_path,
            metadata=_collect_metadata(
                config,
                is_partial=is_partial,
                total_collected=len(all_responses),
                dataset_identity=dataset_identity,
            ),
            summary=_responses_summary(len(all_responses)),
            data=all_responses,
            data_key="responses",
        )

        if not args.quiet and is_partial:
            print(f"  Checkpoint: Saved {len(all_responses)} total responses to {target_path.name}")

    def progress_callback(responses: List[dict]):
        save_responses(responses, is_partial=True)

    if not test_cases:
        responses = merge_healthbench_results(existing_responses, [])
        save_responses(responses, is_partial=False)
        remove_checkpoint(output_path)
        if not args.quiet:
            print("No HealthBench baseline responses remain to collect.")
        return 0

    # Initialize the provider only after confirming work remains.
    model_pool = ModelPool()
    testee = Testee(
        model_id=config.testee_model,
        model_pool=model_pool,
        system_prompt=config.testee_system_prompt,
        config=config.testee_config
    )

    # Run collection with error handling
    if not args.quiet:
        print(f"\n[Collecting Testee Responses]")

    try:
        new_responses = evaluate_items_fail_fast(
            items=test_cases,
            evaluate_fn=collect_response,
            description="Collecting",
            progress_callback=progress_callback,
            save_every=10,
            verbose=not args.quiet,
            use_tqdm=True
        )

        # Combine with existing responses if resuming
        responses = merge_healthbench_results(existing_responses, new_responses)

    except KeyboardInterrupt:
        print(f"\n\n[WARNING] Collection interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected error during collection: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save final results
    if not args.quiet:
        print(f"\n[Saving Responses]")

    save_responses(responses, is_partial=False)
    remove_checkpoint(output_path)

    # Print summary
    if not args.quiet:
        print(f"\n{'='*80}")
        print("COLLECTION COMPLETE")
        print(f"{'='*80}")
        print(f"\nResponses saved to: {output_path}")
        print(f"Total collected: {len(responses)}")
        if existing_responses:
            print(f"  Previously collected: {len(existing_responses)}")
            print(f"  Newly collected: {len(new_responses)}")
        print(f"\nTo grade these responses:")
        print(f"  python -m scripts.healthbench.run_baseline_split \\")
        print(f"      --mode grade \\")
        print(f"      --dataset {config.dataset_path} \\")
        print(f"      --testee-responses {output_path} \\")
        print(f"      --grader-model <your-grader-model>")

    return 0


# ==============================================================================
# Grade Mode
# ==============================================================================

def run_grade_mode(args: argparse.Namespace, config: HealthBenchConfig) -> int:
    """
    Run grade mode: Grade saved testee responses.

    Args:
        args: Command-line arguments
        config: HealthBench configuration

    Returns:
        Exit code
    """
    # Validate required settings
    if config.grader_model is None:
        print("[ERROR] --grader-model is required for grade mode")
        return 1

    if args.testee_responses is None:
        print("[ERROR] --testee-responses is required for grade mode")
        return 1

    # Load testee responses
    if not args.quiet:
        print(f"\n[Loading Testee Responses]")

    responses_path = Path(args.testee_responses)
    try:
        testee_responses, testee_metadata = load_results_from_json(
            responses_path,
            data_key="responses",
            quiet=args.quiet
        )
        validate_healthbench_artifact_metadata(
            testee_metadata,
            phase="baseline_collect",
            label=f"HealthBench testee responses {responses_path}",
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    response_testee_model = _healthbench_model_id(testee_metadata, "testee") or "unknown"
    responses_identity = healthbench_source_artifact_identity(
        responses_path,
        key_prefix="testee_responses",
    )

    existing_results = []
    processed_ids = set()
    resume_metadata = {}
    resume_path = Path(args.resume) if args.resume else None

    if resume_path:
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {args.resume}")
            return 1

        if not args.quiet:
            print(f"\n[Loading Resume File]")

        existing_results, resume_metadata = load_results_from_json(
            resume_path,
            data_key="results",
            quiet=args.quiet,
            allow_missing=False,
            reconstruct_fn=reconstruct_healthbench_result
        )

        response_dataset = (
            testee_metadata.get("dataset")
            if isinstance(testee_metadata.get("dataset"), dict)
            else {}
        )
        try:
            validate_healthbench_checkpoint_metadata(
                resume_metadata,
                phase="baseline",
                expected_testee_model=response_testee_model,
                expected_testee_config=_healthbench_generation_config(
                    testee_metadata,
                    "testee",
                ),
                expected_testee_system_prompt=_healthbench_testee_system_prompt(
                    testee_metadata
                ),
                expected_grader_model=config.grader_model,
                expected_grader_config=config.grader_config.to_dict(),
                expected_source_results_sha256=responses_identity.get(
                    "testee_responses_sha256"
                ),
                expected_config_sample_limit=config.max_samples,
                expected_dataset_sample_limit=response_dataset.get("sample_limit"),
            )
            processed_ids = validate_healthbench_resume_rows(
                existing_results,
                allowed_case_ids=(row["test_case_id"] for row in testee_responses),
            )
        except ValueError as exc:
            print(f"[ERROR] Cannot resume - {exc}")
            return 1
        if not args.quiet:
            print(f"  Found {len(existing_results)} existing results")

    # Filter out already processed responses
    if processed_ids:
        unprocessed_responses = [r for r in testee_responses if r["test_case_id"] not in processed_ids]
        if not args.quiet:
            print(f"  {len(unprocessed_responses)} responses remaining to grade (resuming)")
        testee_responses = unprocessed_responses

    # Apply max_samples limit
    if config.max_samples is not None:
        already_graded = len(existing_results)
        remaining_to_grade = config.max_samples - already_graded

        if remaining_to_grade <= 0:
            if not args.quiet:
                print(f"\n[WARNING] Already graded {already_graded} responses, max_samples={config.max_samples} reached")
                print(f"  Nothing left to grade")
            testee_responses = []

        elif len(testee_responses) > remaining_to_grade:
            if not args.quiet:
                print(f"  Limiting to first {remaining_to_grade} responses (max_samples={config.max_samples}, already graded={already_graded})")
            testee_responses = testee_responses[:remaining_to_grade]

    # Print configuration
    if not args.quiet:
        print("=" * 80)
        print("HEALTHBENCH BASELINE - GRADE MODE")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Testee model: {response_testee_model}")
        print(f"  Grader model: {config.grader_model}")
        print(f"  Max samples: {'All' if config.max_samples is None else config.max_samples}")
        print(f"  Responses to grade: {len(testee_responses)}")
        if existing_results:
            print(f"  Previously graded: {len(existing_results)}")

    pipeline = HealthBenchRobustnessPipeline(
        testee=None,
        grader=None,
        attack_strategies=[],
        verbose=False,
        bootstrap_seed=0,
    )

    # Generate output path
    # Use testee model from metadata
    testee_model_id = response_testee_model
    total_expected = len(existing_results) + len(testee_responses)
    output_path = generate_output_path(
        log_dir=config.log_dir,
        testee_model=testee_model_id,
        mode="baseline",
        strategies=None,
        max_samples=total_expected if existing_results else len(testee_responses)
    )

    # Validate output path is writable before starting grading
    if not validate_output_path(output_path, quiet=args.quiet):
        return 1

    if args.resume and existing_results:
        if not args.quiet:
            print(f"\n[Resume Mode]")
            print(f"  Resuming from: {Path(args.resume).name}")
            print(f"  Saving combined results to: {output_path.name}")

    def grade_response(response: dict, idx: int) -> HealthBenchRobustnessResult:
        """Grade a single testee response."""
        test_case_id = response["test_case_id"]
        conversation = response["conversation"]
        completion = response["completion"]
        rubrics = [RubricItem.from_dict(r) for r in response["rubrics"]]

        parse_error = response.get("parse_error")
        if parse_error:
            reason = parse_error.get("error_message", "Unknown parse error") if isinstance(parse_error, dict) else str(parse_error)
            return HealthBenchRobustnessResult(
                test_case_id=test_case_id,
                sample_number=response["sample_number"],
                original_conversation=conversation,
                original_rubrics=rubrics,
                baseline_completion=completion,
                baseline_grades=[],
                baseline_score=None,
                attack_result=None,
                attacked_conversation=None,
                attacked_rubrics=None,
                attacked_completion=None,
                attacked_grades=None,
                attacked_score=None,
                skipped=True,
                skip_reason=reason,
            )

        grades = grader.grade(
            conversation=conversation,
            completion=completion,
            rubric_items=rubrics
        )
        score = calculate_score(rubrics, grades)

        result = HealthBenchRobustnessResult(
            test_case_id=test_case_id,
            sample_number=response["sample_number"],
            original_conversation=conversation,
            original_rubrics=rubrics,
            baseline_completion=completion,
            baseline_grades=grades,
            baseline_score=score,
            attack_result=None,
            attacked_conversation=None,
            attacked_rubrics=None,
            attacked_completion=None,
            attacked_grades=None,
            attacked_score=None,
        )
        validation_error = pipeline._grading_validation_error(
            rubrics,
            grades,
            label="HealthBench baseline grades",
        )
        if validation_error is not None:
            result.skipped = True
            result.skip_reason = validation_error
        return result

    def save_graded_results(results: List[HealthBenchRobustnessResult], *, is_partial: bool):
        """Save graded results through the shared HealthBench envelope path."""
        all_results = merge_healthbench_results(existing_results, results)
        summary = pipeline._compute_summary(all_results, baseline_only=True)
        pipeline.save_results(
            results=all_results,
            summary=summary,
            output_path=str(checkpoint_path(output_path) if is_partial else output_path),
            metadata=_grade_metadata(
                config,
                testee_metadata,
                responses_path,
                is_partial=is_partial,
                total_cases_evaluated=len(all_results),
            ),
        )

        if not args.quiet and is_partial:
            print(f"  Checkpoint: Graded {len(all_results)} total responses")

    def progress_callback(results: List[HealthBenchRobustnessResult]):
        save_graded_results(results, is_partial=True)

    if not testee_responses:
        results = merge_healthbench_results(existing_results, [])
        save_graded_results(results, is_partial=False)
        remove_checkpoint(output_path)
        if not args.quiet:
            print("No HealthBench baseline responses remain to grade.")
        return 0

    # Initialize the provider only after confirming work remains.
    model_pool = ModelPool()
    grader = RubricGrader(
        model_id=config.grader_model,
        model_pool=model_pool,
        config=config.grader_config,
        verbose=args.verbose
    )
    pipeline = HealthBenchRobustnessPipeline(
        testee=SimpleNamespace(model_id=response_testee_model),
        grader=grader,
        attack_strategies=[],
        verbose=not args.quiet,
        bootstrap_seed=0,
    )

    # Run grading with error handling
    if not args.quiet:
        print(f"\n[Grading Testee Responses]")

    try:
        new_results = evaluate_items_fail_fast(
            items=testee_responses,
            evaluate_fn=grade_response,
            description="Grading",
            progress_callback=progress_callback,
            save_every=10,
            verbose=not args.quiet,
            use_tqdm=True
        )

        # Combine with existing results if resuming
        results = merge_healthbench_results(existing_results, new_results)

    except KeyboardInterrupt:
        print(f"\n\n[WARNING] Grading interrupted by user")
        return 1

    except Exception as e:
        print(f"\n[ERROR] Unexpected error during grading: {e}")
        import traceback
        traceback.print_exc()
        return 1

    summary = pipeline._compute_summary(results, baseline_only=True)

    if not args.quiet:
        print(f"\n[Saving Graded Results]")

    save_graded_results(results, is_partial=False)
    remove_checkpoint(output_path)

    # Print summary
    if not args.quiet:
        print(f"\n{'='*80}")
        print("GRADING COMPLETE")
        print(f"{'='*80}")
        print(f"\nGraded results saved to: {output_path}")
        print(f"Total graded: {len(results)}")
        print(f"\nSummary:")
        print(f"  Average score: {summary.baseline_avg_score:.3f}")
        print(f"  Score std dev: {summary.baseline_score_std:.3f}")
        print(f"  Total rubrics met: {summary.total_rubrics_met_baseline}")
        dataset_arg = healthbench_dataset_path(testee_metadata) or '<dataset-path>'
        print(f"\nTo run attack evaluation using this baseline:")
        print(f"  python -m scripts.healthbench.run_attack \\")
        print(f"      --baseline-results {output_path} \\")
        print(f"      --dataset {dataset_arg}")

    return 0


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run baseline HealthBench evaluation in split mode."""
    # Parse arguments
    args = parse_args()

    # Create config
    config = create_config_from_args(args)

    # Route to appropriate mode
    if args.mode == "collect":
        return run_collect_mode(args, config)
    elif args.mode == "grade":
        return run_grade_mode(args, config)
    else:
        print(f"[ERROR] Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
