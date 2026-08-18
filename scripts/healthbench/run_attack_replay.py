#!/usr/bin/env python3
"""
HealthBench Attack Replay Evaluation
=====================================

Replay pre-generated attacks (e.g., from o3 model) on different testee models.
This allows you to test multiple models against the same attacks without regenerating them.

The script reads attack results from a previous run and extracts the attacked conversations
to test new models directly.

Usage:
    # Stage 1: Collect responses using pre-generated attacks
    python -m scripts.healthbench.run_attack_replay \
        --mode collect \
        --attack-results logs/healthbench/o3/impossible_measurement_100.json \
        --testee-model gemma-2-27b-it

    # Stage 2: Grade collected responses (--attack-results not needed!)
    python -m scripts.healthbench.run_attack_replay \
        --mode grade \
        --testee-responses logs/healthbench/gemma-2-27b-it/collect_*.json \
        --grader-model gemini-2.5-flash

Example:
    >>> # Replay o3's attacks on gemma
    >>> python -m scripts.healthbench.run_attack_replay \
    ...     --mode collect \
    ...     --attack-results logs/healthbench/o3/impossible_measurement_100.json \
    ...     --testee-model gemma-2-27b-it

    >>> # Grade the gemma responses (all info is in the responses file)
    >>> python -m scripts.healthbench.run_attack_replay \
    ...     --mode grade \
    ...     --testee-responses logs/healthbench/gemma-2-27b-it/collect_*.json \
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
from med_red_team.healthbench.data import (
    HealthBenchRobustnessResult,
    RubricItem,
    RubricGradeResult,
    AttackResult,
    build_healthbench_metadata,
    healthbench_attack_strategy_config,
    healthbench_attack_strategy_order,
    healthbench_dataset_path,
    healthbench_source_artifact_identity,
    _healthbench_generation_config,
    _healthbench_model_id,
    _healthbench_testee_system_prompt,
    validate_healthbench_artifact_metadata,
)
from med_red_team.healthbench.utils import calculate_score
from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.models import GenerationConfig
from med_red_team.shared.checkpoints import checkpoint_path, remove_checkpoint
from med_red_team.shared.config_loading import clone_config, resolve_sample_limit
from med_red_team.shared.io import atomic_write_result_envelope
from med_red_team.utils import evaluate_items_fail_fast

from scripts.healthbench.checkpoints import (
    merge_healthbench_results,
    validate_healthbench_checkpoint_metadata,
    validate_healthbench_resume_rows,
)
from scripts.utils import (
    generate_output_path,
    load_config_from_py,
    load_results_from_json,
    validate_output_path,
    reconstruct_healthbench_result
)

# Import baseline config as default
from configs.examples.healthbench.baseline import CONFIG as BASELINE_CONFIG


# ==============================================================================
# Metadata compatibility readers
# ==============================================================================


def _metadata_model_id(metadata: dict, role: str = "testee") -> str | None:
    return _healthbench_model_id(metadata, role)


def _metadata_dataset_path(metadata: dict) -> str | None:
    return healthbench_dataset_path(metadata)


def _metadata_attack_strategies(metadata: dict) -> list[str]:
    return healthbench_attack_strategy_order(metadata)


def _metadata_attacker_config(metadata: dict) -> dict:
    return healthbench_attack_strategy_config(metadata)


def _metadata_source_value(metadata: dict, key: str, *legacy_keys: str):
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    for source_key in (key, *legacy_keys):
        if source.get(source_key) is not None:
            return source[source_key]
    for legacy_key in (key, *legacy_keys):
        if metadata.get(legacy_key) is not None:
            return metadata[legacy_key]
    return None


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Replay pre-generated attacks on different testee models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect testee responses using pre-generated attacks
  python -m scripts.healthbench.run_attack_replay \\
      --mode collect \\
      --attack-results logs/healthbench/o3/impossible_measurement_100.json \\
      --testee-model gemma-2-27b-it

  # Grade collected responses (attack metadata embedded in responses file)
  python -m scripts.healthbench.run_attack_replay \\
      --mode grade \\
      --testee-responses logs/healthbench/gemma-2-27b-it/collect_*.json \\
      --grader-model gemini-2.5-flash
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
        help="Path to config .py file (optional, e.g., configs/examples/healthbench/baseline.py)"
    )

    # Attack results - source of attacks to replay (required for collect, optional for grade)
    parser.add_argument(
        "--attack-results",
        type=str,
        default=None,
        help="Path to attack results JSON file (required for collect mode, optional for grade mode)"
    )

    # Model settings
    parser.add_argument(
        "--testee-model",
        type=str,
        default=None,
        help="Model ID for the testee (required for collect mode)"
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help="Model ID for rubric grading (required for grade mode)"
    )

    # Testee configuration
    parser.add_argument(
        "--testee-system-prompt",
        type=str,
        default=None,
        help="System prompt for testee (optional, uses default if not specified)"
    )
    parser.add_argument(
        "--testee-temperature",
        type=float,
        default=None,
        help="Temperature for testee generation (overrides config)"
    )

    # Grader configuration
    parser.add_argument(
        "--grader-temperature",
        type=float,
        default=None,
        help="Temperature for grader generation (overrides config)"
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
        help="Maximum number of attacks to replay"
    )
    parser.add_argument(
        "--skip-non-applicable",
        action="store_true",
        help="Skip attacks that were not applicable in the original run"
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


def _override_temperature(
    config: GenerationConfig,
    temperature: float,
) -> GenerationConfig:
    payload = config.to_dict()
    payload["temperature"] = temperature
    return GenerationConfig(**payload)


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
        # Start with independent baseline defaults for testee/grader settings.
        if not args.quiet:
            print(f"[Config] Using baseline config as default")
        config = clone_config(BASELINE_CONFIG)

    # Override with command-line arguments
    overrides = {
        "max_samples": resolve_sample_limit(args.max_samples, config.max_samples),
    }

    # Model settings
    if args.testee_model is not None:
        overrides["testee_model"] = args.testee_model
    if args.grader_model is not None:
        overrides["grader_model"] = args.grader_model

    # Testee config overrides
    if args.testee_temperature is not None:
        overrides["testee_config"] = _override_temperature(
            config.testee_config,
            args.testee_temperature,
        )
    if args.testee_system_prompt is not None:
        overrides["testee_system_prompt"] = args.testee_system_prompt

    # Grader config overrides
    if args.grader_temperature is not None:
        overrides["grader_config"] = _override_temperature(
            config.grader_config,
            args.grader_temperature,
        )

    # Output settings
    if args.output_dir is not None:
        overrides["log_dir"] = args.output_dir

    # Create new config with overrides
    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = HealthBenchConfig.from_dict(config_dict)

    return config


def _replay_collect_metadata(
    config: HealthBenchConfig,
    attack_metadata: dict,
    attack_results_path: Path,
    *,
    is_partial: bool,
    total_collected: int,
    skip_non_applicable: bool,
) -> dict:
    dataset_info = attack_metadata.get("dataset") if isinstance(attack_metadata.get("dataset"), dict) else {}
    source_models = attack_metadata.get("models") if isinstance(attack_metadata.get("models"), dict) else {}
    source_dataset = attack_metadata.get("dataset") if isinstance(attack_metadata.get("dataset"), dict) else {}
    source_config = attack_metadata.get("config") if isinstance(attack_metadata.get("config"), dict) else {}
    attack_source_identity = healthbench_source_artifact_identity(attack_results_path)
    return build_healthbench_metadata(
        phase="attack_replay_collect",
        config=config,
        is_partial=is_partial,
        dataset=dataset_info or {
            "path": _metadata_dataset_path(attack_metadata),
            "sample_limit": config.max_samples,
            "filter_single_turn": None,
        },
        source={
            "source_results_file": str(attack_results_path),
            **attack_source_identity,
            "source_schema_version": attack_metadata.get("schema_version"),
            "source_axis": attack_metadata.get("axis") or attack_metadata.get("evaluation_type"),
            "source_phase": attack_metadata.get("phase") or attack_metadata.get("mode"),
            "source_models": source_models,
            "source_dataset": source_dataset,
            "source_config": source_config,
            "source_testee_model": _metadata_model_id(attack_metadata),
            "workflow": "run_attack_replay_collect",
        },
        attack_strategies=_metadata_attack_strategies(attack_metadata),
        attack_strategy_config=_metadata_attacker_config(attack_metadata),
        extra={
            "skip_non_applicable": skip_non_applicable,
            "replay_scope": "impossible_measurement_only",
            "total_collected": total_collected,
            "bootstrap_seed": 0,
            "bootstrap_clips_mean_to_unit_interval": True,
        },
    )



def _replay_grade_metadata(
    config: HealthBenchConfig,
    responses_metadata: dict,
    responses_path: Path,
    *,
    is_partial: bool,
    total_cases_evaluated: int,
) -> dict:
    config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    source_config = responses_metadata.get("config") if isinstance(responses_metadata.get("config"), dict) else {}
    dataset_info = responses_metadata.get("dataset") if isinstance(responses_metadata.get("dataset"), dict) else {}
    models = responses_metadata.get("models") if isinstance(responses_metadata.get("models"), dict) else {}
    testee_data = models.get("testee") if isinstance(models.get("testee"), dict) else {}
    config_dict["dataset_path"] = (
        dataset_info.get("path")
        or source_config.get("dataset_path")
        or _metadata_dataset_path(responses_metadata)
        or config_dict.get("dataset_path")
    )
    config_dict["testee_model"] = (
        testee_data.get("model_id")
        or source_config.get("testee_model")
        or _metadata_model_id(responses_metadata)
        or config_dict.get("testee_model")
    )
    config_dict["testee_config"] = (
        testee_data.get("generation_config")
        if testee_data.get("generation_config") is not None
        else source_config.get("testee_config", responses_metadata.get("testee_config", config_dict.get("testee_config")))
    )
    config_dict["testee_system_prompt"] = (
        testee_data.get("system_prompt")
        if testee_data.get("system_prompt") is not None
        else source_config.get("testee_system_prompt", responses_metadata.get("testee_system_prompt", config_dict.get("testee_system_prompt")))
    )
    source_meta = responses_metadata.get("source") if isinstance(responses_metadata.get("source"), dict) else {}
    responses_identity = healthbench_source_artifact_identity(
        responses_path,
        key_prefix="testee_responses",
    )
    return build_healthbench_metadata(
        phase="attack_replay",
        config=config_dict,
        is_partial=is_partial,
        dataset=dataset_info or {
            "path": config_dict.get("dataset_path"),
            "sample_limit": source_config.get("max_samples"),
            "filter_single_turn": None,
        },
        source={
            "source_results_file": str(responses_path),
            "source_results_path": responses_identity.get("testee_responses_path"),
            "source_results_sha256": responses_identity.get("testee_responses_sha256"),
            "source_schema_version": responses_metadata.get("schema_version"),
            "source_axis": responses_metadata.get("axis"),
            "source_phase": responses_metadata.get("phase"),
            "source_models": responses_metadata.get("models", {}),
            "source_dataset": responses_metadata.get("dataset", {}),
            "source_config": responses_metadata.get("config", {}),
            "source_testee_model": _metadata_model_id(responses_metadata),
            "testee_responses_file": str(responses_path),
            **responses_identity,
            "upstream_attack": source_meta,
            "workflow": "run_attack_replay_grade",
        },
        attack_strategies=_metadata_attack_strategies(responses_metadata),
        attack_strategy_config=_metadata_attacker_config(responses_metadata),
        extra={
            "testee_responses_file": str(responses_path),
            "replay_scope": "impossible_measurement_only",
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
    Run collect mode: Get testee responses using pre-generated attacks.

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

    if args.attack_results is None:
        print("[ERROR] --attack-results is required for collect mode")
        return 1

    # Load attack results
    if not args.quiet:
        print(f"\n[Loading Attack Results]")

    attack_results_path = Path(args.attack_results)
    try:
        attack_results, attack_metadata = load_results_from_json(
            attack_results_path,
            data_key="results",
            quiet=args.quiet,
            reconstruct_fn=reconstruct_healthbench_result
        )
        validate_healthbench_artifact_metadata(
            attack_metadata,
            phase="attack",
            label=f"HealthBench replay source {attack_results_path}",
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    # Filter to applicable attacks if requested
    if args.skip_non_applicable:
        applicable_attacks = [r for r in attack_results if r.attack_result and r.attack_result.applicable]
        if not args.quiet:
            print(f"  Filtered to {len(applicable_attacks)} applicable attacks (out of {len(attack_results)})")
        attack_results = applicable_attacks

    # Filter to attacks with attacked_conversation
    valid_attacks = [r for r in attack_results if r.attacked_conversation is not None]
    if len(valid_attacks) < len(attack_results) and not args.quiet:
        print(f"  Filtered to {len(valid_attacks)} attacks with valid attacked_conversation")
    attack_results = valid_attacks

    if not attack_results:
        print("[ERROR] No valid attacks found to replay")
        return 1

    # Apply max_samples limit
    if config.max_samples is not None:
        attack_results = attack_results[:config.max_samples]

    strategies = _metadata_attack_strategies(attack_metadata)
    if strategies != ["impossible_measurement"]:
        print(
            "[ERROR] HealthBench replay only supports impossible_measurement attack artifacts. "
            "Selected-rubric attacks are model-dependent and must not be replayed."
        )
        return 1

    source_attack_model = _metadata_model_id(attack_metadata) or "unknown"
    attack_source_identity = healthbench_source_artifact_identity(attack_results_path)

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

        source_dataset = (
            attack_metadata.get("dataset")
            if isinstance(attack_metadata.get("dataset"), dict)
            else {}
        )
        try:
            validate_healthbench_checkpoint_metadata(
                resume_metadata,
                phase="attack_replay_collect",
                expected_testee_model=config.testee_model,
                expected_testee_config=config.testee_config.to_dict(),
                expected_testee_system_prompt=config.testee_system_prompt,
                expected_source_results_sha256=attack_source_identity.get(
                    "source_results_sha256"
                ),
                expected_config_sample_limit=config.max_samples,
                expected_dataset_sample_limit=source_dataset.get("sample_limit"),
                expected_attack_strategies=strategies,
                expected_attack_strategy_config=_metadata_attacker_config(
                    attack_metadata
                ),
                expected_extra={
                    "skip_non_applicable": args.skip_non_applicable,
                    "replay_scope": "impossible_measurement_only",
                },
            )
            processed_ids = validate_healthbench_resume_rows(
                existing_responses,
                allowed_case_ids=(result.test_case_id for result in attack_results),
            )
        except ValueError as exc:
            print(f"[ERROR] Cannot resume - {exc}")
            return 1
        if not args.quiet:
            print(f"  Found {len(existing_responses)} existing responses")

    # Filter out already processed attacks
    if processed_ids:
        unprocessed_attacks = [a for a in attack_results if a.test_case_id not in processed_ids]
        if not args.quiet:
            print(f"  {len(unprocessed_attacks)} attacks remaining to collect (resuming)")
        attack_results = unprocessed_attacks

    # Print configuration
    if not args.quiet:
        print("=" * 80)
        print("HEALTHBENCH ATTACK REPLAY - COLLECT MODE")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Source attack results: {attack_results_path.name}")
        print(f"  Source attack model: {source_attack_model}")
        print(f"  Attack strategies: {', '.join(_metadata_attack_strategies(attack_metadata))}")
        print(f"  Testee model (replay): {config.testee_model}")
        print(f"  Attacks to replay: {len(attack_results)}")
        if args.skip_non_applicable:
            print(f"  Skipping non-applicable attacks: Yes")
        if resume_path:
            print(f"  Resuming from: {resume_path.name}")

    # Generate output path
    total_expected = len(existing_responses) + len(attack_results)
    # Extract strategy names from attack metadata
    strategies = _metadata_attack_strategies(attack_metadata)
    output_path = generate_output_path(
        log_dir=config.log_dir,
        testee_model=config.testee_model,
        mode="replay_collect",
        strategies=strategies,
        max_samples=total_expected if existing_responses else len(attack_results)
    )

    # Validate output path is writable before starting collection
    if not validate_output_path(output_path, quiet=args.quiet):
        return 1

    if resume_path and existing_responses:
        if not args.quiet:
            print(f"\n[Resume Mode]")
            print(f"  Resuming from: {resume_path.name}")
            print(f"  Saving combined responses to: {output_path.name}")

    def collect_response(result: HealthBenchRobustnessResult, idx: int) -> dict:
        """Collect testee response for a single attacked conversation."""
        conv_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in result.attacked_conversation
        ])

        response = testee.answer(conv_text)
        completion = response.final_answer

        return {
            "test_case_id": result.test_case_id,
            "sample_number": idx + 1,
            "original_conversation": result.original_conversation,
            "original_rubrics": [r.to_dict() for r in result.original_rubrics],
            "baseline_completion": None,
            "baseline_grades": [],
            "baseline_score": None,
            "attacked_conversation": result.attacked_conversation,
            "attacked_rubrics": [r.to_dict() for r in result.attacked_rubrics] if result.attacked_rubrics else [r.to_dict() for r in result.original_rubrics],
            "attack_result": result.attack_result.to_dict() if result.attack_result else None,
            "completion": completion,
            "parse_error": response.metadata.get("parse_error"),
        }

    def save_responses(responses: List[dict], *, is_partial: bool):
        """Save replay responses to a stable result envelope."""
        all_responses = merge_healthbench_results(existing_responses, responses)
        target_path = checkpoint_path(output_path) if is_partial else output_path
        atomic_write_result_envelope(
            target_path,
            metadata=_replay_collect_metadata(
                config,
                attack_metadata,
                attack_results_path,
                is_partial=is_partial,
                total_collected=len(all_responses),
                skip_non_applicable=args.skip_non_applicable,
            ),
            summary=_responses_summary(len(all_responses)),
            data=all_responses,
            data_key="responses",
        )

        if not args.quiet and is_partial:
            print(f"  Checkpoint: Saved {len(all_responses)} total responses to {target_path.name}")

    def progress_callback(responses: List[dict]):
        save_responses(responses, is_partial=True)

    if not attack_results:
        responses = merge_healthbench_results(existing_responses, [])
        save_responses(responses, is_partial=False)
        remove_checkpoint(output_path)
        if not args.quiet:
            print("No HealthBench replay attacks remain to collect.")
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
            items=attack_results,
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
        print(f"  python -m scripts.healthbench.run_attack_replay \\")
        print(f"      --mode grade \\")
        print(f"      --testee-responses {output_path} \\")
        print(f"      --grader-model <your-grader-model>")
        print(f"\n  Note: Attack metadata is embedded in the responses file, no need for --attack-results!")

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
            phase="attack_replay_collect",
            label=f"HealthBench replay responses {responses_path}",
        )
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1

    # Extract attack metadata from testee_metadata (saved during collect mode)
    response_testee_model = _metadata_model_id(testee_metadata) or "unknown"
    source_testee_model = _metadata_source_value(
        testee_metadata, "source_testee_model"
    ) or response_testee_model
    attack_strategies = _metadata_attack_strategies(testee_metadata)
    if attack_strategies != ["impossible_measurement"]:
        print(
            "[ERROR] HealthBench replay grading only supports impossible_measurement responses. "
            "Selected-rubric attacks are model-dependent and must not be replayed."
        )
        return 1
    attacker_strategies = _metadata_attacker_config(testee_metadata)
    attack_results_path_str = _metadata_source_value(
        testee_metadata, "source_results_file", "source_attack_results"
    ) or 'unknown'
    responses_identity = healthbench_source_artifact_identity(
        responses_path,
        key_prefix="testee_responses",
    )

    if not args.quiet:
        print(f"  Using attack metadata from testee responses file")
        print(f"  Source attack results: {attack_results_path_str}")
        print(f"  Source testee model (original): {source_testee_model}")
        print(f"  Attack strategies: {', '.join(attack_strategies)}")

        # Show attacker models if available
        if attacker_strategies:
            attacker_models = [f"{s}: {cfg.get('model_id', 'unknown')}" for s, cfg in attacker_strategies.items()]
            print(f"  Attacker models: {', '.join(attacker_models)}")

    # Handle resume
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
            reconstruct_fn=reconstruct_healthbench_result,
        )

        response_dataset = (
            testee_metadata.get("dataset")
            if isinstance(testee_metadata.get("dataset"), dict)
            else {}
        )
        try:
            validate_healthbench_checkpoint_metadata(
                resume_metadata,
                phase="attack_replay",
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
                expected_attack_strategies=attack_strategies,
                expected_attack_strategy_config=attacker_strategies,
                expected_extra={
                    "replay_scope": "impossible_measurement_only",
                },
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
        print("HEALTHBENCH ATTACK REPLAY - GRADE MODE")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Source testee model (original): {source_testee_model}")
        print(f"  Attack strategies: {', '.join(attack_strategies)}")
        print(f"  Testee model (replay): {response_testee_model}")
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
        mode="replay",
        strategies=attack_strategies,
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
        attacked_conversation = response["attacked_conversation"]
        completion = response["completion"]

        if not response.get("attacked_rubrics"):
            raise ValueError(f"Test case {test_case_id} missing attacked_rubrics in response file")

        original_rubrics = [
            RubricItem.from_dict(r) for r in response.get("original_rubrics", [])
        ]
        rubrics = [RubricItem.from_dict(r) for r in response["attacked_rubrics"]]
        attack_result_data = response.get("attack_result")
        attack_result = (
            AttackResult.from_dict(attack_result_data)
            if attack_result_data else None
        )

        parse_error = response.get("parse_error")
        if parse_error:
            reason = parse_error.get("error_message", "Unknown parse error") if isinstance(parse_error, dict) else str(parse_error)
            return HealthBenchRobustnessResult(
                test_case_id=test_case_id,
                sample_number=response["sample_number"],
                original_conversation=response.get("original_conversation", []),
                original_rubrics=original_rubrics,
                baseline_completion=None,
                baseline_grades=[],
                baseline_score=None,
                attack_result=attack_result,
                attacked_conversation=attacked_conversation,
                attacked_rubrics=rubrics,
                attacked_completion=completion,
                attacked_grades=[],
                attacked_score=None,
                skipped=True,
                skip_reason=reason,
            )

        grades = grader.grade(
            conversation=attacked_conversation,
            completion=completion,
            rubric_items=rubrics
        )
        score = calculate_score(rubrics, grades)

        result = HealthBenchRobustnessResult(
            test_case_id=test_case_id,
            sample_number=response["sample_number"],
            original_conversation=response.get("original_conversation", []),
            original_rubrics=original_rubrics,
            baseline_completion=None,
            baseline_grades=[],
            baseline_score=None,
            attack_result=attack_result,
            attacked_conversation=attacked_conversation,
            attacked_rubrics=rubrics,
            attacked_completion=completion,
            attacked_grades=grades,
            attacked_score=score,
        )
        validation_error = pipeline._grading_validation_error(
            rubrics,
            grades,
            label="HealthBench replay attacked grades",
        )
        if validation_error is not None:
            result.skipped = True
            result.skip_reason = validation_error
        return result

    def save_graded_results(results: List[HealthBenchRobustnessResult], *, is_partial: bool):
        """Save replay grading results through the shared HealthBench envelope path."""
        all_results = merge_healthbench_results(existing_results, results)
        summary = pipeline._compute_summary(all_results, baseline_only=False)
        pipeline.save_results(
            results=all_results,
            summary=summary,
            output_path=str(checkpoint_path(output_path) if is_partial else output_path),
            metadata=_replay_grade_metadata(
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
            print("No HealthBench replay responses remain to grade.")
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

    summary = pipeline._compute_summary(results, baseline_only=False)

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
        print(f"  Attacks attempted: {summary.attacks_attempted}")
        print(f"  Attacks applicable: {summary.attacks_applicable}")
        print(f"  Comparable attacks: {summary.attacks_comparable}")
        applicable_rate = (
            summary.attacks_applicable / summary.attacks_attempted * 100
            if summary.attacks_attempted > 0 else 0
        )
        print(f"  Applicability rate: {applicable_rate:.1f}%")
        if summary.attacked_avg_score is not None:
            attacked_std = summary.attacked_score_std or 0.0
            print(f"  Attacked avg score: {summary.attacked_avg_score:.3f}")
            print(f"  Attacked score std: {attacked_std:.3f}")
        else:
            print("  Attacked avg score: unavailable for standalone impossible-measurement replay")
        print(f"  Rubric attack successes: {summary.total_rubric_attack_successes}")

    return 0


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run attack replay evaluation."""
    # Parse arguments
    args = parse_args()

    # Create config from args
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
