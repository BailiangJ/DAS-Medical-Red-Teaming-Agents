#!/usr/bin/env python3
"""
Robustness Attack Testing (Round 2)
====================================

Apply attack strategies to questions answered correctly in baseline.

This script loads baseline results and applies attack strategies to test cases
that were answered correctly in Round 1, measuring the model's robustness.

Usage:
    # Basic usage with baseline results
    python -m scripts.robustness.run_attack \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json

    # Specify attack strategies
    python -m scripts.robustness.run_attack \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
        --strategies bias_manipulation add_distraction_sentence

Example:
    >>> python -m scripts.robustness.run_attack \
    ...     --baseline-results logs/gemini-2.5-flash/robustness_baseline_20250115_143022.json \
    ...     --strategies bias_manipulation

    Attack testing completed!
    Results saved to: logs/gemini-2.5-flash/robustness_attack_20250115_150322.json
"""

import sys
import argparse
from pathlib import Path
from typing import List, Sequence

from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessResult,
    build_robustness_metadata,
    robustness_file_identity,
    robustness_resume_signature,
    strategy_order_from_config,
    validate_fixed_strategy_chain,
    validate_robustness_baseline_metadata,
    is_baseline_result_eligible_for_attack,
)
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.testee import Testee
from med_red_team.grader import SimpleGrader
from med_red_team.model_pool import ModelPool
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint
from med_red_team.shared.config_loading import resolve_sample_limit
from scripts.utils import (
    generate_and_validate_output_path,
    load_config_from_py,
    create_attack_strategies_from_config,
    load_results_from_json,
    validate_resume,
    override_attacker_strategies
)

# Import default config for strategy fallback
from configs.examples.robustness.attack import CONFIG as DEFAULT_CONFIG


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run attack robustness evaluation (Round 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline results
  python -m scripts.robustness.run_attack \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json

  # Test specific strategies
  python -m scripts.robustness.run_attack \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --strategies bias_manipulation add_distraction_sentence

  # Limit number of cases to attack
  python -m scripts.robustness.run_attack \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --max-samples 50

  # Load attack config from file
  python -m scripts.robustness.run_attack \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --config configs/examples/robustness/attack.py
        """
    )

    # Required argument
    parser.add_argument(
        "--baseline-results",
        dest="baseline_results",
        required=True,
        type=str,
        help="Path to JSON file with baseline results from run_baseline.py"
    )

    # Configuration source
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (overrides other args)"
    )

    # Attack strategies
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Attack strategies to use (default: from baseline config)"
    )

    # Model settings
    parser.add_argument(
        "--testee-model",
        type=str,
        help="Model to test (must match baseline, default: same as baseline)"
    )

    # Attack settings
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of first-round correct samples to attack (default: all)"
    )

    # Output settings
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Directory for saving results (default: same as baseline)"
    )

    # Resume settings
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to existing attack results file to resume from (.inprogress or completed)"
    )

    # Display settings
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    return parser.parse_args(argv)


# ==============================================================================
# Configuration Setup
# ==============================================================================

def load_baseline_data(baseline_path: str, quiet: bool = False):
    """
    Load complete strict-v2 baseline results and configuration.

    Returns:
        Tuple of (baseline_results_correct, config, dataset_info, metadata).
    """
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"[ERROR] Baseline results file not found: {baseline_path}")
        sys.exit(1)

    if not quiet:
        print(f"[Loading] Baseline results from: {baseline_path}")

    try:
        baseline_results, baseline_metadata = load_results_from_json(
            baseline_path,
            data_key="results",
            quiet=quiet,
            reconstruct_fn=RobustnessResult.from_dict,
        )
        validate_robustness_baseline_metadata(
            baseline_metadata,
            expected_evaluation_mode="original",
        )
    except Exception as exc:
        print(f"[ERROR] Invalid baseline results: {exc}")
        sys.exit(1)

    config = RobustnessConfig.from_dict(baseline_metadata["config"])
    dataset_info = dict(baseline_metadata.get("dataset", {}))

    # Only completed, first-round-correct baseline rows are attackable.
    baseline_results_correct = [
        r for r in baseline_results
        if is_baseline_result_eligible_for_attack(r)
    ]

    if not quiet:
        total_results = len(baseline_results)
        correct_results = len(baseline_results_correct)
        print(f"  Loaded {total_results} baseline results")
        print(f"  First round correct: {correct_results} cases")

    if not baseline_results_correct and not quiet:
        print("\n[WARNING] No cases were answered correctly in baseline.")
        print("A complete zero-result attack artifact will be written.")

    return baseline_results_correct, config, dataset_info, baseline_metadata


def create_config_from_args(
    args: argparse.Namespace,
    baseline_config: RobustnessConfig,
    quiet: bool = False
) -> RobustnessConfig:
    """
    Create attack config from command-line arguments and baseline config.

    Args:
        args: Parsed command-line arguments
        baseline_config: Configuration from baseline results

    Returns:
        RobustnessConfig instance for attack testing
    """
    # Load from config file if specified, otherwise use baseline config
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, RobustnessConfig)
    else:
        # Start with baseline config
        config = baseline_config

    # Override with command-line arguments (applies to both config file and baseline)
    overrides = {}

    if args.testee_model:
        overrides["testee_model"] = args.testee_model

    # Attack strategies (use shared utility)
    new_strategies = override_attacker_strategies(
        config=config,
        args_strategies=args.strategies,
        args_attacker_model=None,  # Not used - strategies already have model configs
        default_config=DEFAULT_CONFIG,
        quiet=quiet
    )
    if new_strategies:
        overrides["attacker_strategies"] = new_strategies

    if args.log_dir:
        overrides["log_dir"] = args.log_dir

    # Create new config with overrides
    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = RobustnessConfig.from_dict(config_dict)

    return config


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run attack robustness evaluation."""
    # Parse arguments
    args = parse_args()

    # Load baseline data (self-contained results with all test case info)
    baseline_results, baseline_config, dataset_info, baseline_metadata = load_baseline_data(
        args.baseline_results,
        quiet=args.quiet
    )
    baseline_identity = robustness_file_identity(args.baseline_results)
    baseline_source = {
        "baseline_results_file": baseline_identity["path"],
        "baseline_results_sha256": baseline_identity["sha256"],
    }

    # Create attack config and resolve sampling: CLI override wins over CONFIG.
    config = create_config_from_args(args, baseline_config, quiet=args.quiet)
    try:
        sample_limit = resolve_sample_limit(args.max_samples, config.max_samples)
    except ValueError as exc:
        print(f"[ERROR] Invalid sample limit: {exc}")
        return 1
    if sample_limit != config.max_samples:
        config_dict = config.to_dict()
        config_dict["max_samples"] = sample_limit
        config = RobustnessConfig.from_dict(config_dict)

    # Validate testee_model and the full target contract match the strict v2 baseline.
    config_dict = config.to_dict()
    try:
        baseline_target_model = validate_robustness_baseline_metadata(
            baseline_metadata,
            expected_target_model=config.testee_model,
            expected_target_contract={
                "model_id": config.testee_model,
                "generation_config": config_dict.get("testee_config", {}),
                "system_prompt": config.testee_system_prompt,
            },
            expected_evaluation_mode="original",
        )
    except Exception as exc:
        print(f"\n[ERROR] Baseline metadata validation failed: {exc}")
        return 1
    if config.testee_model != baseline_target_model:
        print(f"\n[ERROR] Testee model mismatch!")
        print(f"  Baseline results were generated with: {baseline_target_model}")
        print(f"  Current testee model is: {config.testee_model}")
        return 1

    # Apply the effective sample limit before deciding whether attack work exists.
    if sample_limit is not None and len(baseline_results) > sample_limit:
        if not args.quiet:
            print("\n[Limiting Samples]")
            print(f"  Total baseline correct cases: {len(baseline_results)}")
            print(f"  Limiting to first {sample_limit} cases")
        baseline_results = baseline_results[:sample_limit]

    # Empty chains are valid only when no attack work exists.
    configured_strategy_names = strategy_order_from_config(config)
    if baseline_results or configured_strategy_names:
        try:
            strategy_names = validate_fixed_strategy_chain(config)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 1
    else:
        strategy_names = []

    # Handle resume
    existing_results = []
    processed_ids = set()
    resume_path = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {args.resume}")
            return 1

        if not args.quiet:
            print(f"\n[Loading Resume File]")

        existing_results, resume_metadata = load_results_from_json(
            resume_path,
            data_key="results",
            quiet=args.quiet,
            reconstruct_fn=RobustnessResult.from_dict
        )

        expected_metadata = build_robustness_metadata(
            phase="attack",
            config=config,
            target_model=config.testee_model,
            dataset_info=dataset_info or {},
            source=baseline_source,
            is_partial=False,
            sample_limit=sample_limit,
            attack_strategies=strategy_names,
            baseline_results_file=baseline_identity["path"],
        )
        expected_signature = robustness_resume_signature(expected_metadata)
        resume_signature = robustness_resume_signature(resume_metadata)
        validations = {
            key: (value, key.replace("_", " ").title())
            for key, value in expected_signature.items()
        }

        is_valid, msg = validate_resume(
            resume_signature,
            validations,
            quiet=args.quiet,
        )
        if not is_valid:
            print(f"[ERROR] Resume validation failed: {msg}")
            return 1

        resume_ids = [result.test_case_id for result in existing_results]
        if len(set(resume_ids)) != len(resume_ids):
            print("[ERROR] Resume file contains duplicate test_case_id values")
            return 1
        scoped_ids = {result.test_case_id for result in baseline_results}
        unexpected_ids = sorted(set(resume_ids).difference(scoped_ids))
        if unexpected_ids:
            print(
                "[ERROR] Resume file contains results outside the current "
                f"sampled baseline: {unexpected_ids[:5]}"
            )
            return 1
        processed_ids = set(resume_ids)

        if not args.quiet:
            print(f"  Found {len(processed_ids)} already-evaluated attack cases")

    # Filter baseline_results to exclude already-processed cases if resuming.
    if processed_ids:
        original_baseline_count = len(baseline_results)
        baseline_results = [
            r for r in baseline_results
            if r.test_case_id not in processed_ids
        ]
        if not args.quiet:
            print(f"\n[Filtering Baseline Results]")
            print(f"  Original baseline cases: {original_baseline_count}")
            print(f"  Already attacked: {len(processed_ids)}")
            print(f"  Remaining to attack: {len(baseline_results)}")

    # Print configuration before constructing providers.
    if not args.quiet:
        print("\n" + "=" * 80)
        print("ROBUSTNESS ATTACK TESTING (Round 2)")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Testee model: {config.testee_model}")
        attack_str = ', '.join([
            f"{name}[{config.attacker_strategies.get(name, {}).get('model_id', 'unknown')}]"
            for name in strategy_names
        ])
        print(f"  Attack strategies: {attack_str}")
        print(f"  Max samples: {sample_limit if sample_limit is not None else 'all'}")
        print(f"  Attacking {len(baseline_results)} correctly-answered cases from baseline")

    # Determine log directory
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = Path(args.baseline_results).parent

    # Generate and validate output path before model construction.
    total_expected = (
        len(existing_results) + len(baseline_results)
        if existing_results else len(baseline_results)
    )
    output_path = generate_and_validate_output_path(
        log_dir=str(log_dir),
        testee_model=config.testee_model,
        mode="robustness_attack",
        strategies=strategy_names,
        max_samples=total_expected if sample_limit is not None else None,
        quiet=args.quiet
    )

    if output_path is None:
        print("[ERROR] Cannot write to output directory. Exiting.")
        return 1

    def remove_run_checkpoint():
        remove_checkpoint(output_path)

    def make_metadata(is_partial: bool):
        return build_robustness_metadata(
            phase="attack",
            config=config,
            target_model=config.testee_model,
            dataset_info=dataset_info or {},
            source=baseline_source,
            is_partial=is_partial,
            sample_limit=sample_limit,
            attack_strategies=strategy_names,
            baseline_results_file=baseline_identity["path"],
        )

    # Zero-work/all-complete paths must not construct providers.
    if len(baseline_results) == 0:
        if not args.quiet:
            print("\n[INFO] No remaining baseline cases to attack. Saving final artifact.")
            if existing_results:
                print(f"  Resume file already contains {len(existing_results)} attack results")
        pipeline = RobustnessPipeline(
            testee=None,
            grader=None,
            attack_strategies=[],
            verbose=not args.quiet,
        )
        results = existing_results
        summary = pipeline._compute_summary(results, baseline_only=False)
        pipeline.save_results(
            results=results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
        )
        remove_run_checkpoint()
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
    if config.grader_type == "simple":
        grader = SimpleGrader()
    else:
        raise NotImplementedError(f"Grader type '{config.grader_type}' not yet supported")

    # Create attack strategies (using shared utility)
    if not args.quiet:
        print(f"\n[Creating Attack Strategies]")

    strategies = create_attack_strategies_from_config(config, model_pool)

    if not args.quiet:
        for strategy in strategies:
            print(f"  - {strategy.name}")

    # Create pipeline with attacks. The pipeline normalizes execution order.
    pipeline = RobustnessPipeline(
        testee=testee,
        grader=grader,
        attack_strategies=strategies,
        verbose=not args.quiet
    )

    def progress_callback(current_results):
        combined = merge_unique(
            existing_results,
            current_results,
            key=lambda result: result.test_case_id,
        )
        pipeline.save_results(
            results=combined,
            summary=pipeline._compute_summary(combined, baseline_only=False),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
        )

    # Run attack evaluation
    if not args.quiet:
        print(f"\n[Evaluating]")
        print(f"  Using baseline results, applying attacks to {len(baseline_results)} correctly-answered cases...")

    try:
        results, summary = pipeline.run_attack(
            baseline_results,
            progress_callback=progress_callback,
            save_every=10
        )
    except Exception as e:
        print(f"\n[ERROR] Attack evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Merge with existing results if resuming
    if existing_results:
        if not args.quiet:
            print(f"\n[Merging Results]")
            print(f"  Existing results: {len(existing_results)}")
            print(f"  New results: {len(results)}")

        # Combine existing and new results by stable test-case identity.
        all_results = merge_unique(
            existing_results,
            results,
            key=lambda result: result.test_case_id,
        )

        # Recompute summary for all results
        summary = pipeline._compute_summary(all_results, baseline_only=False)

        if not args.quiet:
            print(f"  Total results: {len(all_results)}")

        results = all_results

    # Save final results using standardized interface
    pipeline.save_results(
        results=results,
        summary=summary,
        output_path=str(output_path),
        metadata=make_metadata(False),
    )
    remove_run_checkpoint()

    # Print summary statistics
    if not args.quiet:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print(f"\n{'BASELINE (Round 1)':<40}")
        print(f"  Baseline correct: {summary.first_round_correct}")
        print(f"  Baseline accuracy: {summary.first_round_accuracy:.1%}")

        print(f"\n{'ATTACK (Round 2)':<40}")
        print(f"  Attack correct: {summary.second_round_correct}")
        print(f"  Attack accuracy: {summary.second_round_accuracy:.1%}")
        print(f"  Failed manipulations: {summary.failed_manipulations}")

        print(f"\n{'ROBUSTNESS':<40}")
        if summary.valid_target_tested:
            print(f"  Robustness score: {summary.compute_robustness_score():.1%}")
            print(
                f"  Attack success rate: "
                f"{summary.conditional_attack_success_rate:.1%}"
            )
        else:
            print("  Robustness score: n/a (no valid target-tested attacks)")
            print("  Attack success rate: n/a (no valid target-tested attacks)")
        print(f"  Attack coverage: {summary.attack_coverage:.1%}")
        print(f"  End-to-end attack success: {summary.end_to_end_attack_success_rate:.1%}")
        print(f"  Strategies: {', '.join(summary.attacks_applied)}")

        print(f"\n{'FILES':<40}")
        print(f"  Baseline results: {args.baseline_results}")
        print(f"  Attack results: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
