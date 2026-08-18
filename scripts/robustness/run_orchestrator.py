#!/usr/bin/env python3
"""
Orchestrator-Based Robustness Attack Testing
============================================

Apply iterative, adaptive orchestrator attacks to questions answered correctly in baseline.

This script uses an LLM-based orchestrator that strategically selects and applies
manipulation tools across multiple iterations, learning from failed attempts and
stopping early when the testee is fooled.

Key differences from run_attack.py:
- Iterative: Keeps attacking until testee fails or max iterations reached
- Adaptive: Orchestrator chooses tools based on what failed previously
- Early stopping: Stops immediately when testee is fooled
- Rich logging: Tracks orchestrator decisions and tool effectiveness

Usage:
    # Basic usage with baseline results
    python -m scripts.robustness.run_orchestrator \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json

    # Specify orchestrator and tools models
    python -m scripts.robustness.run_orchestrator \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
        --orchestrator-model gpt-4o \
        --tools-model gpt-4o-mini

    # Set max iterations
    python -m scripts.robustness.run_orchestrator \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
        --max-iterations 10

    # Use different model for testee
    python -m scripts.robustness.run_orchestrator \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
        --testee-model gpt-4o-mini

    # Resume from previous run
    python -m scripts.robustness.run_orchestrator \
        --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
        --resume logs/gpt-4o/robustness_orchestrator_gpt-4o_20250115_150322.json.inprogress

Example:
    >>> python -m scripts.robustness.run_orchestrator \
    ...     --baseline-results logs/gemini-2.5-flash/robustness_baseline_20250115_143022.json \
    ...     --max-iterations 5

    Orchestrator attack testing completed!
    Results saved to: logs/gemini-2.5-flash/robustness_orchestrator_20250115_150322.json
"""

import sys
import argparse
from pathlib import Path

from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessResult,
    build_robustness_metadata,
    robustness_file_identity,
    robustness_resume_signature,
    summarize_robustness_results,
    validate_robustness_baseline_metadata,
    is_baseline_result_eligible_for_attack,
)
from med_red_team.testee import Testee
from med_red_team.grader import SimpleGrader
from med_red_team.model_pool import ModelPool
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint
from med_red_team.shared.config_loading import resolve_sample_limit
from med_red_team.robustness.tool_policy import POLICY_VERSION
from med_red_team.shared.io import atomic_write_result_envelope
from scripts.utils import (
    load_config_from_py,
    load_results_from_json,
    validate_resume,
    generate_and_validate_output_path
)


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run orchestrator-based robustness evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with baseline results
  python -m scripts.robustness.run_orchestrator \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json

  # Specify orchestrator and tools models
  python -m scripts.robustness.run_orchestrator \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --orchestrator-model gpt-4o \
      --tools-model gpt-4o-mini

  # Set max iterations and samples
  python -m scripts.robustness.run_orchestrator \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --max-iterations 10 \
      --max-samples 50

  # Use different testee model
  python -m scripts.robustness.run_orchestrator \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --testee-model gpt-4o-mini

  # Resume from previous run
  python -m scripts.robustness.run_orchestrator \
      --baseline-results logs/gpt-4o/robustness_baseline_20250115_143022.json \
      --resume logs/gpt-4o/robustness_orchestrator_gpt-4o_20250115_150322.json.inprogress
        """
    )

    # Required argument
    parser.add_argument(
        "--baseline-results",
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

    # Model settings
    parser.add_argument(
        "--testee-model",
        type=str,
        help="Model to test (default: same as baseline)"
    )
    parser.add_argument(
        "--orchestrator-model",
        type=str,
        default="gpt-4o",
        help="Model for orchestrator agent (default: gpt-4o)"
    )
    parser.add_argument(
        "--tools-model",
        type=str,
        default="gpt-4o",
        help="Model for manipulation tools (default: gpt-4o)"
    )

    # Orchestrator settings
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum manipulation rounds per test case (default: 5)"
    )
    parser.add_argument(
        "--max-planner-attempts",
        type=int,
        default=2,
        help="Maximum planner proposals per attack round (default: 2)"
    )
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
        help="Path to existing orchestrator results file to resume from (.inprogress or completed)"
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
        Tuple of (baseline_data, config, correct_results, metadata).
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
    correct_results = [
        r for r in baseline_results
        if is_baseline_result_eligible_for_attack(r)
    ]

    if not quiet:
        print(f"  Loaded {len(baseline_results)} baseline results")
        print(f"  First round correct: {len(correct_results)} cases")

    if not correct_results and not quiet:
        print("\n[WARNING] No cases were answered correctly in baseline.")
        print("A complete zero-result orchestrator artifact will be written.")

    baseline_data = {
        "metadata": baseline_metadata,
        "results": [result.to_dict() for result in baseline_results],
    }
    return baseline_data, config, correct_results, baseline_metadata


def load_orchestrator_params_from_config(config_path: str):
    """
    Load orchestrator-specific parameters from config module.

    Args:
        config_path: Path to config .py file

    Returns:
        Tuple of orchestrator model, tools model, iteration limit, and planner-attempt limit
    """
    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("user_config", str(config_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        orchestrator_model = getattr(module, 'orchestrator_model', None)
        tools_model = getattr(module, 'tools_model', None)
        max_iterations = getattr(module, 'max_iterations', None)
        max_planner_attempts = getattr(module, 'max_planner_attempts', None)

        return (
            orchestrator_model,
            tools_model,
            max_iterations,
            max_planner_attempts,
        )
    except Exception:
        return None, None, None, None


def create_config_from_args(
    args: argparse.Namespace,
    baseline_config: RobustnessConfig
) -> RobustnessConfig:
    """
    Create orchestrator config from command-line arguments and baseline config.

    Args:
        args: Parsed command-line arguments
        baseline_config: Configuration from baseline results

    Returns:
        RobustnessConfig instance for orchestrator testing
    """
    # Load from file if specified, otherwise use baseline config
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
    """Run orchestrator-based robustness evaluation."""
    args = parse_args()

    # Load baseline data
    baseline_data, baseline_config, correct_results, baseline_metadata = load_baseline_data(
        args.baseline_results,
        quiet=args.quiet
    )
    baseline_identity = robustness_file_identity(args.baseline_results)
    baseline_source = {
        "baseline_results_file": baseline_identity["path"],
        "baseline_results_sha256": baseline_identity["sha256"],
    }

    # Create config and resolve sampling: CLI override wins over CONFIG.
    config = create_config_from_args(args, baseline_config)
    try:
        sample_limit = resolve_sample_limit(args.max_samples, config.max_samples)
    except ValueError as exc:
        print(f"[ERROR] Invalid sample limit: {exc}")
        return 1
    if sample_limit != config.max_samples:
        config_dict = config.to_dict()
        config_dict["max_samples"] = sample_limit
        config = RobustnessConfig.from_dict(config_dict)

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
        print(f"[ERROR] Baseline metadata validation failed: {exc}")
        return 1
    if config.testee_model != baseline_target_model:
        print(
            "[ERROR] Orchestrator attack must use the same target model as the "
            f"baseline ({baseline_target_model}); got {config.testee_model}."
        )
        return 1

    # Baseline results are provided, so always skip first round.
    skip_first_round = True

    # Load orchestrator parameters (config file values override argparse defaults)
    orchestrator_model = args.orchestrator_model
    tools_model = args.tools_model
    max_iterations = args.max_iterations
    max_planner_attempts = args.max_planner_attempts

    if args.config:
        (
            config_orchestrator_model,
            config_tools_model,
            config_max_iterations,
            config_max_planner_attempts,
        ) = load_orchestrator_params_from_config(args.config)

        if config_orchestrator_model:
            orchestrator_model = config_orchestrator_model
        if config_tools_model:
            tools_model = config_tools_model
        if config_max_iterations:
            max_iterations = config_max_iterations
        if config_max_planner_attempts:
            max_planner_attempts = config_max_planner_attempts

        if not args.quiet:
            print(f"[Config] Loaded orchestrator parameters:")
            print(f"  orchestrator_model: {orchestrator_model}")
            print(f"  tools_model: {tools_model}")
            print(f"  max_iterations: {max_iterations}")
            print(f"  max_planner_attempts: {max_planner_attempts}")

    if max_iterations < 1:
        print(f"[ERROR] max_iterations must be at least 1, got {max_iterations}")
        return 1
    if max_planner_attempts < 1:
        print(
            f"[ERROR] max_planner_attempts must be at least 1, "
            f"got {max_planner_attempts}"
        )
        return 1

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
            phase="orchestrator_attack",
            config=config,
            target_model=config.testee_model,
            dataset_info=dict(baseline_metadata.get("dataset", {})),
            source=baseline_source,
            is_partial=False,
            sample_limit=sample_limit,
            baseline_results_file=baseline_identity["path"],
            orchestrator_model=orchestrator_model,
            tools_model=tools_model,
            max_iterations=max_iterations,
            max_planner_attempts=max_planner_attempts,
            attack_mode="orchestrator_progressive",
            extra={"tool_policy_version": POLICY_VERSION},
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

    # Apply effective sample limit to the ordered attack target list.
    if sample_limit is not None and len(correct_results) > sample_limit:
        correct_results = correct_results[:sample_limit]

    if existing_results:
        resume_ids = [result.test_case_id for result in existing_results]
        if len(set(resume_ids)) != len(resume_ids):
            print("[ERROR] Resume file contains duplicate test_case_id values")
            return 1
        scoped_ids = {result.test_case_id for result in correct_results}
        unexpected_ids = sorted(set(resume_ids).difference(scoped_ids))
        if unexpected_ids:
            print(
                "[ERROR] Resume file contains results outside the current "
                f"sampled baseline: {unexpected_ids[:5]}"
            )
            return 1
        processed_ids = set(resume_ids)
        if not args.quiet:
            print(f"  Found {len(processed_ids)} already-evaluated orchestrator attack cases")

    # Baseline rows are self-contained; reconstruct the exact questions evaluated.
    if not args.quiet:
        print(f"\n[Loading Data]")
    test_cases = [result.to_test_case() for result in correct_results]
    dataset_info = dict(baseline_metadata.get("dataset", {}))

    # Filter out already-processed cases if resuming.
    if processed_ids:
        original_test_case_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.id not in processed_ids]
        if not args.quiet:
            print(f"\n[Filtering Test Cases]")
            print(f"  Original test cases: {original_test_case_count}")
            print(f"  Already attacked: {len(processed_ids)}")
            print(f"  Remaining to attack: {len(test_cases)}")

    adjusted_max_samples = sample_limit
    if sample_limit is not None and existing_results:
        adjusted_max_samples = max(sample_limit - len(existing_results), 0)

    # Print configuration before constructing providers.
    if not args.quiet:
        print("\n" + "=" * 80)
        print("ORCHESTRATOR-BASED ROBUSTNESS ATTACK TESTING")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Orchestrator model: {orchestrator_model}")
        print(f"  Tools model: {tools_model}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Max planner attempts: {max_planner_attempts}")
        print(f"  Max samples: {sample_limit if sample_limit is not None else 'all'}")
        print(f"  Attacking {len(test_cases)} remaining correctly-answered cases")

    # Determine and validate output path before model construction.
    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = str(Path(args.baseline_results).parent)

    if resume_path:
        output_path = (
            Path(str(resume_path).replace('.inprogress', ''))
            if str(resume_path).endswith('.inprogress') else resume_path
        )
    else:
        suffix = f"_iter{max_iterations}_plan{max_planner_attempts}"
        output_path = generate_and_validate_output_path(
            log_dir=log_dir,
            testee_model=config.testee_model,
            mode="robustness_orchestrator",
            strategies=None,
            max_samples=sample_limit,
            suffix=suffix,
            quiet=args.quiet
        )
        if output_path is None:
            print("[ERROR] Cannot write to output directory. Exiting.")
            return 1

    summary_metadata = {
        "orchestrator_model": orchestrator_model,
        "tools_model": tools_model,
        "max_iterations": max_iterations,
        "max_planner_attempts": max_planner_attempts,
        "tool_policy_version": POLICY_VERSION,
    }

    def make_metadata(is_partial: bool):
        return build_robustness_metadata(
            phase="orchestrator_attack",
            config=config,
            target_model=config.testee_model,
            dataset_info=dataset_info or {},
            source=baseline_source,
            is_partial=is_partial,
            sample_limit=sample_limit,
            baseline_results_file=baseline_identity["path"],
            orchestrator_model=orchestrator_model,
            tools_model=tools_model,
            max_iterations=max_iterations,
            max_planner_attempts=max_planner_attempts,
            attack_mode="orchestrator_progressive",
            extra={"tool_policy_version": POLICY_VERSION},
        )

    def save_orchestrator_artifact(results, is_partial: bool):
        total = len(results)
        fooled = sum(1 for result in results if result.status == "fooled")
        avg_iterations = (
            sum(result.metadata.get("iterations", 0) for result in results) / total
            if total else 0.0
        )
        avg_fooled_iterations = (
            sum(
                result.metadata.get("fooled_at_iteration", 0)
                for result in results
                if result.status == "fooled"
            ) / fooled
            if fooled else 0.0
        )
        summary = summarize_robustness_results(
            results,
            evaluation_type="robustness_orchestrator",
            attacks_applied=list(dict.fromkeys(
                attack
                for result in results
                for attack in (result.attacks_applied or [])
            )),
            metadata={
                **summary_metadata,
                "avg_iterations_per_sample": avg_iterations,
                "avg_iterations_to_fool": avg_fooled_iterations,
            },
        )
        atomic_write_result_envelope(
            str(checkpoint_path(output_path)) if is_partial else output_path,
            metadata=make_metadata(is_partial),
            summary=summary,
            data=[result.to_dict() for result in results],
            data_key="results",
        )
        return summary

    # Zero-work/all-complete paths must not construct providers.
    if len(test_cases) == 0 or adjusted_max_samples == 0:
        if not args.quiet:
            print("\n[INFO] No remaining orchestrator cases to attack. Saving final artifact.")
        results = existing_results
        save_orchestrator_artifact(results, is_partial=False)
        remove_checkpoint(output_path)
        return 0

    try:
        from med_red_team.robustness.orchestrator_pipeline import OrchestratorPipeline
    except ModuleNotFoundError as exc:
        if exc.name == "agents":
            print(
                "[ERROR] The orchestrator requires the optional API dependencies. "
                "Install with: python -m pip install -e '.[api]'"
            )
            return 1
        raise

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
    grader = SimpleGrader()

    # Create orchestrator pipeline
    if not args.quiet:
        print(f"\n[Creating Orchestrator Pipeline]")
        print(f"  Orchestrator: {orchestrator_model}")
        print(f"  Tools: {tools_model}")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Max planner attempts: {max_planner_attempts}")

    strategy_configs = getattr(config, 'attacker_strategies', None)
    if strategy_configs and not args.quiet:
        print(f"  Strategy configs: {list(strategy_configs.keys())}")

    pipeline = OrchestratorPipeline(
        testee=testee,
        grader=grader,
        orchestrator_model_id=orchestrator_model,
        tools_model_id=tools_model,
        model_pool=model_pool,
        max_iterations=max_iterations,
        max_planner_attempts=max_planner_attempts,
        baseline_results=baseline_data,
        strategy_configs=strategy_configs,
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
            summary=pipeline._create_summary(combined),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
        )

    # Run orchestrator evaluation
    if not args.quiet:
        print(f"\n[Evaluating]")
        print(f"  Using baseline results (skipping first round)")
        if sample_limit is not None and existing_results:
            print(f"  Target: {sample_limit} total samples ({len(existing_results)} existing + {adjusted_max_samples} new)")
        print(f"  Applying orchestrator attacks to {len(test_cases)} cases...")

    try:
        results, summary = pipeline.run_orchestrator_attack(
            test_cases,
            progress_callback=progress_callback,
            save_every=10,
            max_samples=adjusted_max_samples,
            skip_first_round=skip_first_round
        )
    except Exception as e:
        print(f"\n[ERROR] Orchestrator evaluation failed: {e}")
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
        summary = pipeline._create_summary(all_results)

        if not args.quiet:
            print(f"  Total results: {len(all_results)}")

        results = all_results

    # Save to final output path (output_path is always the .json base path)
    pipeline.save_results(
        results=results,
        summary=summary,
        output_path=output_path,
        metadata=make_metadata(False),
    )

    remove_checkpoint(output_path)

    # Print output path
    if not args.quiet:
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED")
        print(f"{'='*80}")
        print(f"  Output file: {output_path}")
        print(f"  Baseline file: {args.baseline_results}")
        print(f"\nOrchestrator attack testing completed successfully!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
