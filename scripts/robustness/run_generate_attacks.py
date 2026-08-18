#!/usr/bin/env python3
"""
Robustness Attack Dataset Generation
=====================================

Generate attacked test datasets WITHOUT evaluating any testee model.
Creates reusable attacked datasets that can be tested on multiple models.

This script:
1. Loads original test cases from MedQA
2. Applies attack strategies to generate modified questions
3. Saves both original and attacked versions to JSON
4. Does NOT require a testee model

Usage:
    # Generate with specific strategies
    python -m scripts.robustness.run_generate_attacks \
        --strategies bias_manipulation add_distraction_sentence \
        --max-samples 200

    # Use config file
    python -m scripts.robustness.run_generate_attacks \
        --config configs/examples/robustness/attack.py \
        --max-samples 200

Example:
    >>> python -m scripts.robustness.run_generate_attacks \
    ...     --strategies bias_manipulation \
    ...     --max-samples 100 \
    ...     --attacker-model gpt-4o

    Output: logs/attacked_datasets/bias_manipulation_100_20250128_143022.json
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List

from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.robustness.data import (
    build_robustness_metadata,
    robustness_file_identity,
    robustness_resume_signature,
    strategy_order_from_metadata,
    validate_attacked_dataset_rows,
    validate_fixed_strategy_chain,
    validate_robustness_attack_generation_metadata,
)
from med_red_team.robustness.medqa_loader import load_medqa
from med_red_team.model_pool import ModelPool
from med_red_team.shared.checkpoints import merge_unique
from med_red_team.shared.io import atomic_write_json
from scripts.utils import (
    load_config_from_py,
    create_attack_strategies_from_config,
    override_attacker_strategies,
    load_results_from_json,
    validate_output_path,
    validate_resume,
)

# Import default config
from configs.examples.robustness.attack import CONFIG as DEFAULT_CONFIG


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate attacked test dataset (testee-independent)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with specific strategies
  python -m scripts.robustness.run_generate_attacks \\
      --strategies bias_manipulation add_distraction_sentence \\
      --max-samples 200

  # Use custom config
  python -m scripts.robustness.run_generate_attacks \\
      --config configs/examples/robustness/attack.py \\
      --max-samples 100

  # Override attacker model
  python -m scripts.robustness.run_generate_attacks \\
      --strategies bias_manipulation \\
      --attacker-model claude-3-5-sonnet \\
      --max-samples 50
        """
    )

    # Configuration source
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (e.g., configs/examples/robustness/attack.py)"
    )

    # Attack strategies
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Attack strategies to use (default: from config or all strategies in default config)"
    )

    # Model settings
    parser.add_argument(
        "--attacker-model",
        type=str,
        help="Model for attack generation (overrides config for all strategies)"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to MedQA dataset (default: from config)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (default: all)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/attacked_datasets",
        help="Directory for saving attacked dataset (default: logs/attacked_datasets)"
    )

    # Resume settings
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to existing attacked dataset file to resume from (.inprogress or completed)"
    )

    # Display settings
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages"
    )

    return parser.parse_args()


# ==============================================================================
# Configuration Setup
# ==============================================================================

def create_config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    """
    Create config from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        RobustnessConfig instance for attack generation
    """
    # Load from config file if specified
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, RobustnessConfig)
    else:
        # Start with default config
        config = DEFAULT_CONFIG

    # Override with command-line arguments
    overrides = {}

    # Attack strategies (use shared utility)
    new_strategies = override_attacker_strategies(
        config=config,
        args_strategies=args.strategies,
        args_attacker_model=args.attacker_model,
        default_config=DEFAULT_CONFIG,
        quiet=args.quiet
    )
    if new_strategies:
        overrides["attacker_strategies"] = new_strategies

    if args.dataset:
        overrides["dataset_path"] = args.dataset

    if args.max_samples is not None:
        overrides["max_samples"] = args.max_samples

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
    """Generate attacked test dataset."""
    # Parse arguments
    args = parse_args()

    # Create config
    config = create_config_from_args(args)

    if config.attacker_strategies:
        try:
            strategy_names = validate_fixed_strategy_chain(config)
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 1
    else:
        strategy_names = []
    try:
        dataset_identity = robustness_file_identity(config.dataset_path)
    except OSError as exc:
        print(f"[ERROR] Cannot identify source dataset: {exc}")
        return 1

    def make_generation_metadata(*, is_partial: bool, total_samples: int) -> dict:
        return build_robustness_metadata(
            phase="attack_generation",
            config=config,
            target_model=config.testee_model,
            is_partial=is_partial,
            dataset_info={
                "path": dataset_identity["path"],
                "sample_limit": config.max_samples,
            },
            source={
                "dataset_path": dataset_identity["path"],
                "dataset_sha256": dataset_identity["sha256"],
                "workflow": "run_generate_attacks",
            },
            attack_strategies=strategy_names,
            extra={
                "total_samples": total_samples,
                "generation_timestamp": datetime.now().isoformat(),
            },
        )

    # Handle resume
    existing_attacked_dataset = []
    processed_ids = set()
    resume_path = None

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {args.resume}")
            return 1

        if not args.quiet:
            print(f"\n[Loading Resume File]")

        existing_attacked_dataset, resume_metadata = load_results_from_json(
            resume_path,
            data_key="attacked_test_cases",
            quiet=args.quiet
        )
        validate_robustness_attack_generation_metadata(
            resume_metadata,
            expected_strategy_order=strategy_names,
            expected_dataset_identity=dataset_identity,
            require_complete=False,
        )
        validate_attacked_dataset_rows(
            existing_attacked_dataset,
            expected_strategy_order=strategy_names,
        )

        expected_signature = robustness_resume_signature(
            make_generation_metadata(
                is_partial=True,
                total_samples=len(existing_attacked_dataset),
            )
        )
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

        processed_ids = {
            item["original"]["id"] for item in existing_attacked_dataset
        }

        if not args.quiet:
            print(f"  Found {len(processed_ids)} already-generated attacked cases")

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 80)
        print("ROBUSTNESS ATTACK DATASET GENERATION")
        print("=" * 80)
        print(f"\n[Configuration]")
        print(f"  Attack strategies: {', '.join(config.attacker_strategies.keys())}")
        print(f"  Dataset: {config.dataset_path}")
        print(
            "  Max samples: "
            f"{config.max_samples if config.max_samples is not None else 'all'}"
        )

    # Load dataset
    if not args.quiet:
        print(f"\n[Loading Dataset]")
        print(f"  Reading from: {config.dataset_path}")

    load_limit = None if processed_ids else config.max_samples
    test_cases, dataset_info = load_medqa(
        config.dataset_path,
        limit=load_limit,
    )

    if not args.quiet:
        print(f"  Loaded {len(test_cases)} test cases")

    # Filter out already-processed test cases if resuming.
    if processed_ids:
        source_ids = {test_case.id for test_case in test_cases}
        unexpected_ids = sorted(processed_ids.difference(source_ids))
        if unexpected_ids:
            print(
                "[ERROR] Resume contains attacked rows outside the current dataset: "
                f"{unexpected_ids[:5]}"
            )
            return 1
        original_count = len(test_cases)
        test_cases = [tc for tc in test_cases if tc.id not in processed_ids]
        if not args.quiet:
            print(f"  Filtered out {original_count - len(test_cases)} already-generated cases")
            print(f"  Remaining to generate: {len(test_cases)} cases")

        if len(test_cases) == 0 and not args.quiet:
            print("\n[INFO] All test cases already generated. Nothing to do.")
            print(f"  Resume file already contains {len(existing_attacked_dataset)} attacked cases")

    # Calculate remaining samples when resuming
    remaining_samples = None
    if config.max_samples is not None:
        if processed_ids:
            # Already have some cases, calculate how many more to generate
            remaining_samples = config.max_samples - len(existing_attacked_dataset)
            if remaining_samples <= 0 and not args.quiet:
                print(f"\n[INFO] Already have {len(existing_attacked_dataset)} cases (>= max_samples {config.max_samples})")
                print(f"  Nothing more to generate.")
            if remaining_samples > 0 and not args.quiet:
                print(f"  Will generate {remaining_samples} more cases to reach total of {config.max_samples}")
        else:
            # Fresh run, use max_samples directly
            remaining_samples = config.max_samples

    # Calculate expected total for filename before provider construction.
    new_expected = (
        remaining_samples if remaining_samples is not None else len(test_cases)
    )
    total_expected = len(existing_attacked_dataset) + max(new_expected, 0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies_str = "_".join(strategy_names) or "no_strategy"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{strategies_str}_{total_expected}_{timestamp}.json"
    final_output_path = output_dir / filename
    partial_output_path = output_dir / (filename + ".inprogress")

    if not validate_output_path(final_output_path, quiet=args.quiet):
        print("[ERROR] Cannot write to output directory. Exiting.")
        return 1

    # Finalize no-work/all-complete runs without constructing providers.
    if not test_cases or (remaining_samples is not None and remaining_samples <= 0):
        final_dataset = merge_unique(
            existing_attacked_dataset,
            [],
            key=lambda item: item["original"]["id"],
        )
        validate_attacked_dataset_rows(
            final_dataset,
            expected_strategy_order=strategy_names,
        )
        atomic_write_json(
            final_output_path,
            {
                "metadata": make_generation_metadata(
                    is_partial=False,
                    total_samples=len(final_dataset),
                ),
                "attacked_test_cases": final_dataset,
            },
            indent=2,
            ensure_ascii=False,
        )
        if partial_output_path.exists():
            partial_output_path.unlink()
        return 0

    if not strategy_names:
        print("[ERROR] No Robustness attack strategies are configured")
        return 1

    # Initialize model pool only after all no-work and configuration checks.
    model_pool = ModelPool()

    if not args.quiet:
        print(f"\n[Creating Attack Strategies]")

    strategies = create_attack_strategies_from_config(config, model_pool)

    if not args.quiet:
        for strategy in strategies:
            model_info = f" (model: {strategy.model_id})" if hasattr(strategy, 'model_id') and strategy.model_id else ""
            print(f"  - {strategy.name}{model_info}")

    pipeline = RobustnessPipeline(
        testee=None,
        grader=None,
        attack_strategies=strategies,
        verbose=not args.quiet
    )

    # Create progress callback for checkpoint saving (similar to healthbench pattern)
    def progress_callback(current_attacked_dataset):
        """Save checkpoint with merged results to .inprogress file."""
        try:
            merged_dataset = merge_unique(
                existing_attacked_dataset,
                current_attacked_dataset,
                key=lambda item: item["original"]["id"],
            )
            validate_attacked_dataset_rows(
                merged_dataset,
                expected_strategy_order=strategy_names,
            )

            # Prepare stable nested metadata.
            metadata = make_generation_metadata(
                is_partial=True,
                total_samples=len(merged_dataset),
            )

            # Save checkpoint to .inprogress file
            payload = {
                "metadata": metadata,
                "attacked_test_cases": merged_dataset
            }
            atomic_write_json(partial_output_path, payload, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[WARNING] Failed to write checkpoint: {e}")

    # Generate attacked dataset
    if not args.quiet:
        print(f"\n[Generating Attacks]")

    attacked_dataset = pipeline.generate_attacked_dataset(
        test_cases=test_cases,
        max_samples=remaining_samples,
        progress_callback=progress_callback,
        save_every=10
    )

    # Merge with existing attacked dataset if resuming
    if existing_attacked_dataset:
        if not args.quiet:
            print(f"\n[Merging Results]")
            print(f"  Existing attacked cases: {len(existing_attacked_dataset)}")
            print(f"  New attacked cases: {len(attacked_dataset)}")

        # Combine existing and new attacked datasets by stable source identity.
        attacked_dataset = merge_unique(
            existing_attacked_dataset,
            attacked_dataset,
            key=lambda item: item["original"]["id"],
        )
        validate_attacked_dataset_rows(
            attacked_dataset,
            expected_strategy_order=strategy_names,
        )

        if not args.quiet:
            print(f"  Total attacked cases: {len(attacked_dataset)}")

    validate_attacked_dataset_rows(
        attacked_dataset,
        expected_strategy_order=strategy_names,
    )

    # Prepare stable nested metadata.
    metadata = make_generation_metadata(
        is_partial=False,
        total_samples=len(attacked_dataset),
    )

    # Save to final JSON file
    payload = {
        "metadata": metadata,
        "attacked_test_cases": attacked_dataset
    }

    atomic_write_json(final_output_path, payload, indent=2, ensure_ascii=False)

    # Clean up checkpoint file if it exists
    if partial_output_path.exists():
        partial_output_path.unlink()

    # Print summary
    if not args.quiet:
        successful = sum(1 for item in attacked_dataset if not item["attack_metadata"]["manipulation_failed"])
        failed = len(attacked_dataset) - successful

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Show breakdown if resuming
        if existing_attacked_dataset:
            # Calculate stats for existing dataset
            existing_successful = sum(1 for item in existing_attacked_dataset if not item["attack_metadata"]["manipulation_failed"])
            existing_failed = len(existing_attacked_dataset) - existing_successful

            # Calculate stats for newly generated dataset (before merge)
            new_dataset = attacked_dataset[len(existing_attacked_dataset):]
            new_successful = sum(1 for item in new_dataset if not item["attack_metadata"]["manipulation_failed"])
            new_failed = len(new_dataset) - new_successful

            print(f"\nExisting dataset (from resume):")
            print(f"  Total cases: {len(existing_attacked_dataset)}")
            print(f"  Successfully attacked: {existing_successful}")
            print(f"  Failed manipulations: {existing_failed}")

            print(f"\nNewly generated:")
            print(f"  Total cases: {len(new_dataset)}")
            print(f"  Successfully attacked: {new_successful}")
            print(f"  Failed manipulations: {new_failed}")

        print(f"\nFinal dataset (total):")
        print(f"  Total cases: {len(attacked_dataset)}")
        print(f"  Successfully attacked: {successful}")
        print(f"  Failed manipulations: {failed}")
        print(f"  Strategies: {', '.join(list(config.attacker_strategies.keys()))}")
        print(f"\nOutput:")
        print(f"  {final_output_path}")
        print(f"\nNext steps:")
        print("  1. Run the original-question baseline for a target model:")
        print("    python -m scripts.robustness.run_baseline \\")
        print("      --testee-model gpt-4o")
        print("  2. Replay this generated attack artifact on that baseline:")
        print("    python -m scripts.robustness.run_attack_replay \\")
        print("      --baseline-results <baseline-results.json> \\")
        print(f"      --attacked-dataset {final_output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
