#!/usr/bin/env python3
"""
Robustness Baseline Testing (Round 1)
======================================

Evaluate a target model on original, unmodified questions. Cases answered
correctly become the target-specific cohort for live attacks or attack replay.

Usage:
    python -m scripts.robustness.run_baseline \
        --testee-model gpt-4o \
        --max-samples 100

    python -m scripts.robustness.run_baseline \
        --config configs/examples/robustness/baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from med_red_team.grader import SimpleGrader
from med_red_team.model_pool import ModelPool
from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessResult,
    build_robustness_metadata,
    robustness_file_identity,
    robustness_resume_signature,
)
from med_red_team.robustness.medqa_loader import load_medqa
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint
from med_red_team.shared.config_loading import resolve_sample_limit
from med_red_team.testee import Testee
from scripts.utils import (
    generate_and_validate_output_path,
    load_config_from_py,
    load_results_from_json,
    validate_resume,
)

from configs.examples.robustness.baseline import CONFIG as BASELINE_CONFIG


# ==============================================================================
# Argument parsing and configuration
# ==============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original-question Robustness baseline evaluation (Round 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.robustness.run_baseline

  python -m scripts.robustness.run_baseline \
      --testee-model gpt-4o \
      --max-samples 100

  python -m scripts.robustness.run_baseline \
      --config configs/examples/robustness/baseline.py
        """,
    )
    parser.add_argument(
        "--config",
        help="Path to config .py file (default: public Robustness baseline preset)",
    )
    parser.add_argument(
        "--testee-model",
        help="Model to test (for example, gpt-4o or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--dataset",
        help="Path to the original dataset JSONL file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of original questions to evaluate",
    )
    parser.add_argument(
        "--log-dir",
        help="Directory for saving results (default: from config)",
    )
    parser.add_argument(
        "--resume",
        help="Existing baseline result artifact or checkpoint to resume",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    return parser.parse_args(argv)


def create_config_from_args(args: argparse.Namespace) -> RobustnessConfig:
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, RobustnessConfig)
    else:
        if not args.quiet:
            print("[Config] Using baseline config")
        config = BASELINE_CONFIG

    overrides = {}
    if args.testee_model:
        overrides["testee_model"] = args.testee_model
    if args.dataset:
        overrides["dataset_path"] = args.dataset
    if args.log_dir:
        overrides["log_dir"] = args.log_dir

    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = RobustnessConfig.from_dict(config_dict)
    return config


# ==============================================================================
# Main workflow
# ==============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = create_config_from_args(args)
    try:
        sample_limit = resolve_sample_limit(args.max_samples, config.max_samples)
    except ValueError as exc:
        print(f"[ERROR] Invalid sample limit: {exc}")
        return 1
    if sample_limit != config.max_samples:
        config_dict = config.to_dict()
        config_dict["max_samples"] = sample_limit
        config = RobustnessConfig.from_dict(config_dict)

    if not args.quiet:
        print("=" * 80)
        print("ROBUSTNESS BASELINE TESTING (Round 1)")
        print("=" * 80)
        print("\n[Configuration]")
        print("  Mode: Original (unmodified questions)")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Dataset: {config.dataset_path}")
        print(
            "  Max samples: "
            f"{config.max_samples if config.max_samples is not None else 'All'}"
        )
        print("\n[Loading Data]")

    try:
        test_cases, dataset_info = load_medqa(
            dataset_path=config.dataset_path,
            convert_to_test_cases=True,
            limit=config.max_samples,
        )
        dataset_identity = robustness_file_identity(config.dataset_path)
    except Exception as exc:
        print(f"[ERROR] Cannot load baseline dataset: {exc}")
        return 1

    if not args.quiet:
        print(f"  Loaded {len(test_cases)} test cases")

    source_info = {
        "dataset_path": dataset_identity["path"],
        "dataset_sha256": dataset_identity["sha256"],
    }

    def make_metadata(is_partial: bool):
        return build_robustness_metadata(
            phase="baseline",
            config=config,
            target_model=config.testee_model,
            dataset_info=dataset_info or {},
            source=source_info,
            is_partial=is_partial,
            sample_limit=config.max_samples,
            evaluation_mode="original",
        )

    existing_results: list[RobustnessResult] = []
    processed_ids: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {resume_path}")
            return 1
        try:
            if not args.quiet:
                print("\n[Loading Resume File]")
            existing_results, resume_metadata = load_results_from_json(
                resume_path,
                data_key="results",
                quiet=args.quiet,
                reconstruct_fn=RobustnessResult.from_dict,
            )
            expected_signature = robustness_resume_signature(make_metadata(False))
            resume_signature = robustness_resume_signature(resume_metadata)
            validations = {
                key: (value, key.replace("_", " ").title())
                for key, value in expected_signature.items()
            }
            valid, message = validate_resume(
                resume_signature,
                validations,
                quiet=args.quiet,
            )
            if not valid:
                raise ValueError(message)

            resume_ids = [result.test_case_id for result in existing_results]
            if len(set(resume_ids)) != len(resume_ids):
                raise ValueError("Resume artifact contains duplicate test_case_id values")
            scoped_ids = {test_case.id for test_case in test_cases}
            unexpected_ids = sorted(set(resume_ids).difference(scoped_ids))
            if unexpected_ids:
                raise ValueError(
                    "Resume artifact contains results outside the current sampled "
                    f"dataset: {unexpected_ids[:5]}"
                )
            processed_ids = set(resume_ids)
        except Exception as exc:
            print(f"[ERROR] Resume validation failed: {exc}")
            return 1

        if not args.quiet:
            print(f"  Found {len(processed_ids)} already-evaluated test cases")

    if processed_ids:
        original_count = len(test_cases)
        test_cases = [
            test_case for test_case in test_cases if test_case.id not in processed_ids
        ]
        if not args.quiet:
            print(
                f"  Filtered out {original_count - len(test_cases)} "
                "already-evaluated cases"
            )
            print(f"  Remaining to evaluate: {len(test_cases)} cases")

    output_path = generate_and_validate_output_path(
        log_dir=config.log_dir,
        testee_model=config.testee_model,
        mode="robustness_baseline",
        max_samples=config.max_samples,
        quiet=args.quiet,
    )
    if output_path is None:
        print("[ERROR] Cannot write to output directory. Exiting.")
        return 1

    if not test_cases:
        if not args.quiet:
            print("\n[INFO] No remaining test cases. Saving final baseline artifact.")
        pipeline = RobustnessPipeline(
            testee=None,
            grader=None,
            attack_strategies=[],
            verbose=not args.quiet,
        )
        summary = pipeline._compute_summary(existing_results, baseline_only=True)
        pipeline.save_results(
            results=existing_results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
        )
        remove_checkpoint(output_path)
        return 0

    if config.grader_type != "simple":
        print(f"[ERROR] Grader type {config.grader_type!r} is not supported")
        return 1

    model_pool = ModelPool()
    testee = Testee(
        model_id=config.testee_model,
        model_pool=model_pool,
        system_prompt=config.testee_system_prompt,
        config=config.testee_config,
    )
    pipeline = RobustnessPipeline(
        testee=testee,
        grader=SimpleGrader(),
        attack_strategies=[],
        verbose=not args.quiet,
    )

    def progress_callback(current_results):
        combined = merge_unique(
            existing_results,
            current_results,
            key=lambda result: result.test_case_id,
        )
        pipeline.save_results(
            results=combined,
            summary=pipeline._compute_summary(combined, baseline_only=True),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
        )

    if not args.quiet:
        print("\n[Evaluating]")
        print(f"  Testing {len(test_cases)} original questions (no attacks)...")

    try:
        new_results, _ = pipeline.run_baseline(
            test_cases,
            max_samples=None,
            progress_callback=progress_callback,
            save_every=10,
        )
    except Exception as exc:
        print(f"[ERROR] Baseline evaluation failed: {exc}")
        return 1

    results = merge_unique(
        existing_results,
        new_results,
        key=lambda result: result.test_case_id,
    )
    summary = pipeline._compute_summary(results, baseline_only=True)
    pipeline.save_results(
        results=results,
        summary=summary,
        output_path=str(output_path),
        metadata=make_metadata(False),
    )
    remove_checkpoint(output_path)

    if not args.quiet:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nTotal cases tested: {summary.total_samples}")
        print(
            f"First round correct: {summary.first_round_correct} "
            f"({summary.first_round_accuracy:.1%})"
        )
        print(f"Skipped: {summary.skipped_samples}")
        print(f"\nResults saved to: {output_path}")
        print(
            "\nCases answered correctly can be used with run_attack.py or "
            "run_attack_replay.py"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
