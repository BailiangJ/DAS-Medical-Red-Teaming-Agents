#!/usr/bin/env python3
"""
Privacy Baseline Testing (Round 1)
===================================

Run baseline privacy evaluation without attack strategies.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from med_red_team import Testee, ModelPool
from med_red_team.privacy import (
    PHIGenerator,
    PrivacyGrader,
    PrivacyPipeline,
    PrivacyResult,
    load_privacy_test_cases,
)
from med_red_team.privacy.config import PrivacyConfig
from med_red_team.privacy.data import (
    build_privacy_metadata,
    is_completed_baseline_result,
    privacy_file_identity,
    validate_privacy_resume_metadata,
)
from med_red_team.privacy.results_io import create_baseline_summary
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint
from med_red_team.shared.config_loading import resolve_sample_limit
from scripts.utils import (
    generate_and_validate_output_path,
    load_config_from_py,
    load_results_from_json,
)

# Import baseline config
from configs.examples.privacy.baseline import CONFIG as BASELINE_CONFIG


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline privacy evaluation (Round 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python -m scripts.privacy.run_baseline

  # Specify model and data
  python -m scripts.privacy.run_baseline \
      --testee-model gpt-4o \
      --data-file data/RT_Privacy.xlsx \
      --max-samples 100

  # Resume from a completed or .inprogress baseline artifact
  python -m scripts.privacy.run_baseline --resume logs/privacy.json.inprogress
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (default: configs/examples/privacy/baseline.py)"
    )
    parser.add_argument("--data-file", type=str, default=None)
    parser.add_argument("--sheet-name", type=str, default=None)
    parser.add_argument("--prompt-column", type=str, default=None)
    parser.add_argument("--testee-model", type=str, default=None)
    parser.add_argument("--grader-model", type=str, default=None)
    parser.add_argument("--generator-model", type=str, default=None)
    parser.add_argument(
        "--system-prompt-mode",
        type=str,
        choices=["easy", "hard"],
        default=None,
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-attempts", type=int, default=None)
    parser.add_argument(
        "--grader-max-structured-output-retries",
        type=int,
        default=None,
        help="Local grader structured-output retries per testee response",
    )
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing baseline result file to resume from (.inprogress or complete)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


# ==============================================================================
# Configuration Setup
# ==============================================================================

def create_config_from_args(args: argparse.Namespace) -> PrivacyConfig:
    """Create PrivacyConfig from command-line arguments."""
    if args.config:
        print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, PrivacyConfig)
    else:
        print("[Config] Using baseline config")
        config = BASELINE_CONFIG

    overrides = {}
    for arg_name, config_name in (
        ("data_file", "data_file"),
        ("sheet_name", "sheet_name"),
        ("prompt_column", "prompt_column"),
        ("testee_model", "testee_model"),
        ("grader_model", "grader_model"),
        ("generator_model", "generator_model"),
        ("system_prompt_mode", "testee_system_prompt_mode"),
        ("max_samples", "max_samples"),
        ("num_attempts", "num_attempts"),
        ("grader_max_structured_output_retries", "grader_max_structured_output_retries"),
        ("log_dir", "log_dir"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_name] = value

    config_dict = config.to_dict()
    config_dict.update(overrides)
    return PrivacyConfig.from_dict(config_dict)


def _baseline_dataset_metadata(config: PrivacyConfig) -> dict:
    return {
        "data_file": config.data_file,
        "sheet_name": config.sheet_name,
        "prompt_column": config.prompt_column,
        "max_samples": config.max_samples,
    }


def _baseline_source_metadata(config: PrivacyConfig) -> dict:
    identity = privacy_file_identity(config.data_file)
    source = {
        "data_file": config.data_file,
        "sheet_name": config.sheet_name,
        "prompt_column": config.prompt_column,
    }
    if identity.get("sha256"):
        source["data_file_path"] = identity["path"]
        source["data_file_sha256"] = identity["sha256"]
    return source


def _make_dummy_pipeline(config: PrivacyConfig) -> PrivacyPipeline:
    return PrivacyPipeline(
        testee=SimpleNamespace(model_id=config.testee_model),
        grader=SimpleNamespace(model_id=config.grader_model),
        generator=SimpleNamespace(model_id=config.generator_model),
        attack_strategies=[],
        verbose=False,
    )


def _unique_baseline_ids(results: list[PrivacyResult]) -> set[str]:
    seen = set()
    duplicates = set()
    for result in results:
        if result.test_case_id in seen:
            duplicates.add(result.test_case_id)
        seen.add(result.test_case_id)
    if duplicates:
        raise ValueError(
            "Resume baseline contains duplicate test_case_id values: "
            + ", ".join(sorted(duplicates))
        )
    return seen


# ==============================================================================
# Main Execution
# ==============================================================================

def main() -> int:
    """Run baseline privacy evaluation."""
    args = parse_args()
    config = create_config_from_args(args)

    try:
        sample_limit = resolve_sample_limit(None, config.max_samples)
    except ValueError as exc:
        print(f"[ERROR] Invalid sample limit: {exc}")
        return 1
    if sample_limit != config.max_samples:
        config_dict = config.to_dict()
        config_dict["max_samples"] = sample_limit
        config = PrivacyConfig.from_dict(config_dict)

    dataset_metadata = _baseline_dataset_metadata(config)
    source_metadata = _baseline_source_metadata(config)

    def make_metadata(is_partial: bool) -> dict:
        return build_privacy_metadata(
            phase="baseline",
            config=config,
            is_partial=is_partial,
            dataset=dataset_metadata,
            source=source_metadata,
        )

    if not args.quiet:
        print("=" * 80)
        print("PRIVACY BASELINE TESTING (Round 1)")
        print("=" * 80)
        print("\n[Configuration]")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Grader model: {config.grader_model}")
        print(f"  Generator model: {config.generator_model}")
        print(f"  System prompt mode: {config.testee_system_prompt_mode}")
        print(f"  Data file: {config.data_file}")
        print(f"  Max samples: {config.max_samples if config.max_samples is not None else 'All'}")
        print(f"  Attempts per case: {config.num_attempts}")

    if args.output_name:
        output_path = Path(config.log_dir) / args.output_name
        from scripts.utils import validate_output_path
        if not validate_output_path(output_path, quiet=args.quiet):
            return 1
    else:
        output_path = generate_and_validate_output_path(
            log_dir=config.log_dir,
            testee_model=config.testee_model,
            mode="baseline",
            strategies=None,
            max_samples=config.max_samples,
            suffix=f"_{config.testee_system_prompt_mode}",
            quiet=args.quiet,
        )
        if output_path is None:
            return 1

    existing_results: list[PrivacyResult] = []
    processed_ids: set[str] = set()
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        try:
            existing_results, resume_metadata = load_results_from_json(
                resume_path,
                data_key="results",
                quiet=args.quiet,
                reconstruct_fn=PrivacyResult.from_dict,
            )
            validate_privacy_resume_metadata(make_metadata(False), resume_metadata)
            _unique_baseline_ids(existing_results)
            processed_ids = {
                result.test_case_id
                for result in existing_results
                if is_completed_baseline_result(result)
            }
        except Exception as exc:
            print(f"[ERROR] Failed to load baseline resume artifact: {exc}")
            return 1
        if not args.quiet:
            print(f"  Resume contains {len(processed_ids)} completed baseline cases")

    try:
        if config.max_samples == 0:
            test_cases = []
        else:
            if not args.quiet:
                print(f"\n[Loading] Test cases from: {config.data_file}")
            test_cases = load_privacy_test_cases(
                file_path=config.data_file,
                sheet_name=config.sheet_name,
                prompt_column=config.prompt_column,
                max_samples=config.max_samples,
            )
            if not args.quiet:
                print(f"  Loaded {len(test_cases)} test cases")
    except Exception as exc:
        print(f"\n[ERROR] Failed to load test cases: {exc}")
        return 1

    expected_ids = {test_case.case_id for test_case in test_cases}
    unexpected_ids = processed_ids - expected_ids if test_cases else processed_ids if config.max_samples == 0 else set()
    if unexpected_ids:
        print(
            "[ERROR] Resume baseline contains test_case_id values outside this run: "
            + ", ".join(sorted(unexpected_ids))
        )
        return 1

    pending_test_cases = [
        test_case for test_case in test_cases
        if test_case.case_id not in processed_ids
    ]

    if not pending_test_cases:
        if not args.quiet:
            print("\n[INFO] No pending privacy baseline cases. Saving final artifact.")
        pipeline = _make_dummy_pipeline(config)
        results = merge_unique(
            [],
            existing_results,
            key=lambda result: result.test_case_id,
        )
        summary = create_baseline_summary(
            results,
            expected_population=len(test_cases),
        )
        pipeline.save_results(
            results=results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
        )
        remove_checkpoint(output_path)
        return 0

    # Initialize providers only after resume/zero-work/pending resolution.
    model_pool = ModelPool()
    testee = Testee(
        model_id=config.testee_model,
        model_pool=model_pool,
        system_prompt=config.testee_system_prompt,
        config=config.testee_config,
    )
    grader = PrivacyGrader(
        model_id=config.grader_model,
        model_pool=model_pool,
        config=config.grader_config,
        max_structured_output_retries=config.grader_max_structured_output_retries,
    )
    generator = PHIGenerator(
        model_id=config.generator_model,
        model_pool=model_pool,
        config=config.generator_config,
        basic_identifier_keys=config.basic_identifier_keys,
        extra_identifier_keys=config.extra_identifier_keys,
    )
    pipeline = PrivacyPipeline(
        testee=testee,
        grader=grader,
        generator=generator,
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
            summary=create_baseline_summary(
                combined,
                expected_population=len(test_cases),
            ),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
        )

    try:
        results, summary = pipeline.run_baseline(
            test_cases=pending_test_cases,
            num_attempts=config.num_attempts,
            progress_callback=progress_callback,
            save_every=10,
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    results = merge_unique(
        existing_results,
        results,
        key=lambda result: result.test_case_id,
    )
    summary = create_baseline_summary(
        results,
        expected_population=len(test_cases),
    )

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
        print(f"\nTotal cases evaluated: {summary.total_samples}")
        print(f"Completed cases: {summary.baseline_total}")
        if summary.baseline_total > 0:
            print(f"Privacy upheld: {summary.baseline_safe} ({summary.baseline_safe/summary.baseline_total:.1%})")
            print(f"Borderline: {summary.baseline_borderline} ({summary.baseline_borderline/summary.baseline_total:.1%})")
            print(f"Weighted leakage: {summary.baseline_violations:g} ({summary.baseline_violation_rate:.1%})")
        else:
            print("No baseline cases completed successfully.")
        print(f"\nSkipped: {summary.skipped_samples}")
        print(f"Invalid/incomplete grading: {summary.baseline_incomplete}")
        print(f"\nResults saved to: {output_path}")
        print("\nCases that upheld privacy can be used for attack testing with run_attack.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
