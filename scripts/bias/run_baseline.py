#!/usr/bin/env python3
"""Run baseline Bias evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from configs.examples.bias.baseline import CONFIG as BASELINE_CONFIG
from med_red_team.bias import (
    BiasGrader,
    BiasPipeline,
    BiasReferenceResult,
    load_bias_test_cases,
)
from med_red_team.bias.config import BiasConfig
from med_red_team.bias.data import (
    bias_file_identity,
    build_bias_metadata,
    create_baseline_summary,
    reclassify_bias_reference_result,
    validate_bias_reference_source_binding,
    validate_bias_resume_metadata,
)
from med_red_team.model_pool import ModelPool
from med_red_team.shared.checkpoints import (
    checkpoint_path,
    merge_unique,
    remove_checkpoint,
    retire_foreign_checkpoint,
)
from med_red_team.testee import Testee
from scripts.utils import (
    generate_and_validate_output_path,
    load_config_from_py,
    load_results_from_json,
)


BASELINE_KEY = lambda result: (result.case_id, result.question_idx)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline bias evaluation (Round 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (default: configs/examples/bias/baseline.py)",
    )
    parser.add_argument("--testee-model", type=str)
    parser.add_argument("--data-file", type=str)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--vote-num-baseline", type=int)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-file", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to existing baseline result file to resume from (.inprogress or complete)",
    )
    return parser.parse_args(argv)



def create_config_from_args(args: argparse.Namespace) -> BiasConfig:
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, BiasConfig)
    else:
        if not args.quiet:
            print("[Config] Using baseline config")
        config = BiasConfig.from_dict(BASELINE_CONFIG.to_dict())

    overrides = {}
    if args.testee_model is not None:
        overrides["testee_model"] = args.testee_model
    if args.data_file is not None:
        overrides["data_file"] = args.data_file
    if args.max_samples is not None:
        overrides["max_samples"] = args.max_samples
    if args.vote_num_baseline is not None:
        overrides["vote_num_baseline"] = args.vote_num_baseline
    if args.log_dir is not None:
        overrides["log_dir"] = args.log_dir
    if args.no_cache:
        overrides["use_baseline_cache"] = False
    if args.cache_file is not None:
        overrides["baseline_cache_file"] = args.cache_file

    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = BiasConfig.from_dict(config_dict)
    return config



def _baseline_dataset_metadata(config: BiasConfig) -> dict:
    return {
        "data_file": config.data_file,
        "sheet_name": config.sheet_name,
        "max_samples": config.max_samples,
    }



def _baseline_source_metadata(config: BiasConfig) -> dict:
    identity = bias_file_identity(config.data_file)
    source = {
        "data_file": config.data_file,
        "sheet_name": config.sheet_name,
    }
    if identity["sha256"]:
        source["data_file_path"] = identity["path"]
        source["data_file_sha256"] = identity["sha256"]
    return source



def _make_dummy_pipeline(config: BiasConfig) -> BiasPipeline:
    return BiasPipeline(
        testee=SimpleNamespace(
            model_id=config.testee_model,
            config=config.testee_config,
            system_prompt=config.testee_system_prompt,
        ),
        grader=BiasGrader(verbose=False),
        attack_strategies=[],
        vote_num_baseline=config.vote_num_baseline,
        vote_num_attack=config.vote_num_attack,
        system_prompt=config.testee_system_prompt,
        verbose=False,
    )



def _expected_baseline_keys(test_cases: list) -> set[tuple[str, int]]:
    return {
        (test_case.case_id, question_idx)
        for test_case in test_cases
        for question_idx in range(4)
        if test_case.has_valid_question(question_idx)
    }



def _unique_baseline_keys(results: list[BiasReferenceResult]) -> set[tuple[str, int]]:
    seen = set()
    duplicates = set()
    for result in results:
        key = BASELINE_KEY(result)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        formatted = ", ".join(f"{case}:{idx}" for case, idx in sorted(duplicates))
        raise ValueError(f"Resume baseline contains duplicate case/question keys: {formatted}")
    return seen



def main() -> int:
    args = parse_args()
    try:
        config = create_config_from_args(args)
        dataset_metadata = _baseline_dataset_metadata(config)
        source_metadata = _baseline_source_metadata(config)
    except (OSError, ImportError, SyntaxError, KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] Failed to prepare Bias baseline configuration: {exc}")
        return 1

    def make_metadata(is_partial: bool) -> dict:
        return build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=is_partial,
            dataset=dataset_metadata,
            source=source_metadata,
        )

    if not args.quiet:
        print("=" * 80)
        print("BIAS BASELINE TESTING (Round 1)")
        print("=" * 80)
        print("\n[Configuration]")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Dataset: {config.data_file}")
        print(f"  Max samples: {config.max_samples if config.max_samples is not None else 'All'}")
        print(f"  Votes per question: {config.vote_num_baseline}")
        print(f"  Cache enabled: {config.use_baseline_cache}")

    try:
        output_path = generate_and_validate_output_path(
            log_dir=config.log_dir,
            testee_model=config.testee_model,
            mode="bias_baseline",
            strategies=None,
            max_samples=config.max_samples,
            quiet=args.quiet,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] Failed to prepare Bias baseline output: {exc}")
        return 1
    if output_path is None:
        return 1

    try:
        if not args.quiet:
            print("\n[Loading Data]")
        test_cases = (
            []
            if config.max_samples == 0
            else load_bias_test_cases(
                file_path=config.data_file,
                n_subjects=config.max_samples,
                sheet=config.sheet_name,
            )
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load bias test cases: {exc}")
        return 1

    expected_keys = _expected_baseline_keys(test_cases)
    dataset_metadata["case_population"] = len(test_cases)
    dataset_metadata["question_population"] = len(expected_keys)

    current_cases_by_id = {
        test_case.case_id: test_case
        for test_case in test_cases
    }
    existing_results: list[BiasReferenceResult] = []
    processed_keys: set[tuple[str, int]] = set()
    resume_keys: set[tuple[str, int]] = set()
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        try:
            existing_results, resume_metadata = load_results_from_json(
                resume_path,
                data_key="results",
                quiet=args.quiet,
                reconstruct_fn=BiasReferenceResult.from_dict,
            )
            validate_bias_resume_metadata(make_metadata(False), resume_metadata)
            resume_keys = _unique_baseline_keys(existing_results)
            accepted_existing_results: list[BiasReferenceResult] = []
            for result in existing_results:
                mismatch = validate_bias_reference_source_binding(
                    result,
                    current_cases_by_id.get(result.case_id),
                )
                if mismatch is not None:
                    raise ValueError(
                        "Resume baseline row "
                        f"{result.case_id}:{result.question_idx} is incompatible with the current source case "
                        f"({mismatch})"
                    )
                if reclassify_bias_reference_result(
                    result,
                    configured_total_attempts=config.vote_num_baseline,
                ):
                    accepted_existing_results.append(result)
                    processed_keys.add(BASELINE_KEY(result))
            existing_results = accepted_existing_results
        except Exception as exc:
            print(f"[ERROR] Failed to load baseline resume artifact: {exc}")
            return 1
        if not args.quiet:
            print(f"  Resume contains {len(processed_keys)} completed baseline question instances")

    unexpected_keys = resume_keys - expected_keys
    if unexpected_keys:
        formatted = ", ".join(f"{case}:{idx}" for case, idx in sorted(unexpected_keys))
        print(f"[ERROR] Resume baseline contains case/question keys outside this run: {formatted}")
        return 1

    if not args.quiet:
        total_questions = len(expected_keys)
        print(f"  Loaded {len(test_cases)} test cases")
        print(f"  Total questions: {total_questions}")
        print(f"  Pending baseline question instances: {len(expected_keys - processed_keys)}")

    if not (expected_keys - processed_keys):
        if not args.quiet:
            print("\n[INFO] No pending bias baseline work. Saving final artifact.")
        pipeline = _make_dummy_pipeline(config)
        results = merge_unique([], existing_results, key=BASELINE_KEY)
        summary = create_baseline_summary(
            results,
            expected_population=len(expected_keys),
        )
        cache_source_metadata = {
            "dataset": dataset_metadata,
            "source": source_metadata,
        }
        if config.use_baseline_cache and config.baseline_cache_file:
            pipeline.rebuild_baseline_cache(
                results,
                config.baseline_cache_file,
                cache_source_metadata,
            )
        pipeline.save_results(
            results=results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, output_path)
        remove_checkpoint(output_path)
        if not args.quiet:
            summary.print_summary()
            print(f"\nResults saved to: {output_path}")
            if config.use_baseline_cache and config.baseline_cache_file:
                print(f"Cache saved to: {config.baseline_cache_file}")
        return 0

    canonical_checkpoint = checkpoint_path(output_path)
    if (
        resume_path is not None
        and resume_path.name.endswith(".inprogress")
        and resume_path != canonical_checkpoint
    ):
        checkpoint_results = merge_unique([], existing_results, key=BASELINE_KEY)
        migration_pipeline = _make_dummy_pipeline(config)
        migration_pipeline.save_results(
            results=checkpoint_results,
            summary=create_baseline_summary(
                checkpoint_results,
                expected_population=len(expected_keys),
            ),
            output_path=str(canonical_checkpoint),
            metadata=make_metadata(True),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, canonical_checkpoint)
        resume_path = canonical_checkpoint

    model_pool = ModelPool()
    testee = Testee(
        model_id=config.testee_model,
        model_pool=model_pool,
        system_prompt=config.testee_system_prompt,
        config=config.testee_config,
    )
    grader = BiasGrader(verbose=not args.quiet)
    pipeline = BiasPipeline(
        testee=testee,
        grader=grader,
        attack_strategies=[],
        vote_num_baseline=config.vote_num_baseline,
        vote_num_attack=config.vote_num_attack,
        system_prompt=config.testee_system_prompt,
        verbose=not args.quiet,
    )

    def progress_callback(current_results):
        checkpoint_results = merge_unique(
            existing_results,
            current_results,
            key=BASELINE_KEY,
        )
        canonical = checkpoint_path(output_path)
        pipeline.save_results(
            results=checkpoint_results,
            summary=create_baseline_summary(
                checkpoint_results,
                expected_population=len(expected_keys),
            ),
            output_path=str(canonical),
            metadata=make_metadata(True),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, canonical)

    try:
        results, summary = pipeline.run_baseline(
            test_cases=test_cases,
            load_from_cache=config.use_baseline_cache,
            cache_file=config.baseline_cache_file,
            save_cache=config.use_baseline_cache,
            cache_source_metadata={
                "dataset": dataset_metadata,
                "source": source_metadata,
            },
            progress_callback=progress_callback,
            save_every=10,
            completed_baseline_keys=processed_keys,
            existing_results=existing_results,
            expected_population=len(expected_keys),
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    accepted_keys = {
        BASELINE_KEY(result)
        for result in results
        if reclassify_bias_reference_result(
            result,
            configured_total_attempts=config.vote_num_baseline,
        )
    }
    if accepted_keys != expected_keys:
        unresolved = expected_keys - accepted_keys
        canonical = checkpoint_path(output_path)
        pipeline.save_results(
            results=results,
            summary=summary,
            output_path=str(canonical),
            metadata=make_metadata(True),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, canonical)
        if not args.quiet:
            print(
                "[INCOMPLETE] Bias baseline retains "
                f"{len(unresolved)} missing or incomplete result(s)"
            )
        return 1

    pipeline.save_results(
        results=results,
        summary=summary,
        output_path=str(output_path),
        metadata=make_metadata(False),
        announce=False,
    )
    retire_foreign_checkpoint(resume_path, output_path)
    remove_checkpoint(output_path)

    if not args.quiet:
        summary.print_summary()
        print(f"\nResults saved to: {output_path}")
        if config.use_baseline_cache and config.baseline_cache_file:
            print(f"Cache saved to: {config.baseline_cache_file}")
        print("\nBaseline results can be used for attack testing with run_attack.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
