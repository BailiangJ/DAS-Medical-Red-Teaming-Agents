#!/usr/bin/env python3
"""Run attack Bias evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from configs.examples.bias.attack import CONFIG as DEFAULT_CONFIG
from med_red_team.bias import (
    BiasConfig,
    BiasGrader,
    BiasPipeline,
    BiasReferenceResult,
    BiasResult,
    load_bias_test_cases,
)
from med_red_team.bias.data import (
    bias_file_identity,
    build_bias_metadata,
    create_attack_summary,
    is_completed_bias_attack_result,
    is_terminal_bias_attack_result,
    reclassify_bias_attack_result,
    reclassify_bias_reference_result,
    validate_bias_attack_reference_binding,
    validate_bias_reference_source_binding,
    validate_bias_resume_metadata,
)
from med_red_team.model_pool import ModelPool
from med_red_team.shared.checkpoints import (
    checkpoint_path,
    merge_unique,
    remove_checkpoint,
    require_complete_artifact,
    retire_foreign_checkpoint,
)
from med_red_team.testee import Testee
from scripts.utils import (
    create_attack_strategies_from_config,
    generate_and_validate_output_path,
    load_config_from_py,
    load_results_from_json,
    override_attacker_strategies,
)


ATTACK_KEY = lambda result: (result.case_id, result.question_idx, result.attack_strategy)



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run attack bias evaluation (Round 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-results",
        required=True,
        type=str,
        help="Path to JSON file with baseline results from run_baseline.py",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config .py file (selected first, then overridden by CLI args)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        help="Attack strategies to use (default: standard four Bias attacks)",
    )
    parser.add_argument("--testee-model", type=str)
    parser.add_argument("--attacker-model", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to existing attack results file to resume from (.inprogress or complete)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)



def load_baseline_data(
    baseline_path: str,
    quiet: bool = False,
    *,
    include_metadata: bool = False,
):
    baseline_path = Path(baseline_path)
    if not baseline_path.exists():
        print(f"[ERROR] Baseline results file not found: {baseline_path}")
        sys.exit(1)

    if not quiet:
        print(f"[Loading] Baseline results from: {baseline_path}")

    baseline_results, baseline_metadata = load_results_from_json(
        baseline_path,
        data_key="results",
        quiet=quiet,
        reconstruct_fn=BiasReferenceResult.from_dict,
    )
    if not isinstance(baseline_metadata, dict):
        print("[ERROR] Baseline results are missing metadata")
        sys.exit(1)

    try:
        require_complete_artifact(
            baseline_metadata,
            label=f"Bias baseline input {baseline_path}",
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    phase = baseline_metadata.get("phase") or baseline_metadata.get("mode")
    axis = baseline_metadata.get("axis") or baseline_metadata.get("evaluation_type")
    if axis != "bias":
        print("[ERROR] Baseline results are not a Bias evaluation")
        sys.exit(1)
    if phase != "baseline":
        print("[ERROR] Bias attack requires results from baseline phase")
        sys.exit(1)

    config_data = baseline_metadata.get("config")
    if not isinstance(config_data, dict):
        print("[ERROR] Baseline metadata is missing a valid config")
        sys.exit(1)
    try:
        config = BiasConfig.from_dict(config_data)
    except (TypeError, ValueError) as exc:
        print(f"[ERROR] Invalid Bias config in baseline metadata: {exc}")
        sys.exit(1)

    models = baseline_metadata.get("models") if isinstance(baseline_metadata.get("models"), dict) else {}
    testee_data = models.get("testee") if isinstance(models.get("testee"), dict) else {}
    config_target_model = config_data.get("testee_model")
    canonical_target_model = testee_data.get("model_id")
    legacy_target_model = baseline_metadata.get("target_model")
    if canonical_target_model and config_target_model and canonical_target_model != config_target_model:
        print("[ERROR] Baseline models.testee.model_id does not match metadata.config.testee_model")
        sys.exit(1)
    if canonical_target_model and legacy_target_model and canonical_target_model != legacy_target_model:
        print("[ERROR] Baseline target_model does not match models.testee.model_id")
        sys.exit(1)
    target_model = canonical_target_model or config_target_model or legacy_target_model
    if not target_model:
        print("[ERROR] Baseline metadata is missing models.testee.model_id")
        sys.exit(1)
    if target_model != config.testee_model:
        print("[ERROR] Baseline target_model does not match metadata.config")
        print(f"  target_model: {target_model}")
        print(f"  config.testee_model: {config.testee_model}")
        sys.exit(1)

    dataset_info = baseline_metadata.get("dataset") if isinstance(baseline_metadata.get("dataset"), dict) else {}
    source_info = baseline_metadata.get("source") if isinstance(baseline_metadata.get("source"), dict) else {}
    source_path = source_info.get("data_file_path")
    if not dataset_info:
        dataset_info = {
            "data_file": baseline_metadata.get("data_file"),
            "sheet_name": baseline_metadata.get("sheet_name"),
            "max_samples": baseline_metadata.get("max_samples"),
        }
    if not dataset_info.get("data_file") or not dataset_info.get("sheet_name"):
        print("[ERROR] Baseline dataset metadata must include data_file and sheet_name")
        sys.exit(1)

    if not baseline_results:
        print("[ERROR] No baseline results found; nothing to attack")
        sys.exit(1)

    incompatible_results = [
        f"{result.case_id}:{result.question_idx}"
        for result in baseline_results
        if not reclassify_bias_reference_result(
            result,
            configured_total_attempts=config.vote_num_baseline,
        )
    ]
    if incompatible_results:
        examples = ", ".join(incompatible_results[:5])
        print("[ERROR] Baseline results have incomplete or incompatible vote histories")
        print(f"  Examples: {examples}")
        sys.exit(1)

    missing_test_cases = [
        f"{result.case_id}:{result.question_idx}"
        for result in baseline_results
        if result.test_case is None
    ]
    if missing_test_cases:
        examples = ", ".join(missing_test_cases[:5])
        print("[ERROR] Baseline results are missing embedded test cases")
        print(f"  Examples: {examples}")
        sys.exit(1)

    try:
        current_test_cases = load_bias_test_cases(
            file_path=source_path or dataset_info["data_file"],
            n_subjects=dataset_info.get("max_samples"),
            sheet=dataset_info["sheet_name"],
        )
    except Exception as exc:
        print(f"[ERROR] Failed to reload the recorded baseline source dataset: {exc}")
        sys.exit(1)
    current_cases_by_id = {
        test_case.case_id: test_case
        for test_case in current_test_cases
    }
    source_mismatches = [
        f"{result.case_id}:{result.question_idx} ({mismatch})"
        for result in baseline_results
        for mismatch in [
            validate_bias_reference_source_binding(
                result,
                current_cases_by_id.get(result.case_id),
            )
        ]
        if mismatch is not None
    ]
    if source_mismatches:
        examples = ", ".join(source_mismatches[:5])
        print("[ERROR] Baseline results do not match the current recorded source cases")
        print(f"  Examples: {examples}")
        sys.exit(1)

    expected_population = dataset_info.get("question_population")
    if (
        isinstance(expected_population, bool)
        or not isinstance(expected_population, int)
        or expected_population < 0
    ):
        print("[ERROR] Baseline metadata must include a valid question_population")
        sys.exit(1)
    if len(baseline_results) != expected_population:
        print(
            "[ERROR] Baseline result key set is incomplete: "
            f"expected {expected_population}, found {len(baseline_results)}"
        )
        sys.exit(1)

    seen_keys = set()
    duplicate_keys = set()
    for result in baseline_results:
        key = (result.case_id, result.question_idx)
        if key in seen_keys:
            duplicate_keys.add(key)
        seen_keys.add(key)
    if duplicate_keys:
        examples = ", ".join(
            f"{case}:{question}" for case, question in sorted(duplicate_keys)
        )
        print(f"[ERROR] Baseline results contain duplicate identities: {examples}")
        sys.exit(1)

    if not quiet:
        print(f"  Loaded {len(baseline_results)} baseline results")

    if include_metadata:
        return baseline_results, config, dataset_info, baseline_metadata
    return baseline_results, config, dataset_info



def create_config_from_args(
    args: argparse.Namespace,
    baseline_config: BiasConfig,
    quiet: bool = False,
) -> BiasConfig:
    if args.config:
        if not quiet:
            print(f"[Config] Loading from: {args.config}")
        config = load_config_from_py(args.config, BiasConfig)
    else:
        config = BiasConfig.from_dict(baseline_config.to_dict())
        if not config.attacker_strategies:
            config_dict = config.to_dict()
            config_dict["attacker_strategies"] = DEFAULT_CONFIG.to_dict()["attacker_strategies"]
            config = BiasConfig.from_dict(config_dict)
            if not quiet:
                print("[Config] Using standard Bias attack strategies")

    overrides = {}
    if args.testee_model is not None:
        overrides["testee_model"] = args.testee_model
    if args.log_dir is not None:
        overrides["log_dir"] = args.log_dir
    if args.attacker_model is not None:
        overrides["attacker_model"] = args.attacker_model

    new_strategies = override_attacker_strategies(
        config=config,
        args_strategies=args.strategies,
        args_attacker_model=args.attacker_model,
        default_config=DEFAULT_CONFIG,
        quiet=quiet,
    )
    if new_strategies is not None:
        overrides["attacker_strategies"] = new_strategies

    if overrides:
        config_dict = config.to_dict()
        config_dict.update(overrides)
        config = BiasConfig.from_dict(config_dict)

    if not config.attacker_strategies:
        raise ValueError(
            "No Bias attack strategies configured; pass --strategies or provide a non-empty attack config"
        )

    if not quiet:
        print(f"[Config] Resolved Bias attack strategies: {', '.join(config.attacker_strategies.keys())}")

    return config



def _attack_dataset_metadata(
    dataset_info: dict,
    baseline_results: list[BiasReferenceResult],
) -> dict:
    return {
        "source_dataset": dataset_info,
        "baseline_question_count": len(baseline_results),
    }



def _attack_source_metadata(
    baseline_results_file: Path,
    baseline_config: BiasConfig,
    dataset_info: dict,
    source_metadata: dict | None = None,
) -> dict:
    identity = bias_file_identity(baseline_results_file)
    source_metadata = source_metadata or {}
    return {
        "source_results_file": identity["path"],
        "source_results_path": identity["path"],
        "source_results_sha256": identity["sha256"],
        "source_schema_version": source_metadata.get("schema_version"),
        "source_axis": (
            source_metadata.get("axis") or source_metadata.get("evaluation_type")
        ),
        "source_phase": source_metadata.get("phase") or source_metadata.get("mode"),
        "source_config": baseline_config.to_dict(),
        "source_dataset": dataset_info,
    }



def _make_dummy_pipeline(config: BiasConfig) -> BiasPipeline:
    return BiasPipeline(
        testee=SimpleNamespace(model_id=config.testee_model),
        grader=BiasGrader(verbose=False),
        attack_strategies=[],
        vote_num_baseline=config.vote_num_baseline,
        vote_num_attack=config.vote_num_attack,
        system_prompt=config.testee_system_prompt,
        verbose=False,
    )



def _unique_attack_keys(results: list[BiasResult]) -> set[tuple[str, int, str]]:
    seen = set()
    duplicates = set()
    for result in results:
        key = ATTACK_KEY(result)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        formatted = ", ".join(
            f"{case}:{question}:{strategy}"
            for case, question, strategy in sorted(duplicates)
        )
        raise ValueError(f"Resume attack contains duplicate case/question/strategy keys: {formatted}")
    return seen



def _reclassify_resume_attack_result(
    result: BiasResult,
    *,
    baseline_by_key: dict[tuple[str, int], BiasReferenceResult],
    configured_total_attempts: int,
) -> bool:
    baseline = baseline_by_key.get((result.case_id, result.question_idx))
    if baseline is None or baseline.test_case is None:
        return False
    if validate_bias_attack_reference_binding(result, baseline) is not None:
        return False
    if (
        result.ref_vote_outcome != baseline.ref_vote_outcome
        or result.ref_majority_vote != baseline.ref_majority_vote
    ):
        return False

    has_attack_votes = bool(result.manipulated_responses)
    vote_outcomes = {
        "bias_detected",
        "no_bias",
        "attack_invalid",
        "attack_no_winner",
    }
    if has_attack_votes or result.evaluation_outcome in vote_outcomes:
        question_text = baseline.test_case.questions[baseline.question_idx]
        return reclassify_bias_attack_result(
            result,
            configured_total_attempts=configured_total_attempts,
            question_text=question_text,
        )

    no_vote_invariants = (
        not result.manipulated_responses
        and not result.manipulated_majority_vote
        and result.manipulated_vote_outcome == "invalid"
        and result.manipulated_valid_vote_count == 0
        and result.manipulated_invalid_vote_count == 0
        and not result.manipulated_vote_counts
        and result.manipulated_configured_total_attempts == 0
        and result.manipulated_required_votes == 0
        and not result.bias_detected
        and result.skipped
    )
    if not no_vote_invariants:
        return False

    baseline_outcome = f"baseline_{baseline.ref_vote_outcome}"
    if baseline.ref_vote_outcome != "valid":
        if result.evaluation_outcome != baseline_outcome:
            return False
        result.attack_status = "complete"
        result.attack_failure_category = baseline_outcome
        return True

    if (
        result.evaluation_outcome == "attack_excluded"
        and result.attack_status == "not_applicable"
        and result.attack_failure_category in {"attack_excluded", "not_applicable"}
    ):
        return True
    if (
        result.evaluation_outcome == "manipulation_failed"
        and result.attack_status == "manipulation_failed"
        and result.attack_failure_category not in {None, "not_applicable"}
    ):
        return True
    return False


def _expected_attack_keys(
    baseline_results: list[BiasReferenceResult],
    strategy_names: list[str],
) -> set[tuple[str, int, str]]:
    return {
        (result.case_id, result.question_idx, strategy_name)
        for result in baseline_results
        for strategy_name in strategy_names
    }



def _strategy_populations(
    expected_keys: set[tuple[str, int, str]],
    strategy_names: list[str],
) -> dict[str, int]:
    return {
        strategy_name: sum(
            strategy == strategy_name
            for _, _, strategy in expected_keys
        )
        for strategy_name in strategy_names
    }


def _valid_question_population(
    baseline_results: list[BiasReferenceResult],
) -> int:
    return len({
        (result.case_id, result.question_idx)
        for result in baseline_results
        if result.ref_vote_outcome == "valid"
    })


def _baseline_noncomparable_attack_results(
    baseline_results: list[BiasReferenceResult],
    strategy_names: list[str],
    completed_keys: set[tuple[str, int, str]],
) -> list[BiasResult]:
    """Materialize invalid/no-winner baseline rows without constructing providers."""
    results = []
    for baseline_idx, baseline_result in enumerate(baseline_results):
        if baseline_result.ref_vote_outcome == "valid":
            continue
        for strategy_idx, strategy_name in enumerate(strategy_names):
            key = (baseline_result.case_id, baseline_result.question_idx, strategy_name)
            if key in completed_keys:
                continue
            result = BiasResult.from_baseline(
                baseline_result,
                baseline_idx * len(strategy_names) + strategy_idx + 1,
                strategy_name,
            )
            result.skipped = True
            result.evaluation_outcome = f"baseline_{baseline_result.ref_vote_outcome}"
            result.attack_status = "complete"
            result.attack_failure_category = result.evaluation_outcome
            result.skip_reason = (
                "Baseline vote had no strict majority winner"
                if baseline_result.ref_vote_outcome == "no_winner"
                else "Baseline vote had insufficient valid answers for strict majority"
            )
            result.comparison_reason = result.skip_reason
            results.append(result)
    return results


def main() -> int:
    args = parse_args()
    try:
        (
            baseline_results,
            baseline_config,
            dataset_info,
            baseline_metadata,
        ) = load_baseline_data(
            args.baseline_results,
            quiet=args.quiet,
            include_metadata=True,
        )
        config = create_config_from_args(args, baseline_config, quiet=args.quiet)
    except SystemExit as exc:
        return int(exc.code or 1)
    except (OSError, ImportError, SyntaxError, KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] Failed to prepare Bias attack inputs: {exc}")
        return 1

    target_mismatches = []
    for field_name in ("testee_model", "testee_config", "testee_system_prompt"):
        if getattr(config, field_name) != getattr(baseline_config, field_name):
            target_mismatches.append(field_name)
    if target_mismatches:
        print("\n[ERROR] Target evaluation settings do not match the baseline")
        print(f"  Mismatched fields: {', '.join(target_mismatches)}")
        print("  Attack evaluation requires identical target model, config, and system prompt.")
        return 1

    strategy_names = list(config.attacker_strategies.keys())
    dataset_metadata = _attack_dataset_metadata(dataset_info, baseline_results)
    source_metadata = _attack_source_metadata(
        Path(args.baseline_results),
        baseline_config,
        dataset_info,
        baseline_metadata,
    )

    def make_metadata(is_partial: bool) -> dict:
        return build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=is_partial,
            dataset=dataset_metadata,
            source=source_metadata,
            attack_strategies=strategy_names,
            baseline_results_file=args.baseline_results,
        )

    if not args.quiet:
        print("\n" + "=" * 80)
        print("BIAS ATTACK TESTING (Round 2)")
        print("=" * 80)
        print("\n[Configuration]")
        print(f"  Testee model: {config.testee_model}")
        print(f"  Attack strategies: {', '.join(strategy_names)}")
        print(f"  Baseline questions: {len(baseline_results)}")

    log_dir = args.log_dir or config.log_dir or Path(args.baseline_results).parent
    try:
        output_path = generate_and_validate_output_path(
            log_dir=str(log_dir),
            testee_model=config.testee_model,
            mode="bias_attack",
            strategies=strategy_names,
            max_samples=None,
            quiet=args.quiet,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[ERROR] Failed to prepare Bias attack output: {exc}")
        return 1
    if output_path is None:
        return 1

    existing_results: list[BiasResult] = []
    processed_keys: set[tuple[str, int, str]] = set()
    expected_keys = _expected_attack_keys(baseline_results, strategy_names)
    strategy_populations = _strategy_populations(expected_keys, strategy_names)
    question_population = _valid_question_population(baseline_results)
    resume_keys: set[tuple[str, int, str]] = set()
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        try:
            existing_results, resume_metadata = load_results_from_json(
                resume_path,
                data_key="results",
                quiet=args.quiet,
                reconstruct_fn=BiasResult.from_dict,
            )
            validate_bias_resume_metadata(
                make_metadata(False),
                resume_metadata,
                compare_attack=True,
            )
            resume_keys = _unique_attack_keys(existing_results)
            baseline_by_key = {
                (result.case_id, result.question_idx): result
                for result in baseline_results
            }
            accepted_existing_results: list[BiasResult] = []
            for result in existing_results:
                if _reclassify_resume_attack_result(
                    result,
                    baseline_by_key=baseline_by_key,
                    configured_total_attempts=config.vote_num_attack,
                ):
                    accepted_existing_results.append(result)
                    processed_keys.add(ATTACK_KEY(result))
            existing_results = accepted_existing_results
        except Exception as exc:
            print(f"[ERROR] Failed to load attack resume artifact: {exc}")
            return 1

        unexpected_keys = resume_keys - expected_keys
        if unexpected_keys:
            formatted = ", ".join(
                f"{case}:{question}:{strategy}"
                for case, question, strategy in sorted(unexpected_keys)
            )
            print(
                "[ERROR] Resume attack contains case/question/strategy keys outside this run: "
                + formatted
            )
            return 1
        if not args.quiet:
            print(f"  Resume contains {len(processed_keys)} completed attack case/question/strategy triples")

    deterministic_results = _baseline_noncomparable_attack_results(
        baseline_results,
        strategy_names,
        processed_keys,
    )
    existing_results = merge_unique(
        existing_results,
        deterministic_results,
        key=ATTACK_KEY,
    )
    processed_keys.update(ATTACK_KEY(result) for result in deterministic_results)

    pending_keys = expected_keys - processed_keys
    if not args.quiet:
        print(f"  Pending attack case/question/strategy triples: {len(pending_keys)}")

    if not pending_keys:
        if not args.quiet:
            print("\n[INFO] No pending bias attack work. Saving final artifact.")
        pipeline = _make_dummy_pipeline(config)
        summary = create_attack_summary(
            existing_results,
            strategy_names,
            attack_population=len(expected_keys),
            strategy_populations=strategy_populations,
            question_population=question_population,
            baseline_population=len(baseline_results),
            baseline_valid_count=question_population,
        )
        pipeline.save_results(
            results=existing_results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, output_path)
        remove_checkpoint(output_path)
        if not args.quiet:
            summary.print_summary()
            print(f"\nBaseline results: {args.baseline_results}")
            print(f"Attack results: {output_path}")
        return 0

    canonical_checkpoint = checkpoint_path(output_path)
    if (
        resume_path is not None
        and resume_path.name.endswith(".inprogress")
        and resume_path != canonical_checkpoint
    ):
        checkpoint_results = merge_unique([], existing_results, key=ATTACK_KEY)
        migration_pipeline = _make_dummy_pipeline(config)
        migration_pipeline.save_results(
            results=checkpoint_results,
            summary=create_attack_summary(
                checkpoint_results,
                strategy_names,
                attack_population=len(expected_keys),
                strategy_populations=strategy_populations,
                question_population=question_population,
                baseline_population=len(baseline_results),
                baseline_valid_count=question_population,
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

    if not args.quiet:
        print("\n[Creating Attack Strategies]")
    strategies = create_attack_strategies_from_config(
        config,
        model_pool,
        verbose=not args.quiet,
    )
    if not strategies:
        print("[ERROR] No Bias attack strategies were created")
        return 1
    if not args.quiet:
        for strategy in strategies:
            print(f"  - {strategy.name}")

    pipeline = BiasPipeline(
        testee=testee,
        grader=grader,
        attack_strategies=strategies,
        vote_num_baseline=config.vote_num_baseline,
        vote_num_attack=config.vote_num_attack,
        system_prompt=config.testee_system_prompt,
        verbose=not args.quiet,
    )

    def progress_callback(current_results):
        checkpoint_results = merge_unique(existing_results, current_results, key=ATTACK_KEY)
        canonical = checkpoint_path(output_path)
        pipeline.save_results(
            results=checkpoint_results,
            summary=create_attack_summary(
                checkpoint_results,
                strategy_names,
                attack_population=len(expected_keys),
                strategy_populations=strategy_populations,
                question_population=question_population,
                baseline_population=len(baseline_results),
                baseline_valid_count=question_population,
            ),
            output_path=str(canonical),
            metadata=make_metadata(True),
            announce=False,
        )
        retire_foreign_checkpoint(resume_path, canonical)

    try:
        results, summary = pipeline.run_attack(
            baseline_results,
            strategies=strategies,
            progress_callback=progress_callback,
            save_every=10,
            completed_attack_keys=processed_keys,
            existing_results=existing_results,
            attack_population=len(expected_keys),
            strategy_populations=strategy_populations,
            question_population=question_population,
            baseline_population=len(baseline_results),
            baseline_valid_count=question_population,
        )
    except Exception as exc:
        print(f"\n[ERROR] Attack evaluation failed: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    baseline_by_key = {
        (baseline.case_id, baseline.question_idx): baseline
        for baseline in baseline_results
    }
    accepted_keys = {
        ATTACK_KEY(result)
        for result in results
        if _reclassify_resume_attack_result(
            result,
            baseline_by_key=baseline_by_key,
            configured_total_attempts=config.vote_num_attack,
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
                "[INCOMPLETE] Bias attack retains "
                f"{len(unresolved)} retryable or missing result(s)"
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
        print(f"\nBaseline results: {args.baseline_results}")
        print(f"Attack results: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
