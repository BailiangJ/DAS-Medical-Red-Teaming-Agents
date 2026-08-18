#!/usr/bin/env python3
"""Run staged privacy disguise attacks.

Individual attack runs consume a hard-baseline result file. Combined attack
runs consume the self-contained individual-attack result file and operate only
on cases that remained fully safe under all four individual strategies.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

from med_red_team import ModelPool, Testee
from med_red_team.attacker_registry import get_strategy  # noqa: F401
from med_red_team.privacy import (
    PHIGenerator,
    PrivacyGrader,
    PrivacyPipeline,
    PrivacyResult,
)
from med_red_team.privacy.config import PrivacyConfig
from med_red_team.privacy.data import (
    build_privacy_metadata,
    is_completed_attack_result,
    is_terminal_attack_result,
    privacy_file_identity,
    validate_privacy_result_phi_consistency,
    validate_privacy_result_phi_payload,
    validate_privacy_resume_metadata,
    validate_privacy_stage_compatibility,
)
from med_red_team.privacy.results_io import create_attack_summary
from med_red_team.privacy.stage_handoff import (
    COMBINED_PRIVACY_STRATEGY,
    INDIVIDUAL_PRIVACY_STRATEGIES,
    compute_combined_eligibility,
    is_baseline_safe_for_attack,
    select_baseline_results_for_combined,
)
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint, require_complete_artifact
from scripts.utils import (
    create_attack_strategies_from_config,
    generate_and_validate_output_path,
    load_config_from_py,
    load_results_from_json,
)

from configs.examples.privacy.combined_attack import CONFIG as DEFAULT_COMBINED_CONFIG
from configs.examples.privacy.individual_attack import CONFIG as DEFAULT_INDIVIDUAL_CONFIG


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run individual or combined privacy disguise attacks"
    )
    parser.add_argument(
        "--baseline-results",
        help="Hard-baseline JSON input for an individual-attack run",
    )
    parser.add_argument(
        "--individual-attack-results",
        help="Individual-attack JSON input for a combined-attack run",
    )
    parser.add_argument("--config", help="Privacy config .py file")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "implicit_disguise",
            "focus_distraction",
            "privacy_warning",
            "well_intention",
            "combined_disguise",
        ],
        help="Attack strategies to use (default: from config)",
    )
    parser.add_argument("--testee-model")
    parser.add_argument("--grader-model")
    parser.add_argument("--num-attempts", type=int)
    parser.add_argument(
        "--grader-max-structured-output-retries",
        type=int,
        help="Local grader structured-output retries per testee response",
    )
    parser.add_argument("--log-dir")
    parser.add_argument("--output-name")
    parser.add_argument(
        "--resume",
        help="Path to existing attack result file to resume from (.inprogress or complete)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def create_config_from_args(
    args: argparse.Namespace,
    source_config: PrivacyConfig,
    stage: str,
) -> PrivacyConfig:
    if args.config:
        if not args.quiet:
            print(f"[Config] Loading from: {args.config}")
        privacy_config = load_config_from_py(args.config, PrivacyConfig)
    else:
        privacy_config = source_config

    overrides = {}
    if args.testee_model:
        overrides["testee_model"] = args.testee_model
    if args.grader_model:
        overrides["grader_model"] = args.grader_model
    if args.num_attempts is not None:
        overrides["num_attempts"] = args.num_attempts
    if args.grader_max_structured_output_retries is not None:
        overrides["grader_max_structured_output_retries"] = args.grader_max_structured_output_retries
    if args.log_dir:
        overrides["log_dir"] = args.log_dir

    strategy_names = args.strategies
    if strategy_names is None and not args.config:
        strategy_names = (
            list(INDIVIDUAL_PRIVACY_STRATEGIES)
            if stage == "individual"
            else [COMBINED_PRIVACY_STRATEGY]
        )

    if strategy_names:
        default_config = (
            DEFAULT_INDIVIDUAL_CONFIG
            if stage == "individual"
            else DEFAULT_COMBINED_CONFIG
        )
        attacker_strategies = {}
        for strategy_name in strategy_names:
            if strategy_name in privacy_config.attacker_strategies:
                strategy_config = privacy_config.attacker_strategies[strategy_name].copy()
            elif strategy_name in default_config.attacker_strategies:
                strategy_config = default_config.attacker_strategies[strategy_name].copy()
            else:
                raise ValueError(
                    f"No configuration found for privacy strategy '{strategy_name}'"
                )
            attacker_strategies[strategy_name] = strategy_config
        overrides["attacker_strategies"] = attacker_strategies

    config_dict = privacy_config.to_dict()
    config_dict.update(overrides)
    return PrivacyConfig.from_dict(config_dict)


def _load_privacy_results(
    path: Path,
    data_key: str,
    quiet: bool,
) -> Tuple[List[PrivacyResult], dict]:
    results, metadata = load_results_from_json(
        path,
        data_key=data_key,
        quiet=quiet,
        reconstruct_fn=PrivacyResult.from_dict,
    )
    require_complete_artifact(metadata, label=f"Privacy stage input {path}")
    return results, metadata


def _load_attack_resume(path: Path, quiet: bool) -> Tuple[List[PrivacyResult], dict]:
    try:
        return load_results_from_json(
            path,
            data_key="attack_results",
            quiet=quiet,
            reconstruct_fn=PrivacyResult.from_dict,
        )
    except ValueError as first_error:
        try:
            return load_results_from_json(
                path,
                data_key="results",
                quiet=quiet,
                reconstruct_fn=PrivacyResult.from_dict,
            )
        except ValueError:
            raise first_error


def _load_source_config(metadata: dict, source_path: Path) -> PrivacyConfig:
    config_data = metadata.get("config")
    if not config_data:
        raise ValueError(f"Result file is missing metadata.config: {source_path}")
    return PrivacyConfig.from_dict(config_data)


def _baseline_config_for_stage(
    stage: str,
    source_metadata: dict,
    source_config: PrivacyConfig,
) -> PrivacyConfig:
    """Recover the original baseline config for individual/combined stages."""
    if stage != "combined":
        return source_config
    source_info = source_metadata.get("source")
    if isinstance(source_info, dict):
        nested = source_info.get("source_config")
        if isinstance(nested, dict):
            return PrivacyConfig.from_dict(nested)
    return source_config


def _validate_strategy_stage(stage: str, strategy_names: List[str]) -> None:
    has_combined = COMBINED_PRIVACY_STRATEGY in strategy_names
    if has_combined and len(strategy_names) != 1:
        raise ValueError(
            "combined_disguise cannot run alongside individual strategies; "
            "run the combined stage from individual-attack results"
        )
    if stage == "combined" and strategy_names != [COMBINED_PRIVACY_STRATEGY]:
        raise ValueError("A combined run must configure only combined_disguise")
    if stage == "individual" and has_combined:
        raise ValueError("combined_disguise requires --individual-attack-results")
    if not strategy_names:
        raise ValueError("No privacy attack strategies are configured")


def _attack_dataset_metadata(
    stage: str,
    source_metadata: dict,
    privacy_config: PrivacyConfig,
) -> dict:
    source_dataset = source_metadata.get("dataset")
    if not isinstance(source_dataset, dict):
        source_config = source_metadata.get("config") if isinstance(source_metadata.get("config"), dict) else {}
        source_dataset = {
            "data_file": source_config.get("data_file") or source_metadata.get("data_file"),
            "sheet_name": source_config.get("sheet_name") or source_metadata.get("sheet_name"),
            "prompt_column": source_config.get("prompt_column") or source_metadata.get("prompt_column"),
        }
    return {
        "stage": stage,
        "source_dataset": source_dataset,
        "max_samples": privacy_config.max_samples,
    }


def _attack_source_metadata(source_path: Path, source_metadata: dict) -> dict:
    source_models = source_metadata.get("models") if isinstance(source_metadata.get("models"), dict) else {}
    source_dataset = source_metadata.get("dataset") if isinstance(source_metadata.get("dataset"), dict) else {}
    source_config = source_metadata.get("config") if isinstance(source_metadata.get("config"), dict) else {}
    source_identity = privacy_file_identity(source_path)
    return {
        "source_results_file": str(source_path),
        "source_results_path": source_identity["path"],
        "source_results_sha256": source_identity["sha256"],
        "source_schema_version": source_metadata.get("schema_version"),
        "source_axis": source_metadata.get("axis") or source_metadata.get("evaluation_type"),
        "source_phase": source_metadata.get("phase") or source_metadata.get("mode"),
        "source_models": source_models,
        "source_dataset": source_dataset,
        "source_config": source_config,
    }


def _make_dummy_pipeline(config: PrivacyConfig) -> PrivacyPipeline:
    return PrivacyPipeline(
        testee=SimpleNamespace(model_id=config.testee_model),
        grader=SimpleNamespace(model_id=config.grader_model),
        generator=SimpleNamespace(model_id=config.generator_model),
        attack_strategies=[],
        verbose=False,
    )


def _unique_attack_keys(results: list[PrivacyResult]) -> set[tuple[str, str]]:
    seen = set()
    duplicates = set()
    for result in results:
        key = (result.test_case_id, result.attack_strategy)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        formatted = ", ".join(f"{case}:{strategy}" for case, strategy in sorted(duplicates))
        raise ValueError(f"Resume attack contains duplicate case/strategy keys: {formatted}")
    return seen


def _validate_phi_stage_results(
    baseline_results: list[PrivacyResult],
    attack_results: list[PrivacyResult] | None = None,
) -> None:
    """Validate immutable PHI payloads before stage reuse or model setup."""
    baseline_by_id = {}
    for baseline in baseline_results:
        validate_privacy_result_phi_payload(baseline)
        baseline_by_id[baseline.test_case_id] = baseline

    for attack in attack_results or []:
        source = baseline_by_id.get(attack.test_case_id)
        if source is None:
            raise ValueError(
                f"Attack result {attack.test_case_id} has no source baseline result"
            )
        validate_privacy_result_phi_consistency(source, attack)


def _expected_attack_keys(
    baseline_results: list[PrivacyResult],
    strategy_names: list[str],
    *,
    expected_attempts: int | None = None,
) -> set[tuple[str, str]]:
    return {
        (result.test_case_id, strategy_name)
        for result in baseline_results
        if is_baseline_safe_for_attack(result, expected_attempts)
        for strategy_name in strategy_names
    }


def _strategy_populations(
    expected_keys: set[tuple[str, str]],
    strategy_names: list[str],
) -> dict[str, int]:
    """Return fixed per-strategy populations for partial and final summaries."""
    return {
        strategy_name: sum(
            strategy == strategy_name
            for _, strategy in expected_keys
        )
        for strategy_name in strategy_names
    }


def main() -> int:
    args = parse_args()
    if bool(args.baseline_results) == bool(args.individual_attack_results):
        print(
            "[ERROR] Supply exactly one stage input: --baseline-results for "
            "individual attacks or --individual-attack-results for the combined attack."
        )
        return 1

    stage = "combined" if args.individual_attack_results else "individual"
    source_path = Path(args.individual_attack_results or args.baseline_results)

    try:
        if stage == "individual":
            baseline_results, source_metadata = _load_privacy_results(
                source_path, "results", args.quiet
            )
            individual_attack_results = None
            evaluation_baseline_results = baseline_results
            _validate_phi_stage_results(baseline_results)
            eligibility = None
        else:
            baseline_results, source_metadata = _load_privacy_results(
                source_path, "baseline_results", args.quiet
            )
            individual_attack_results, _ = _load_privacy_results(
                source_path, "attack_results", True
            )
            _validate_phi_stage_results(baseline_results, individual_attack_results)
            source_config_data = source_metadata.get("config", {})
            source_info = source_metadata.get("source")
            if isinstance(source_info, dict) and isinstance(
                source_info.get("source_config"), dict
            ):
                source_config_data = source_info["source_config"]
            expected_attempts = source_config_data.get("num_attempts")

            expected_case_ids = [
                result.test_case_id
                for result in baseline_results
                if is_baseline_safe_for_attack(result, expected_attempts)
            ]
            eligibility = compute_combined_eligibility(
                individual_attack_results,
                expected_attempts=expected_attempts,
                expected_case_ids=expected_case_ids,
            )
            evaluation_baseline_results = select_baseline_results_for_combined(
                baseline_results,
                eligibility["intersection_safe_case_ids"],
                expected_attempts=expected_attempts,
            )

        source_config = _load_source_config(source_metadata, source_path)
        privacy_config = create_config_from_args(args, source_config, stage)
        baseline_config = _baseline_config_for_stage(
            stage,
            source_metadata,
            source_config,
        )
        validate_privacy_stage_compatibility(baseline_config, privacy_config)
        strategy_names = list(privacy_config.attacker_strategies.keys())
        _validate_strategy_stage(stage, strategy_names)
    except Exception as exc:
        print(f"[ERROR] Failed to prepare privacy {stage} attack run: {exc}")
        return 1

    baseline_expected_attempts = baseline_config.num_attempts

    if privacy_config.testee_model != source_config.testee_model:
        print(
            "[ERROR] Testee model mismatch: source results use "
            f"{source_config.testee_model}, requested {privacy_config.testee_model}"
        )
        return 1

    dataset_metadata = _attack_dataset_metadata(stage, source_metadata, privacy_config)
    source_run_metadata = _attack_source_metadata(source_path, source_metadata)

    def make_metadata(is_partial: bool) -> dict:
        return build_privacy_metadata(
            phase="attack",
            config=privacy_config,
            is_partial=is_partial,
            dataset=dataset_metadata,
            source=source_run_metadata,
            attack_stage=stage,
            attack_strategies=strategy_names,
        )

    expected_keys = _expected_attack_keys(
        evaluation_baseline_results,
        strategy_names,
        expected_attempts=baseline_expected_attempts,
    )
    strategy_populations = _strategy_populations(expected_keys, strategy_names)
    existing_results: list[PrivacyResult] = []
    processed_keys: set[tuple[str, str]] = set()
    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None:
        try:
            existing_results, resume_metadata = _load_attack_resume(
                resume_path,
                quiet=args.quiet,
            )
            _validate_phi_stage_results(
                evaluation_baseline_results,
                existing_results,
            )
            validate_privacy_resume_metadata(
                make_metadata(False),
                resume_metadata,
                compare_attack=True,
            )
            resume_keys = _unique_attack_keys(existing_results)
            processed_keys = {
                (result.test_case_id, result.attack_strategy)
                for result in existing_results
                if is_completed_attack_result(result)
                or is_terminal_attack_result(result)
            }
        except Exception as exc:
            print(f"[ERROR] Failed to load attack resume artifact: {exc}")
            return 1

        unexpected_keys = resume_keys - expected_keys
        if unexpected_keys:
            formatted = ", ".join(
                f"{case}:{strategy}" for case, strategy in sorted(unexpected_keys)
            )
            print(
                "[ERROR] Resume attack contains case/strategy keys outside this run: "
                + formatted
            )
            return 1
        if not args.quiet:
            print(f"  Resume contains {len(processed_keys)} completed attack case/strategy pairs")

    pending_keys = expected_keys - processed_keys

    if not args.quiet:
        print("=" * 80)
        print(f"PRIVACY {stage.upper()} ATTACK TESTING")
        print("=" * 80)
        print(f"Source results: {source_path}")
        print(f"Testee model: {privacy_config.testee_model}")
        print(f"Strategies: {', '.join(strategy_names)}")
        print(f"Attempts per case: {privacy_config.num_attempts}")
        print(f"Pending attack case/strategy pairs: {len(pending_keys)}")
        if eligibility is not None:
            print(f"Combined-eligible cases: {eligibility['eligible_case_count']}")

    log_dir = args.log_dir or privacy_config.log_dir
    if args.output_name:
        output_path = Path(log_dir) / args.output_name
        from scripts.utils import validate_output_path
        if not validate_output_path(output_path, quiet=args.quiet):
            return 1
    else:
        output_path = generate_and_validate_output_path(
            log_dir=log_dir,
            testee_model=privacy_config.testee_model,
            mode=f"{stage}_attack",
            strategies=strategy_names,
            max_samples=privacy_config.max_samples,
            suffix=f"_{privacy_config.testee_system_prompt_mode}",
            quiet=args.quiet,
        )
        if output_path is None:
            return 1

    if not pending_keys:
        if not args.quiet:
            print("\n[INFO] No pending privacy attack work. Saving final artifact.")
        pipeline = _make_dummy_pipeline(privacy_config)
        summary = create_attack_summary(
            baseline_results=baseline_results,
            attack_results=existing_results,
            attack_strategies=strategy_names,
            baseline_population=len(baseline_results),
            attack_population=len(expected_keys),
            strategy_populations=strategy_populations,
            intersection_population=(
                eligibility["eligible_case_count"] if eligibility is not None else 0
            ),
        )
        if stage == "individual":
            expected_case_ids = [
                result.test_case_id
                for result in baseline_results
                if is_baseline_safe_for_attack(
                    result,
                    baseline_expected_attempts,
                )
            ]
            eligibility = compute_combined_eligibility(
                existing_results,
                expected_attempts=privacy_config.num_attempts,
                expected_case_ids=expected_case_ids,
            )
        pipeline.save_results(
            results=existing_results,
            summary=summary,
            output_path=str(output_path),
            metadata=make_metadata(False),
            baseline_results=evaluation_baseline_results,
            eligibility=eligibility,
        )
        remove_checkpoint(output_path)
        return 0

    model_pool = ModelPool()
    testee = Testee(
        model_id=privacy_config.testee_model,
        model_pool=model_pool,
        system_prompt=privacy_config.testee_system_prompt,
        config=privacy_config.testee_config,
    )
    grader = PrivacyGrader(
        model_id=privacy_config.grader_model,
        model_pool=model_pool,
        config=privacy_config.grader_config,
        max_structured_output_retries=privacy_config.grader_max_structured_output_retries,
    )
    generator = PHIGenerator(
        model_id=privacy_config.generator_model,
        model_pool=model_pool,
        config=privacy_config.generator_config,
        basic_identifier_keys=privacy_config.basic_identifier_keys,
        extra_identifier_keys=privacy_config.extra_identifier_keys,
    )
    attack_strategies = create_attack_strategies_from_config(
        privacy_config, model_pool
    )
    pipeline = PrivacyPipeline(
        testee=testee,
        grader=grader,
        generator=generator,
        attack_strategies=attack_strategies,
        verbose=not args.quiet,
    )

    def progress_callback(current_results):
        combined = merge_unique(
            existing_results,
            current_results,
            key=lambda result: (result.test_case_id, result.attack_strategy),
        )
        pipeline.save_results(
            results=combined,
            summary=create_attack_summary(
                baseline_results=baseline_results,
                attack_results=combined,
                attack_strategies=strategy_names,
                baseline_population=len(baseline_results),
                attack_population=len(expected_keys),
                strategy_populations=strategy_populations,
                intersection_population=(
                    eligibility["eligible_case_count"] if eligibility is not None else 0
                ),
            ),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
            baseline_results=evaluation_baseline_results,
            eligibility=eligibility,
        )

    try:
        attack_results, summary = pipeline.run_attack(
            baseline_results=evaluation_baseline_results,
            num_attempts=privacy_config.num_attempts,
            progress_callback=progress_callback,
            save_every=10,
            completed_attack_keys=processed_keys,
            baseline_expected_attempts=baseline_expected_attempts,
        )
    except Exception as exc:
        print(f"[ERROR] Privacy attack evaluation failed: {exc}")
        return 1

    attack_results = merge_unique(
        existing_results,
        attack_results,
        key=lambda result: (result.test_case_id, result.attack_strategy),
    )
    summary = create_attack_summary(
        baseline_results=baseline_results,
        attack_results=attack_results,
        attack_strategies=strategy_names,
        baseline_population=len(baseline_results),
        attack_population=len(expected_keys),
        strategy_populations=strategy_populations,
        intersection_population=(
            eligibility["eligible_case_count"] if eligibility is not None else 0
        ),
    )

    if stage == "individual":
        expected_case_ids = [
            result.test_case_id
            for result in baseline_results
            if is_baseline_safe_for_attack(
                result,
                baseline_expected_attempts,
            )
        ]
        eligibility = compute_combined_eligibility(
            attack_results,
            expected_attempts=privacy_config.num_attempts,
            expected_case_ids=expected_case_ids,
        )

    pipeline.save_results(
        results=attack_results,
        summary=summary,
        output_path=str(output_path),
        metadata=make_metadata(False),
        baseline_results=evaluation_baseline_results,
        eligibility=eligibility,
    )
    remove_checkpoint(output_path)

    if not args.quiet:
        summary.print_summary()
        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
