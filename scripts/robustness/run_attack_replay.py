#!/usr/bin/env python3
"""Evaluate reusable Robustness attacks against a target's baseline-correct cohort."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from med_red_team.data import TestCase
from med_red_team.grader import SimpleGrader
from med_red_team.model_pool import ModelPool
from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessReplayItem,
    RobustnessResult,
    build_robustness_metadata,
    is_baseline_result_eligible_for_attack,
    robustness_file_identity,
    robustness_population_fingerprint,
    robustness_resume_signature,
    strategy_order_from_metadata,
    validate_attacked_dataset_rows,
    validate_replay_original_binding,
    validate_robustness_attack_generation_metadata,
    validate_robustness_baseline_metadata,
    validate_robustness_baseline_results,
)
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique, remove_checkpoint
from med_red_team.shared.config_loading import validate_sample_limit
from med_red_team.testee import Testee
from scripts.utils import (
    generate_and_validate_output_path,
    load_results_from_json,
    validate_resume,
)


POPULATION_POLICY = "attack_artifact_order_then_baseline_eligible_then_limit"


@dataclass(frozen=True)
class ReplayPopulationItem:
    """One member of the frozen paired replay population."""

    sample_number: int
    source_ordinal: int
    baseline_result: RobustnessResult
    original_payload: Dict[str, Any]
    attacked_payload: Dict[str, Any]
    attack_metadata: Dict[str, Any]
    classification: str
    attacked_test_case: Optional[TestCase]

    @property
    def test_case_id(self) -> str:
        return self.baseline_result.test_case_id

    @property
    def attacks_applied(self) -> List[str]:
        return list(self.attack_metadata.get("strategies_applied", []))


# ==============================================================================
# CLI
# ==============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a complete pre-generated Robustness attack artifact against "
            "the baseline-correct cohort of one target model"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.robustness.run_attack_replay \
      --baseline-results logs/model/robustness_baseline.json \
      --attacked-dataset logs/attacked_datasets/generated.json
        """,
    )
    parser.add_argument(
        "--baseline-results",
        required=True,
        help="Complete original-question Robustness baseline artifact",
    )
    parser.add_argument(
        "--attacked-dataset",
        required=True,
        help="Complete artifact produced by run_generate_attacks.py",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum baseline-correct joined cases to replay (default: all)",
    )
    parser.add_argument(
        "--log-dir",
        help="Output directory (default: baseline artifact directory)",
    )
    parser.add_argument(
        "--resume",
        help="Existing attack_replay result artifact or checkpoint to resume",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    return parser.parse_args(argv)


# ==============================================================================
# Input and population validation
# ==============================================================================


def _baseline_dataset_identity(metadata: Dict[str, Any]) -> Dict[str, str]:
    source = metadata.get("source") if isinstance(metadata.get("source"), dict) else {}
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    path = source.get("dataset_path") or dataset.get("path")
    sha256 = source.get("dataset_sha256")
    if not isinstance(path, str) or not path:
        raise ValueError("Baseline metadata is missing source.dataset_path")
    if not isinstance(sha256, str) or not sha256:
        raise ValueError("Baseline metadata is missing source.dataset_sha256")
    return {"path": str(Path(path).resolve()), "sha256": sha256}


def _validate_complete_row_count(
    metadata: Dict[str, Any],
    *,
    actual_count: int,
    label: str,
    metadata_key: str,
) -> None:
    container: Any = metadata
    for part in metadata_key.split("."):
        if not isinstance(container, dict) or part not in container:
            raise ValueError(
                f"{label} metadata is missing declared row count {metadata_key}"
            )
        container = container[part]
    if not isinstance(container, int) or isinstance(container, bool):
        raise ValueError(f"{label} metadata row count must be an integer")
    if container != actual_count:
        raise ValueError(
            f"{label} row count mismatch: metadata declares {container}, "
            f"artifact contains {actual_count}"
        )


def _validate_baseline_target_binding(metadata: Dict[str, Any]) -> None:
    config = metadata.get("config")
    models = metadata.get("models")
    if not isinstance(config, dict) or not isinstance(models, dict):
        raise ValueError("Baseline metadata requires config and models objects")
    target = models.get("target")
    grader = models.get("grader")
    if not isinstance(target, dict) or not isinstance(grader, dict):
        raise ValueError("Baseline metadata requires target and grader model records")

    expected_target = {
        "model_id": config.get("testee_model"),
        "generation_config": config.get("testee_config", {}),
        "system_prompt": config.get("testee_system_prompt", ""),
    }
    actual_target = {
        "model_id": target.get("model_id"),
        "generation_config": target.get("generation_config", {}),
        "system_prompt": target.get("system_prompt", ""),
    }
    if actual_target != expected_target:
        raise ValueError(
            "Baseline config target settings do not match models.target metadata"
        )
    if grader.get("type") != config.get("grader_type", "simple"):
        raise ValueError(
            "Baseline config grader_type does not match models.grader metadata"
        )


def _source_strategy_config(
    metadata: Dict[str, Any],
    strategy_order: List[str],
) -> Dict[str, Dict[str, Any]]:
    models = metadata.get("models") if isinstance(metadata.get("models"), dict) else {}
    configured = models.get("attacker_strategies")
    if not isinstance(configured, dict):
        config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        configured = config.get("attacker_strategies", {})
    if not isinstance(configured, dict):
        raise ValueError("Attack-generation metadata is missing attacker strategy config")

    missing = [name for name in strategy_order if name not in configured]
    if missing:
        raise ValueError(
            "Attack-generation metadata is missing strategy configuration for "
            f"{missing}"
        )
    return {name: configured[name] for name in strategy_order}


def _options_match(left: Any, right: Any) -> bool:
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and list(left.items()) == list(right.items())
    )


def _is_exact_noop(item: Dict[str, Any]) -> bool:
    original = item["original"]
    attacked = item["attacked"]
    return (
        attacked.get("question") == original.get("question")
        and _options_match(attacked.get("options"), original.get("options"))
        and attacked.get("correct_answer") == original.get("correct_answer")
    )


def _classify_population_item(row: Dict[str, Any]) -> str:
    attack_metadata = row["attack_metadata"]
    if attack_metadata["manipulation_failed"] is True:
        return "declared_failure"
    if _is_exact_noop(row):
        return "invalid_noop"
    return "valid"


def _build_population(
    attacked_rows: List[Dict[str, Any]],
    baseline_by_id: Dict[str, RobustnessResult],
    *,
    max_samples: Optional[int],
) -> tuple[List[ReplayPopulationItem], int]:
    eligible_join: List[tuple[int, RobustnessResult, Dict[str, Any]]] = []
    for source_ordinal, row in enumerate(attacked_rows, start=1):
        baseline_result = baseline_by_id.get(row["original"]["id"])
        if baseline_result is None:
            continue
        validate_replay_original_binding(baseline_result, row["original"])
        if not is_baseline_result_eligible_for_attack(baseline_result):
            continue
        eligible_join.append((source_ordinal, baseline_result, row))

    joined_eligible_count = len(eligible_join)
    selected = eligible_join if max_samples is None else eligible_join[:max_samples]
    population: List[ReplayPopulationItem] = []
    for sample_number, (source_ordinal, baseline_result, row) in enumerate(
        selected,
        start=1,
    ):
        classification = _classify_population_item(row)
        attacked_test_case = None
        if classification == "valid":
            attacked = row["attacked"]
            attacked_test_case = TestCase(
                id=attacked["id"],
                question=attacked["question"],
                options=attacked["options"],
                correct_answer=attacked["correct_answer"],
                task_type="multiple_choice",
                metadata=dict(row["attack_metadata"]),
            )
        population.append(
            ReplayPopulationItem(
                sample_number=sample_number,
                source_ordinal=source_ordinal,
                baseline_result=baseline_result,
                original_payload=dict(row["original"]),
                attacked_payload=dict(row["attacked"]),
                attack_metadata=dict(row["attack_metadata"]),
                classification=classification,
                attacked_test_case=attacked_test_case,
            )
        )
    return population, joined_eligible_count


def _static_failure_result(item: ReplayPopulationItem) -> RobustnessResult:
    attack_metadata = item.attack_metadata
    declared_status = attack_metadata.get("status")
    declared_category = attack_metadata.get("failure_category")
    if item.classification == "invalid_noop":
        status = "invalid_noop"
        failure_category = "invalid_noop"
        reason = "Generated attack made no change to question, options, or answer"
    else:
        failure_category = declared_category or (
            declared_status if declared_status != "generated" else None
        ) or "not_applicable"
        status = (
            declared_status
            if isinstance(declared_status, str) and declared_status != "generated"
            else failure_category
        )
        reason = attack_metadata.get("reason") or "Generated attack was invalid"

    baseline = item.baseline_result
    attacked = item.attacked_payload
    return RobustnessResult(
        sample_number=item.sample_number,
        test_case_id=item.test_case_id,
        original_question=baseline.original_question,
        original_options=dict(baseline.original_options),
        original_correct_answer=baseline.original_correct_answer,
        original_response=baseline.original_response,
        original_correct=True,
        manipulated_question=attacked["question"],
        manipulated_options=dict(attacked["options"]),
        manipulated_correct_answer=attacked["correct_answer"],
        attacks_applied=item.attacks_applied,
        manipulation_failed=True,
        status=status,
        status_reason=reason,
        failure_stage="attack_generation",
        failure_category=failure_category,
        eligible_baseline_correct=True,
        valid_target_tested=False,
        test_case_metadata=dict(attack_metadata),
        metadata={
            "attack_metadata": dict(attack_metadata),
            "replay": {
                "population_ordinal": item.sample_number,
                "source_ordinal": item.source_ordinal,
            },
        },
        skipped=True,
        skip_reason=reason,
    )


def _population_fingerprint(population: List[ReplayPopulationItem]) -> str:
    return robustness_population_fingerprint(
        [
            {
                "population_ordinal": item.sample_number,
                "source_ordinal": item.source_ordinal,
                "test_case_id": item.test_case_id,
            }
            for item in population
        ]
    )


def _canonicalize_results(
    population: List[ReplayPopulationItem],
    results: List[RobustnessResult],
) -> List[RobustnessResult]:
    order = {item.test_case_id: item.sample_number for item in population}
    return sorted(results, key=lambda result: order[result.test_case_id])


def _validate_resume_outcome(
    result: RobustnessResult,
    item: ReplayPopulationItem,
) -> None:
    if item.classification != "valid":
        expected = _static_failure_result(item)
        comparable_fields = (
            "manipulation_failed",
            "status",
            "status_reason",
            "failure_stage",
            "failure_category",
            "valid_target_tested",
            "skipped",
            "skip_reason",
            "manipulated_response",
            "manipulated_correct",
        )
        if any(
            getattr(result, field) != getattr(expected, field)
            for field in comparable_fields
        ):
            raise ValueError(
                f"Resume static outcome mismatch for {result.test_case_id!r}"
            )
        return

    if result.manipulation_failed:
        raise ValueError(
            f"Resume target outcome cannot mark source manipulation failed for "
            f"{result.test_case_id!r}"
        )
    if result.status == "target_safe":
        coherent = (
            result.manipulated_correct is True
            and result.valid_target_tested is True
            and not result.skipped
        )
    elif result.status == "fooled":
        coherent = (
            result.manipulated_correct is False
            and result.valid_target_tested is True
            and not result.skipped
        )
    elif result.status == "parser_error":
        coherent = (
            result.manipulated_correct is None
            and result.valid_target_tested is False
            and result.skipped
            and result.failure_stage == "target_response_parse"
            and result.failure_category == "parser_error"
        )
    elif result.status == "generation_error":
        coherent = (
            result.manipulated_correct is None
            and result.valid_target_tested is False
            and result.skipped
            and result.failure_stage == "target_generation_or_grading"
            and result.failure_category == "generation_error"
        )
    else:
        coherent = False
    if not coherent:
        raise ValueError(
            f"Resume target outcome is not a coherent terminal result for "
            f"{result.test_case_id!r}"
        )


def _validate_resume_rows(
    existing_results: List[RobustnessResult],
    population: List[ReplayPopulationItem],
) -> None:
    population_by_id = {item.test_case_id: item for item in population}
    resume_ids = [result.test_case_id for result in existing_results]
    if len(set(resume_ids)) != len(resume_ids):
        raise ValueError("Resume artifact contains duplicate test_case_id values")

    unexpected = [case_id for case_id in resume_ids if case_id not in population_by_id]
    if unexpected:
        raise ValueError(
            "Resume artifact contains results outside the replay population: "
            f"{unexpected[:5]}"
        )

    canonical_ids = sorted(
        resume_ids,
        key=lambda case_id: population_by_id[case_id].sample_number,
    )
    if resume_ids != canonical_ids:
        raise ValueError("Resume results are not in canonical replay population order")

    for result in existing_results:
        item = population_by_id[result.test_case_id]
        baseline = item.baseline_result
        if not isinstance(result.metadata, dict):
            raise ValueError(
                f"Resume metadata must be an object for {result.test_case_id!r}"
            )
        replay_metadata = result.metadata.get("replay", {})
        if result.sample_number != item.sample_number:
            raise ValueError(
                f"Resume sample_number mismatch for {result.test_case_id!r}"
            )
        if replay_metadata.get("population_ordinal") != item.sample_number:
            raise ValueError(
                f"Resume population ordinal mismatch for {result.test_case_id!r}"
            )
        if replay_metadata.get("source_ordinal") != item.source_ordinal:
            raise ValueError(
                f"Resume source ordinal mismatch for {result.test_case_id!r}"
            )
        if (
            result.original_question != baseline.original_question
            or not _options_match(result.original_options, baseline.original_options)
            or result.original_correct_answer != baseline.original_correct_answer
            or result.original_response != baseline.original_response
            or result.original_correct is not True
            or result.eligible_baseline_correct is not True
        ):
            raise ValueError(
                f"Resume baseline pairing mismatch for {result.test_case_id!r}"
            )
        attacked = item.attacked_payload
        if (
            result.manipulated_question != attacked["question"]
            or not _options_match(result.manipulated_options, attacked["options"])
            or result.manipulated_correct_answer != attacked["correct_answer"]
            or result.attacks_applied != item.attacks_applied
            or result.test_case_metadata != item.attack_metadata
            or result.metadata.get("attack_metadata") != item.attack_metadata
        ):
            raise ValueError(
                f"Resume attacked-source mismatch for {result.test_case_id!r}"
            )
        _validate_resume_outcome(result, item)


def _assert_final_population(
    results: List[RobustnessResult],
    population: List[ReplayPopulationItem],
) -> None:
    expected_ids = [item.test_case_id for item in population]
    actual_ids = [result.test_case_id for result in results]
    if actual_ids != expected_ids:
        raise ValueError("Final replay results do not exactly match the frozen population")
    for item, result in zip(population, results):
        if result.sample_number != item.sample_number:
            raise ValueError(
                f"Final replay sample_number mismatch for {item.test_case_id!r}"
            )


# ==============================================================================
# Main workflow
# ==============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_sample_limit(args.max_samples)
    except ValueError as exc:
        print(f"[ERROR] Invalid sample limit: {exc}")
        return 1

    baseline_path = Path(args.baseline_results)
    attacked_path = Path(args.attacked_dataset)
    if not baseline_path.exists():
        print(f"[ERROR] Baseline results file not found: {baseline_path}")
        return 1
    if not attacked_path.exists():
        print(f"[ERROR] Attacked dataset file not found: {attacked_path}")
        return 1

    try:
        if not args.quiet:
            print(f"[Loading] Baseline results: {baseline_path}")
        baseline_results, baseline_metadata = load_results_from_json(
            baseline_path,
            data_key="results",
            quiet=args.quiet,
            reconstruct_fn=RobustnessResult.from_dict,
        )
        validate_robustness_baseline_metadata(
            baseline_metadata,
            expected_evaluation_mode="original",
        )
        _validate_baseline_target_binding(baseline_metadata)
        baseline_by_id = validate_robustness_baseline_results(baseline_results)
        _validate_complete_row_count(
            baseline_metadata,
            actual_count=len(baseline_results),
            label="Baseline artifact",
            metadata_key="dataset.total_loaded",
        )
        baseline_dataset_identity = _baseline_dataset_identity(baseline_metadata)

        if not args.quiet:
            print(f"[Loading] Generated attacks: {attacked_path}")
        attacked_rows, attacked_metadata = load_results_from_json(
            attacked_path,
            data_key="attacked_test_cases",
            quiet=args.quiet,
        )
        strategy_order = strategy_order_from_metadata(attacked_metadata)
        attacked_dataset_identity = validate_robustness_attack_generation_metadata(
            attacked_metadata,
            expected_strategy_order=strategy_order,
            require_complete=True,
        )
        validate_attacked_dataset_rows(
            attacked_rows,
            expected_strategy_order=strategy_order,
        )
        _validate_complete_row_count(
            attacked_metadata,
            actual_count=len(attacked_rows),
            label="Attack-generation artifact",
            metadata_key="total_samples",
        )
        strategy_config = _source_strategy_config(attacked_metadata, strategy_order)

        if baseline_dataset_identity["sha256"] != attacked_dataset_identity["sha256"]:
            raise ValueError(
                "Baseline and attack-generation artifacts refer to different "
                "source dataset content"
            )

        baseline_file_identity = robustness_file_identity(baseline_path)
        attacked_file_identity = robustness_file_identity(attacked_path)
        population, joined_eligible_count = _build_population(
            attacked_rows,
            baseline_by_id,
            max_samples=args.max_samples,
        )
    except Exception as exc:
        print(f"[ERROR] Invalid replay input: {exc}")
        return 1

    selected_population = len(population)
    population_sha256 = _population_fingerprint(population)
    static_results = [
        _static_failure_result(item)
        for item in population
        if item.classification != "valid"
    ]

    log_dir = Path(args.log_dir) if args.log_dir else baseline_path.parent
    baseline_config_dict = dict(baseline_metadata["config"])
    baseline_config_dict.update(
        {
            "attacker_strategies": strategy_config,
            "max_samples": args.max_samples,
            "log_dir": str(log_dir),
        }
    )
    try:
        config = RobustnessConfig.from_dict(baseline_config_dict)
    except Exception as exc:
        print(f"[ERROR] Invalid replay configuration: {exc}")
        return 1

    dataset_info = dict(baseline_metadata.get("dataset", {}))
    dataset_info.pop("sample_limit", None)
    source_info = {
        "baseline_results_file": baseline_file_identity["path"],
        "baseline_results_sha256": baseline_file_identity["sha256"],
        "attacked_dataset_path": attacked_file_identity["path"],
        "attacked_dataset_sha256": attacked_file_identity["sha256"],
        "dataset_path": baseline_dataset_identity["path"],
        "dataset_sha256": baseline_dataset_identity["sha256"],
        "workflow": "run_attack_replay",
    }

    def make_metadata(is_partial: bool) -> Dict[str, Any]:
        return build_robustness_metadata(
            phase="attack_replay",
            config=config,
            target_model=config.testee_model,
            dataset_info=dataset_info,
            source=source_info,
            is_partial=is_partial,
            sample_limit=args.max_samples,
            attack_strategies=strategy_order,
            attack_mode="pre_generated_replay",
            extra={
                "population_policy": POPULATION_POLICY,
                "attack_artifact_rows": len(attacked_rows),
                "baseline_rows": len(baseline_results),
                "joined_eligible_rows": joined_eligible_count,
                "selected_population": selected_population,
                "ordered_population_sha256": population_sha256,
                "declared_source_failures": sum(
                    item.classification == "declared_failure" for item in population
                ),
                "invalid_noops": sum(
                    item.classification == "invalid_noop" for item in population
                ),
                "target_testable_rows": sum(
                    item.classification == "valid" for item in population
                ),
            },
        )

    existing_results: List[RobustnessResult] = []
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"[ERROR] Resume file not found: {resume_path}")
            return 1
        try:
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
            _validate_resume_rows(existing_results, population)
        except Exception as exc:
            print(f"[ERROR] Resume validation failed: {exc}")
            return 1

    seed_results = merge_unique(
        existing_results,
        static_results,
        key=lambda result: result.test_case_id,
    )
    seed_results = _canonicalize_results(population, seed_results)
    processed_ids = {result.test_case_id for result in seed_results}
    remaining_items = [
        RobustnessReplayItem(
            sample_number=item.sample_number,
            source_ordinal=item.source_ordinal,
            baseline_result=item.baseline_result,
            attacked_test_case=item.attacked_test_case,
            attacks_applied=item.attacks_applied,
            attack_metadata=item.attack_metadata,
        )
        for item in population
        if item.classification == "valid" and item.test_case_id not in processed_ids
    ]

    if not args.quiet:
        print("\n" + "=" * 80)
        print("ROBUSTNESS PRE-GENERATED ATTACK REPLAY")
        print("=" * 80)
        print(f"  Target model: {config.testee_model}")
        print(f"  Strategies: {', '.join(strategy_order)}")
        print(f"  Baseline rows: {len(baseline_results)}")
        print(f"  Baseline-correct joined rows: {joined_eligible_count}")
        print(f"  Selected replay population: {selected_population}")
        print(f"  Remaining target calls: {len(remaining_items)}")

    output_path = generate_and_validate_output_path(
        log_dir=str(log_dir),
        testee_model=config.testee_model,
        mode="robustness_attack_replay",
        strategies=strategy_order,
        max_samples=(selected_population if args.max_samples is not None else None),
        quiet=args.quiet,
    )
    if output_path is None:
        print("[ERROR] Cannot write replay output")
        return 1

    def summarize(
        pipeline: RobustnessPipeline,
        results: List[RobustnessResult],
    ):
        return pipeline._compute_summary(
            results,
            baseline_only=False,
            expected_eligible_population=selected_population,
            attack_strategy_order=strategy_order,
        )

    def save_final_without_provider() -> int:
        pipeline = RobustnessPipeline(
            testee=None,
            grader=None,
            attack_strategies=[],
            verbose=not args.quiet,
        )
        results = _canonicalize_results(population, seed_results)
        _assert_final_population(results, population)
        pipeline.save_results(
            results=results,
            summary=summarize(pipeline, results),
            output_path=str(output_path),
            metadata=make_metadata(False),
        )
        remove_checkpoint(output_path)
        if not args.quiet:
            print(f"\nReplay results saved to: {output_path}")
        return 0

    if not remaining_items:
        return save_final_without_provider()

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

    def progress_callback(current_results: List[RobustnessResult]) -> None:
        combined = merge_unique(
            seed_results,
            current_results,
            key=lambda result: result.test_case_id,
        )
        combined = _canonicalize_results(population, combined)
        pipeline.save_results(
            results=combined,
            summary=summarize(pipeline, combined),
            output_path=str(checkpoint_path(output_path)),
            metadata=make_metadata(True),
        )

    try:
        new_results, _ = pipeline.run_attack_replay(
            remaining_items,
            expected_eligible_population=selected_population,
            attack_strategy_order=strategy_order,
            seed_results=seed_results,
            progress_callback=progress_callback,
            save_every=10,
        )
    except Exception as exc:
        print(f"[ERROR] Attack replay failed: {exc}")
        return 1

    results = merge_unique(
        seed_results,
        new_results,
        key=lambda result: result.test_case_id,
    )
    results = _canonicalize_results(population, results)
    try:
        _assert_final_population(results, population)
    except ValueError as exc:
        print(f"[ERROR] Replay finalization failed: {exc}")
        return 1

    summary = summarize(pipeline, results)
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
        print(f"  Eligible baseline-correct population: {summary.eligible_baseline_correct}")
        print(f"  Valid target-tested attacks: {summary.valid_target_tested}")
        print(f"  Attack coverage: {summary.attack_coverage:.1%}")
        if summary.valid_target_tested:
            print(
                "  Conditional attack success: "
                f"{summary.conditional_attack_success_rate:.1%}"
            )
            print(f"  Conditional robustness: {summary.conditional_robustness:.1%}")
        else:
            print("  Conditional attack success: n/a")
            print("  Conditional robustness: n/a")
        print(
            "  End-to-end attack success: "
            f"{summary.end_to_end_attack_success_rate:.1%}"
        )
        print(f"  Results: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
