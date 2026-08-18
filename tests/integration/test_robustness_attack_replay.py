from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from med_red_team.data import TestCase as EvaluationCase
from med_red_team.grader import SimpleGrader
from med_red_team.models import ModelGenerationError, ModelResponse
from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessReplayItem,
    RobustnessResult,
    build_robustness_metadata,
    robustness_file_identity,
)
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.shared.checkpoints import checkpoint_path
from med_red_team.shared.io import read_json
from scripts.robustness import run_attack_replay


STRATEGY = "add_none_of_the_above"


def _response(answer: str, *, parse_error: dict | None = None) -> ModelResponse:
    metadata = {"parse_error": parse_error} if parse_error else {}
    return ModelResponse(
        raw_text=answer,
        model_id="fake-target",
        final_answer=answer,
        metadata=metadata,
    )


def _case(
    case_id: str,
    *,
    baseline_correct: bool = True,
    question: str | None = None,
    options: dict[str, str] | None = None,
    correct_answer: str = "A",
    attacked_question: str | None = None,
    attacked_options: dict[str, str] | None = None,
    attacked_correct_answer: str | None = None,
    manipulation_failed: bool = False,
    status: str | None = None,
    failure_category: str | None = None,
    reason: str | None = None,
) -> dict:
    original_question = question or f"Original {case_id}?"
    original_options = options or {"A": "alpha", "B": "beta"}
    return {
        "id": case_id,
        "baseline_correct": baseline_correct,
        "original": {
            "id": case_id,
            "question": original_question,
            "options": original_options,
            "correct_answer": correct_answer,
        },
        "attacked": {
            "id": case_id,
            "question": (
                attacked_question
                if attacked_question is not None
                else f"{original_question} Attacked."
            ),
            "options": attacked_options or dict(original_options),
            "correct_answer": attacked_correct_answer or correct_answer,
        },
        "attack_metadata": {
            "manipulation_failed": manipulation_failed,
            "status": status or (
                failure_category if manipulation_failed else "generated"
            ),
            "failure_category": failure_category,
            "reason": reason,
            "strategies_applied": [STRATEGY],
        },
    }


def _write_artifacts(
    root: Path,
    cases: list[dict],
    *,
    baseline_order: list[str] | None = None,
    baseline_partial: bool = False,
    baseline_mode: str = "original",
    attack_partial: bool = False,
) -> tuple[Path, Path, Path]:
    source_path = root / "source.jsonl"
    source_path.write_text('{"source":"same-content"}\n', encoding="utf-8")
    source_identity = robustness_file_identity(source_path)

    baseline_config = RobustnessConfig(
        testee_model="fake-target",
        dataset_path=str(source_path),
        attacker_strategies={},
    )
    baseline_metadata = build_robustness_metadata(
        phase="baseline",
        config=baseline_config,
        target_model="fake-target",
        dataset_info={
            "path": source_identity["path"],
            "total_loaded": len(cases),
        },
        source={
            "dataset_path": source_identity["path"],
            "dataset_sha256": source_identity["sha256"],
        },
        is_partial=baseline_partial,
        evaluation_mode=baseline_mode,
    )
    by_id = {case["id"]: case for case in cases}
    ordered_ids = baseline_order or [case["id"] for case in cases]
    baseline_results = []
    for sample_number, case_id in enumerate(ordered_ids, start=1):
        case = by_id[case_id]
        correct = case["baseline_correct"]
        original = case["original"]
        baseline_results.append(
            RobustnessResult(
                sample_number=sample_number,
                test_case_id=case_id,
                original_question=original["question"],
                original_options=dict(original["options"]),
                original_correct_answer=original["correct_answer"],
                original_response=original["correct_answer"] if correct else "B",
                original_correct=correct,
                status="baseline_correct" if correct else "baseline_incorrect",
                eligible_baseline_correct=correct,
            )
        )
    baseline_path = root / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "metadata": baseline_metadata,
                "summary": {},
                "results": [result.to_dict() for result in baseline_results],
            }
        ),
        encoding="utf-8",
    )

    attack_config = RobustnessConfig(
        testee_model="attack-generation-placeholder",
        dataset_path=str(source_path),
        attacker_strategies={STRATEGY: {}},
    )
    attack_metadata = build_robustness_metadata(
        phase="attack_generation",
        config=attack_config,
        target_model=attack_config.testee_model,
        dataset_info={"path": source_identity["path"]},
        source={
            "dataset_path": source_identity["path"],
            "dataset_sha256": source_identity["sha256"],
        },
        is_partial=attack_partial,
        attack_strategies=[STRATEGY],
        extra={"total_samples": len(cases)},
    )
    attack_path = root / "attacks.json"
    attack_path.write_text(
        json.dumps(
            {
                "metadata": attack_metadata,
                "attacked_test_cases": [
                    {
                        "original": case["original"],
                        "attacked": case["attacked"],
                        "attack_metadata": case["attack_metadata"],
                    }
                    for case in cases
                ],
            }
        ),
        encoding="utf-8",
    )
    return baseline_path, attack_path, source_path


def _run(
    baseline_path: Path,
    attack_path: Path,
    output_path: Path,
    *,
    responses: list[ModelResponse | Exception] | None = None,
    max_samples: int | None = None,
    resume: Path | None = None,
    forbid_provider: bool = False,
) -> tuple[int, list[str]]:
    prompts: list[str] = []
    queued = list(responses or [])

    class FakeTestee:
        def __init__(self, **_kwargs):
            pass

        def answer(self, prompt):
            prompts.append(prompt)
            next_value = queued.pop(0)
            if isinstance(next_value, Exception):
                raise next_value
            return next_value

    def forbidden_model_pool():
        raise AssertionError("ModelPool should not be constructed")

    argv = [
        "--baseline-results",
        str(baseline_path),
        "--attacked-dataset",
        str(attack_path),
        "--quiet",
    ]
    if max_samples is not None:
        argv.extend(["--max-samples", str(max_samples)])
    if resume is not None:
        argv.extend(["--resume", str(resume)])

    model_pool = forbidden_model_pool if forbid_provider else (lambda: object())
    with patch.object(
        run_attack_replay,
        "generate_and_validate_output_path",
        return_value=output_path,
    ), patch.object(run_attack_replay, "ModelPool", model_pool), patch.object(
        run_attack_replay,
        "Testee",
        FakeTestee,
    ):
        return run_attack_replay.main(argv), prompts


def test_replay_pairs_artifact_order_and_grades_attacked_options_and_answer(
    tmp_path: Path,
):
    cases = [
        _case(
            "valid-safe",
            attacked_question="Changed safe question?",
            attacked_options={"A": "wrong", "B": "right", "C": "other"},
            attacked_correct_answer="B",
        ),
        _case("baseline-wrong", baseline_correct=False),
        _case(
            "declared-partial",
            attacked_question="A partial transformation remains auditable?",
            manipulation_failed=True,
            status="not_applicable",
            failure_category="not_applicable",
            reason="Source strategy could not complete",
        ),
        _case("noop", attacked_question="Original noop?", question="Original noop?"),
        _case("valid-fooled", attacked_question="Changed fooled question?"),
    ]
    baseline_path, attack_path, _ = _write_artifacts(
        tmp_path,
        cases,
        baseline_order=list(reversed([case["id"] for case in cases])),
    )
    output_path = tmp_path / "replay.json"

    code, prompts = _run(
        baseline_path,
        attack_path,
        output_path,
        responses=[_response("B"), _response("B")],
    )

    assert code == 0
    payload = read_json(output_path)
    assert [row["test_case_id"] for row in payload["results"]] == [
        "valid-safe",
        "declared-partial",
        "noop",
        "valid-fooled",
    ]
    assert [row["sample_number"] for row in payload["results"]] == [1, 2, 3, 4]
    assert len(prompts) == 2
    assert "(C) other" in prompts[0]

    safe, declared, noop, fooled = payload["results"]
    assert safe["original_response"] == "A"
    assert safe["manipulated_options"] == {
        "A": "wrong",
        "B": "right",
        "C": "other",
    }
    assert safe["manipulated_correct_answer"] == "B"
    assert safe["manipulated_correct"] is True
    assert safe["status"] == "target_safe"

    assert declared["manipulation_failed"] is True
    assert declared["status"] == "not_applicable"
    assert declared["manipulated_question"] == (
        "A partial transformation remains auditable?"
    )
    assert declared["valid_target_tested"] is False
    assert noop["status"] == "invalid_noop"
    assert noop["valid_target_tested"] is False
    assert fooled["manipulated_correct"] is False
    assert fooled["status"] == "fooled"

    summary = payload["summary"]
    assert summary["eligible_baseline_correct"] == 4
    assert summary["valid_target_tested"] == 2
    assert summary["target_safe"] == 1
    assert summary["fooled"] == 1
    assert summary["attack_coverage"] == pytest.approx(0.5)
    assert summary["conditional_attack_success_rate"] == pytest.approx(0.5)
    assert summary["conditional_robustness"] == pytest.approx(0.5)
    assert summary["end_to_end_attack_success_rate"] == pytest.approx(0.25)

    metadata = payload["metadata"]
    assert metadata["phase"] == "attack_replay"
    assert metadata["attack_mode"] == "pre_generated_replay"
    assert metadata["models"]["target"]["model_id"] == "fake-target"
    assert metadata["models"]["attack_strategy_order"] == [STRATEGY]
    assert metadata["selected_population"] == 4
    assert metadata["is_partial"] is False


def test_replay_applies_limit_after_eligible_join_and_never_backfills(tmp_path: Path):
    cases = [
        _case("first-valid"),
        _case("ineligible", baseline_correct=False),
        _case(
            "declared-failure",
            manipulation_failed=True,
            status="not_applicable",
            failure_category="not_applicable",
        ),
        _case("noop", attacked_question="Original noop?", question="Original noop?"),
        _case("later-valid"),
    ]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / "limited.json"

    code, prompts = _run(
        baseline_path,
        attack_path,
        output_path,
        max_samples=3,
        responses=[_response("A")],
    )

    assert code == 0
    payload = read_json(output_path)
    assert [row["test_case_id"] for row in payload["results"]] == [
        "first-valid",
        "declared-failure",
        "noop",
    ]
    assert len(prompts) == 1
    assert payload["metadata"]["joined_eligible_rows"] == 4
    assert payload["metadata"]["selected_population"] == 3
    assert payload["summary"]["eligible_baseline_correct"] == 3
    assert payload["summary"]["valid_target_tested"] == 1
    assert payload["summary"]["attack_coverage"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mode", ["zero-limit", "static-only"])
def test_replay_zero_work_paths_are_provider_free(tmp_path: Path, mode: str):
    if mode == "zero-limit":
        cases = [_case("valid")]
        max_samples = 0
        expected_ids = []
    else:
        cases = [
            _case(
                "failed",
                manipulation_failed=True,
                status="not_applicable",
                failure_category="not_applicable",
            ),
            _case("noop", attacked_question="Original noop?", question="Original noop?"),
        ]
        max_samples = None
        expected_ids = ["failed", "noop"]

    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / f"{mode}.json"
    stale_checkpoint = checkpoint_path(output_path)
    stale_checkpoint.write_text("stale", encoding="utf-8")

    code, prompts = _run(
        baseline_path,
        attack_path,
        output_path,
        max_samples=max_samples,
        forbid_provider=True,
    )

    assert code == 0
    assert prompts == []
    payload = read_json(output_path)
    assert [row["test_case_id"] for row in payload["results"]] == expected_ids
    assert payload["metadata"]["is_partial"] is False
    assert not stale_checkpoint.exists()


def test_replay_parser_error_is_not_counted_as_attack_success(tmp_path: Path):
    cases = [_case("parser-case")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / "parser.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        responses=[
            _response(
                "",
                parse_error={
                    "error_type": "empty_answer",
                    "error_message": "No final answer",
                    "details": {},
                },
            )
        ],
    )

    assert code == 0
    payload = read_json(output_path)
    row = payload["results"][0]
    assert row["status"] == "parser_error"
    assert row["valid_target_tested"] is False
    assert row["manipulated_correct"] is None
    assert payload["summary"]["fooled"] == 0
    assert payload["summary"]["parser_errors"] == 1
    assert payload["summary"]["attack_coverage"] == 0.0


def test_replay_provider_error_is_fail_fast_with_fixed_checkpoint_denominator(
    tmp_path: Path,
):
    cases = [
        _case("first-valid"),
        _case("noop", attacked_question="Original noop?", question="Original noop?"),
        _case("second-valid"),
    ]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / "failed-run.json"

    code, prompts = _run(
        baseline_path,
        attack_path,
        output_path,
        responses=[_response("B"), ModelGenerationError("provider unavailable")],
    )

    assert code == 1
    assert len(prompts) == 2
    assert not output_path.exists()
    partial_path = checkpoint_path(output_path)
    assert partial_path.exists()
    payload = read_json(partial_path)
    assert payload["metadata"]["is_partial"] is True
    assert [row["test_case_id"] for row in payload["results"]] == [
        "first-valid",
        "noop",
    ]
    assert payload["summary"]["eligible_baseline_correct"] == 3
    assert payload["summary"]["valid_target_tested"] == 1
    assert payload["summary"]["fooled"] == 1
    assert payload["summary"]["attack_coverage"] == pytest.approx(1 / 3)
    assert payload["summary"]["end_to_end_attack_success_rate"] == pytest.approx(1 / 3)


def test_replay_all_complete_resume_is_provider_free_and_content_bound(tmp_path: Path):
    cases = [_case("resumed")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    first_output = tmp_path / "first.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        first_output,
        responses=[_response("A")],
    )
    assert code == 0

    resumed_output = tmp_path / "resumed.json"
    code, prompts = _run(
        baseline_path,
        attack_path,
        resumed_output,
        resume=first_output,
        forbid_provider=True,
    )
    assert code == 0
    assert prompts == []
    assert read_json(resumed_output)["results"] == read_json(first_output)["results"]

    attack_payload = read_json(attack_path)
    attack_path.write_text(json.dumps(attack_payload, indent=2), encoding="utf-8")
    rejected_output = tmp_path / "rejected.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        rejected_output,
        resume=first_output,
        forbid_provider=True,
    )
    assert code == 1
    assert not rejected_output.exists()


def test_replay_rejects_wrong_order_resume_rows_before_provider(tmp_path: Path):
    cases = [_case("one"), _case("two")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    completed = tmp_path / "completed.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        completed,
        responses=[_response("A"), _response("A")],
    )
    assert code == 0

    payload = read_json(completed)
    payload["results"].reverse()
    tampered_resume = tmp_path / "wrong-order.json"
    tampered_resume.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        resume=tampered_resume,
        forbid_provider=True,
    )
    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("baseline_partial", "baseline_mode", "attack_partial"),
    [
        (True, "original", False),
        (False, "attacked", False),
        (False, "original", True),
    ],
)
def test_replay_rejects_partial_or_wrong_mode_inputs_before_provider(
    tmp_path: Path,
    baseline_partial: bool,
    baseline_mode: str,
    attack_partial: bool,
):
    cases = [_case("case")]
    baseline_path, attack_path, _ = _write_artifacts(
        tmp_path,
        cases,
        baseline_partial=baseline_partial,
        baseline_mode=baseline_mode,
        attack_partial=attack_partial,
    )
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize("duplicate_source", ["baseline", "attacks"])
def test_replay_rejects_duplicate_case_ids_before_provider(
    tmp_path: Path,
    duplicate_source: str,
):
    cases = [_case("case")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    if duplicate_source == "baseline":
        payload = read_json(baseline_path)
        payload["results"].append(deepcopy(payload["results"][0]))
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = read_json(attack_path)
        payload["attacked_test_cases"].append(
            deepcopy(payload["attacked_test_cases"][0])
        )
        attack_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path = tmp_path / "unused.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


def test_replay_rejects_dataset_sha_and_exact_original_mismatches(tmp_path: Path):
    cases = [_case("case")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / "unused.json"

    attack_payload = read_json(attack_path)
    attack_payload["metadata"]["source"]["dataset_sha256"] = "different"
    attack_path.write_text(json.dumps(attack_payload), encoding="utf-8")
    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )
    assert code == 1

    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    attack_payload = read_json(attack_path)
    attack_payload["attacked_test_cases"][0]["original"]["options"] = {
        "B": "beta",
        "A": "alpha",
    }
    attack_path.write_text(json.dumps(attack_payload), encoding="utf-8")
    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )
    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize("answer", ["", "I cannot answer", "Z"])
def test_replay_rejects_target_answers_without_a_valid_option_label(
    tmp_path: Path,
    answer: str,
):
    cases = [_case("invalid-label")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    output_path = tmp_path / "invalid-label.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        responses=[_response(answer)],
    )

    assert code == 0
    payload = read_json(output_path)
    row = payload["results"][0]
    assert row["status"] == "parser_error"
    assert row["failure_stage"] == "target_response_parse"
    assert row["failure_category"] == "parser_error"
    assert row["valid_target_tested"] is False
    assert row["manipulated_correct"] is None
    assert payload["summary"]["fooled"] == 0
    assert payload["summary"]["parser_errors"] == 1


def test_replay_rejects_outcome_less_resume_rows_before_provider(tmp_path: Path):
    cases = [_case("resume-case")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    completed = tmp_path / "completed.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        completed,
        responses=[_response("A")],
    )
    assert code == 0

    payload = read_json(completed)
    payload["results"][0].update({
        "status": "unknown",
        "manipulated_correct": None,
        "valid_target_tested": False,
    })
    invalid_resume = tmp_path / "outcome-less.json"
    invalid_resume.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        resume=invalid_resume,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize("truncated_source", ["baseline", "attacks"])
def test_replay_rejects_complete_artifacts_with_truncated_rows(
    tmp_path: Path,
    truncated_source: str,
):
    cases = [_case("one"), _case("two")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    if truncated_source == "baseline":
        payload = read_json(baseline_path)
        payload["results"].pop()
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = read_json(attack_path)
        payload["attacked_test_cases"].pop()
        attack_path.write_text(json.dumps(payload), encoding="utf-8")

    output_path = tmp_path / "unused.json"
    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


def test_replay_rejects_contradictory_baseline_outcome_before_provider(
    tmp_path: Path,
):
    cases = [_case("contradictory")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    payload = read_json(baseline_path)
    payload["results"][0]["eligible_baseline_correct"] = False
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize(
    "provenance_error",
    ["missing-success-chain", "wrong-failure-prefix"],
)
def test_replay_rejects_malformed_row_strategy_provenance(
    tmp_path: Path,
    provenance_error: str,
):
    cases = [_case("provenance")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    payload = read_json(attack_path)
    metadata = payload["attacked_test_cases"][0]["attack_metadata"]
    if provenance_error == "missing-success-chain":
        metadata["strategies_applied"] = []
    else:
        metadata.update({
            "manipulation_failed": True,
            "status": "not_applicable",
            "failure_category": "not_applicable",
            "strategies_applied": ["different_strategy"],
        })
    attack_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


def test_replay_binds_joined_baseline_incorrect_rows_before_filtering(
    tmp_path: Path,
):
    cases = [_case("baseline-wrong", baseline_correct=False)]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    payload = read_json(attack_path)
    payload["attacked_test_cases"][0]["original"]["question"] = "Mismatched?"
    attack_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


@pytest.mark.parametrize("noncanonical_source", ["baseline", "attacks"])
def test_replay_rejects_noncanonical_source_answers(
    tmp_path: Path,
    noncanonical_source: str,
):
    cases = [_case("answer")]
    baseline_path, attack_path, _ = _write_artifacts(tmp_path, cases)
    if noncanonical_source == "baseline":
        payload = read_json(baseline_path)
        payload["results"][0]["original_correct_answer"] = "a"
        baseline_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        payload = read_json(attack_path)
        payload["attacked_test_cases"][0]["attacked"]["correct_answer"] = "a"
        attack_path.write_text(json.dumps(payload), encoding="utf-8")
    output_path = tmp_path / "unused.json"

    code, _ = _run(
        baseline_path,
        attack_path,
        output_path,
        forbid_provider=True,
    )

    assert code == 1
    assert not output_path.exists()


def test_pipeline_replay_summary_includes_seed_rows():
    baseline = RobustnessResult(
        sample_number=1,
        test_case_id="target-row",
        original_question="Original?",
        original_options={"A": "right", "B": "wrong"},
        original_correct_answer="A",
        original_response="A",
        original_correct=True,
        eligible_baseline_correct=True,
        status="baseline_correct",
    )
    attack_metadata = {
        "manipulation_failed": False,
        "status": "generated",
        "failure_category": None,
        "reason": None,
        "strategies_applied": [STRATEGY],
    }
    replay_item = RobustnessReplayItem(
        sample_number=1,
        source_ordinal=1,
        baseline_result=baseline,
        attacked_test_case=EvaluationCase(
            id="target-row",
            question="Changed?",
            options={"A": "right", "B": "wrong"},
            correct_answer="A",
            task_type="multiple_choice",
            metadata=attack_metadata,
        ),
        attacks_applied=[STRATEGY],
        attack_metadata=attack_metadata,
    )
    seed = RobustnessResult(
        sample_number=2,
        test_case_id="static-row",
        original_correct=True,
        eligible_baseline_correct=True,
        manipulation_failed=True,
        status="invalid_noop",
        failure_category="invalid_noop",
        skipped=True,
    )

    class FakeTestee:
        def answer(self, _prompt):
            return _response("A")

    pipeline = RobustnessPipeline(
        testee=FakeTestee(),
        grader=SimpleGrader(),
        attack_strategies=[],
        verbose=False,
    )
    new_results, summary = pipeline.run_attack_replay(
        [replay_item],
        expected_eligible_population=2,
        attack_strategy_order=[STRATEGY],
        seed_results=[seed],
    )

    assert len(new_results) == 1
    assert summary.total_samples == 2
    assert summary.eligible_baseline_correct == 2
    assert summary.valid_target_tested == 1
    assert summary.target_safe == 1
    assert summary.invalid_noop == 1
    assert summary.failed_manipulations == 1
    assert summary.attack_coverage == pytest.approx(0.5)
