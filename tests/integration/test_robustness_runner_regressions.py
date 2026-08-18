import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from med_red_team.data import TestCase as FrameworkTestCase
from med_red_team.grader import SimpleGrader
from med_red_team.models import ModelGenerationError, ModelResponse
from med_red_team.robustness.attacker import AddNoneOfTheAboveStrategy
from med_red_team.robustness.config import RobustnessConfig
from med_red_team.robustness.data import (
    RobustnessResult,
    build_robustness_metadata,
    robustness_file_identity,
    robustness_resume_signature,
    strategy_order_from_metadata,
    validate_robustness_baseline_metadata,
)
from med_red_team.robustness.pipeline import RobustnessPipeline
from med_red_team.robustness.tool_policy import POLICY_VERSION
from med_red_team.robustness.medqa_loader import load_medqa, sample_to_test_case
from med_red_team.shared.checkpoints import checkpoint_path
from med_red_team.shared.io import read_json
from scripts.robustness import (
    run_attack,
    run_attack_replay,
    run_baseline,
    run_generate_attacks,
    run_orchestrator,
)


class FakeTestee:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc

    def answer(self, prompt):
        if self.exc is not None:
            raise self.exc
        return self.responses.pop(0)


def baseline_result(*, correct: bool = True) -> RobustnessResult:
    return RobustnessResult(
        sample_number=1,
        test_case_id="case-1",
        original_question="Question?",
        original_options={"A": "a", "B": "b"},
        original_correct_answer="A",
        original_response="A" if correct else "B",
        original_correct=correct,
        eligible_baseline_correct=correct,
        status="baseline_correct" if correct else "baseline_incorrect",
    )


def write_baseline(path: Path, *, metadata=None, results=None):
    config = RobustnessConfig(testee_model="fake", attacker_strategies={})
    payload = {
        "metadata": metadata or build_robustness_metadata(
            phase="baseline",
            config=config,
            target_model="fake",
            dataset_info={"path": "memory"},
            source={"dataset_path": "memory"},
            is_partial=False,
        ),
        "summary": {},
        "results": results or [baseline_result().to_dict()],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class RobustnessRunnerRegressionTests(unittest.TestCase):
    def test_strategy_order_prefers_nested_models_metadata(self):
        metadata = build_robustness_metadata(
            phase="attack",
            config=RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            ),
            target_model="fake",
            dataset_info={"path": "memory"},
            source={"dataset_path": "memory"},
            is_partial=False,
            attack_strategies=["add_distraction_sentence"],
        )
        metadata["attack_strategies"] = ["legacy"]
        metadata["strategy_order"] = ["legacy-order"]
        self.assertEqual(strategy_order_from_metadata(metadata), ["add_distraction_sentence"])

    def test_target_validation_requires_nested_v2_model_metadata(self):
        config = RobustnessConfig(testee_model="fake", attacker_strategies={})
        metadata = build_robustness_metadata(
            phase="baseline",
            config=config,
            target_model="fake",
            dataset_info={"path": "memory"},
            source={"dataset_path": "memory"},
            is_partial=False,
        )
        metadata["models"].pop("target")
        metadata["target_model"] = "fake"
        with self.assertRaisesRegex(ValueError, "models.target.model_id"):
            validate_robustness_baseline_metadata(metadata)

    def test_new_robustness_metadata_keeps_linkage_nested(self):
        config = RobustnessConfig(testee_model="fake", attacker_strategies={"add_distraction_sentence": {}})
        metadata = build_robustness_metadata(
            phase="attack",
            config=config,
            target_model="fake",
            dataset_info={"path": "memory"},
            source={"baseline_results_file": "baseline.json"},
            is_partial=False,
            attack_strategies=["add_distraction_sentence"],
        )
        for legacy_alias in ("target_model", "attack_strategies", "baseline_results_file"):
            self.assertNotIn(legacy_alias, metadata)
        self.assertEqual(metadata["models"]["target"]["model_id"], "fake")
        self.assertEqual(metadata["models"]["attack_strategy_order"], ["add_distraction_sentence"])
        self.assertEqual(metadata["source"]["baseline_results_file"], "baseline.json")

    def test_resume_signature_uses_sha_not_path_when_hash_is_present(self):
        config = RobustnessConfig(testee_model="fake", attacker_strategies={})
        first = build_robustness_metadata(
            phase="baseline",
            config=config,
            target_model="fake",
            dataset_info={"path": "/tmp/old.jsonl", "sample_limit": 1},
            source={"dataset_path": "/tmp/old.jsonl", "dataset_sha256": "same-content"},
            is_partial=False,
        )
        relocated = build_robustness_metadata(
            phase="baseline",
            config=config,
            target_model="fake",
            dataset_info={"path": "/tmp/new.jsonl", "sample_limit": 1},
            source={"dataset_path": "/tmp/new.jsonl", "dataset_sha256": "same-content"},
            is_partial=False,
        )

        self.assertEqual(
            robustness_resume_signature(first),
            robustness_resume_signature(relocated),
        )

    def test_target_parse_error_is_skipped_not_fooled_in_fixed_attack(self):
        baseline = RobustnessResult(
            sample_number=1,
            test_case_id="case-1",
            original_question="Question?",
            original_options={"A": "a", "B": "b"},
            original_correct_answer="A",
            original_response="A",
            original_correct=True,
            eligible_baseline_correct=True,
            status="baseline_correct",
        )
        parse_error_response = ModelResponse(
            raw_text="<think>unfinished",
            model_id="fake",
            final_answer="",
            metadata={
                "parse_error": {
                    "error_type": "unclosed_thinking_tag",
                    "error_message": "Unclosed thinking tag detected",
                    "details": {},
                }
            },
        )
        pipeline = RobustnessPipeline(
            testee=FakeTestee([parse_error_response]),
            grader=SimpleGrader(),
            attack_strategies=[AddNoneOfTheAboveStrategy()],
            verbose=False,
        )

        results, summary = pipeline.run_attack([baseline])

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.status, "parser_error")
        self.assertTrue(result.skipped)
        self.assertFalse(result.valid_target_tested)
        self.assertIsNone(result.manipulated_correct)
        self.assertEqual(summary.fooled, 0)
        self.assertEqual(summary.parser_errors, 1)

    def test_operational_model_execution_errors_fail_fast(self):
        case = FrameworkTestCase(
            id="case-1",
            question="Question?",
            options={"A": "a", "B": "b"},
            correct_answer="A",
        )
        pipeline = RobustnessPipeline(
            testee=FakeTestee(exc=ModelGenerationError("provider down")),
            grader=SimpleGrader(),
            attack_strategies=[],
            verbose=False,
        )

        with self.assertRaises(ModelGenerationError):
            pipeline.run_baseline([case])

    def test_attack_runner_rejects_empty_chain_before_model_construction(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            write_baseline(baseline_path)
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                strategies=None,
                testee_model=None,
                max_samples=None,
                log_dir=tmp,
                resume=None,
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack, "ModelPool", forbidden_model_pool
            ):
                self.assertEqual(run_attack.main(), 1)

    def test_fixed_attack_zero_eligible_writes_complete_artifact_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            output_path = root / "out" / "attack.json"
            output_path.parent.mkdir()
            write_baseline(
                baseline_path,
                results=[baseline_result(correct=False).to_dict()],
            )
            stale_checkpoint = checkpoint_path(output_path)
            stale_checkpoint.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                strategies=None,
                testee_model=None,
                max_samples=None,
                log_dir=str(output_path.parent),
                resume=None,
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack,
                "generate_and_validate_output_path",
                return_value=output_path,
            ), patch.object(run_attack, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_attack.main(), 0)

            payload = read_json(output_path)
            self.assertFalse(payload["metadata"]["is_partial"])
            self.assertEqual(payload["results"], [])
            self.assertEqual(payload["summary"]["total_samples"], 0)
            self.assertFalse(stale_checkpoint.exists())

    def test_orchestrator_zero_eligible_writes_complete_artifact_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            output_path = root / "out" / "orchestrator.json"
            output_path.parent.mkdir()
            write_baseline(
                baseline_path,
                results=[baseline_result(correct=False).to_dict()],
            )
            stale_checkpoint = checkpoint_path(output_path)
            stale_checkpoint.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                testee_model=None,
                orchestrator_model="gpt-4o",
                tools_model="gpt-4o",
                max_iterations=5,
                max_planner_attempts=2,
                max_samples=None,
                log_dir=str(output_path.parent),
                resume=None,
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_orchestrator, "parse_args", return_value=args), patch.object(
                run_orchestrator,
                "generate_and_validate_output_path",
                return_value=output_path,
            ), patch.object(run_orchestrator, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_orchestrator.main(), 0)

            payload = read_json(output_path)
            self.assertFalse(payload["metadata"]["is_partial"])
            self.assertEqual(payload["results"], [])
            self.assertEqual(payload["summary"]["total_samples"], 0)
            self.assertFalse(stale_checkpoint.exists())

    def test_baseline_resume_rejects_changed_generation_config_before_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resume_path = root / "baseline.inprogress.json"
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")
            dataset_identity = robustness_file_identity(dataset_path)
            config = RobustnessConfig(
                testee_model="fake",
                dataset_path=str(dataset_path),
                attacker_strategies={},
            )
            metadata = build_robustness_metadata(
                phase="baseline",
                config=config,
                target_model="fake",
                dataset_info={"path": str(dataset_path)},
                source={
                    "dataset_path": dataset_identity["path"],
                    "dataset_sha256": dataset_identity["sha256"],
                },
                is_partial=True,
                evaluation_mode="original",
            )
            metadata["models"]["target"]["generation_config"]["temperature"] = 0.9
            write_baseline(resume_path, metadata=metadata)
            args = argparse.Namespace(
                config=None,
                testee_model=None,
                mode="original",
                dataset=None,
                attacked_dataset=None,
                max_samples=None,
                log_dir=str(root),
                quiet=True,
                resume=str(resume_path),
            )
            cases = [
                FrameworkTestCase(
                    id="case-1",
                    question="Question?",
                    options={"A": "a", "B": "b"},
                    correct_answer="A",
                )
            ]

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_baseline, "parse_args", return_value=args), patch.object(
                run_baseline,
                "create_config_from_args",
                return_value=config,
            ), patch.object(
                run_baseline,
                "load_medqa",
                return_value=(cases, {"path": str(dataset_path)}),
            ), patch.object(run_baseline, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_baseline.main(), 1)

    def test_baseline_all_complete_resume_finalizes_without_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")
            dataset_identity = robustness_file_identity(dataset_path)
            output_path = root / "out" / "baseline.json"
            output_path.parent.mkdir()
            resume_path = checkpoint_path(root / "previous.json")
            config = RobustnessConfig(
                testee_model="fake",
                dataset_path=str(dataset_path),
                attacker_strategies={},
            )
            metadata = build_robustness_metadata(
                phase="baseline",
                config=config,
                target_model="fake",
                dataset_info={"path": str(dataset_path)},
                source={
                    "dataset_path": dataset_identity["path"],
                    "dataset_sha256": dataset_identity["sha256"],
                },
                is_partial=True,
                evaluation_mode="original",
            )
            write_baseline(resume_path, metadata=metadata)
            args = argparse.Namespace(
                config=None,
                testee_model=None,
                mode="original",
                dataset=None,
                attacked_dataset=None,
                max_samples=None,
                log_dir=str(output_path.parent),
                quiet=True,
                resume=str(resume_path),
            )
            cases = [
                FrameworkTestCase(
                    id="case-1",
                    question="Question?",
                    options={"A": "a", "B": "b"},
                    correct_answer="A",
                )
            ]

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_baseline, "parse_args", return_value=args), patch.object(
                run_baseline,
                "create_config_from_args",
                return_value=config,
            ), patch.object(
                run_baseline,
                "load_medqa",
                return_value=(cases, {"path": str(dataset_path)}),
            ), patch.object(
                run_baseline,
                "generate_and_validate_output_path",
                return_value=output_path,
            ), patch.object(run_baseline, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_baseline.main(), 0)

            payload = read_json(output_path)
            self.assertFalse(payload["metadata"]["is_partial"])
            self.assertEqual(
                [result["test_case_id"] for result in payload["results"]],
                ["case-1"],
            )
            self.assertTrue(resume_path.exists())

    def test_baseline_resume_rejects_out_of_scope_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text("{}\n", encoding="utf-8")
            dataset_identity = robustness_file_identity(dataset_path)
            resume_path = checkpoint_path(root / "previous.json")
            config = RobustnessConfig(
                testee_model="fake",
                dataset_path=str(dataset_path),
                attacker_strategies={},
            )
            metadata = build_robustness_metadata(
                phase="baseline",
                config=config,
                target_model="fake",
                dataset_info={"path": str(dataset_path)},
                source={
                    "dataset_path": dataset_identity["path"],
                    "dataset_sha256": dataset_identity["sha256"],
                },
                is_partial=True,
                evaluation_mode="original",
            )
            out_of_scope = baseline_result()
            out_of_scope.test_case_id = "not-in-dataset"
            write_baseline(
                resume_path,
                metadata=metadata,
                results=[out_of_scope.to_dict()],
            )
            args = argparse.Namespace(
                config=None,
                testee_model=None,
                mode="original",
                dataset=None,
                attacked_dataset=None,
                max_samples=None,
                log_dir=str(root),
                quiet=True,
                resume=str(resume_path),
            )
            cases = [
                FrameworkTestCase(
                    id="case-1",
                    question="Question?",
                    options={"A": "a", "B": "b"},
                    correct_answer="A",
                )
            ]

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_baseline, "parse_args", return_value=args), patch.object(
                run_baseline,
                "create_config_from_args",
                return_value=config,
            ), patch.object(
                run_baseline,
                "load_medqa",
                return_value=(cases, {"path": str(dataset_path)}),
            ), patch.object(run_baseline, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_baseline.main(), 1)

    def test_fixed_attack_resume_rejects_changed_strategy_config_before_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            resume_path = root / "attack.inprogress.json"
            write_baseline(baseline_path)
            baseline_identity = robustness_file_identity(baseline_path)
            config = RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            )
            metadata = build_robustness_metadata(
                phase="attack",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={
                    "baseline_results_file": baseline_identity["path"],
                    "baseline_results_sha256": baseline_identity["sha256"],
                },
                is_partial=True,
                attack_strategies=["add_distraction_sentence"],
                baseline_results_file=baseline_identity["path"],
            )
            metadata["models"]["attacker_strategies"]["add_distraction_sentence"] = {
                "changed": True
            }
            write_baseline(resume_path, metadata=metadata)
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                strategies=None,
                testee_model=None,
                max_samples=None,
                log_dir=str(root),
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_attack, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_attack.main(), 1)

    def test_orchestrator_resume_rejects_changed_strategy_config_before_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            resume_path = root / "orchestrator.inprogress.json"
            write_baseline(baseline_path)
            baseline_identity = robustness_file_identity(baseline_path)
            config = RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            )
            metadata = build_robustness_metadata(
                phase="orchestrator_attack",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={
                    "baseline_results_file": baseline_identity["path"],
                    "baseline_results_sha256": baseline_identity["sha256"],
                },
                is_partial=True,
                baseline_results_file=baseline_identity["path"],
                orchestrator_model="gpt-4o",
                tools_model="gpt-4o",
                max_iterations=5,
                max_planner_attempts=2,
                attack_mode="orchestrator_progressive",
                extra={"tool_policy_version": POLICY_VERSION},
            )
            metadata["config"]["attacker_strategies"]["add_distraction_sentence"] = {
                "changed": True
            }
            write_baseline(resume_path, metadata=metadata)
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                testee_model=None,
                orchestrator_model="gpt-4o",
                tools_model="gpt-4o",
                max_iterations=5,
                max_planner_attempts=2,
                max_samples=None,
                log_dir=str(root),
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_orchestrator, "parse_args", return_value=args), patch.object(
                run_orchestrator,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_orchestrator, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_orchestrator.main(), 1)

    def test_fixed_attack_resume_rejects_changed_baseline_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            resume_path = root / "attack.inprogress.json"
            write_baseline(baseline_path)
            baseline_identity = robustness_file_identity(baseline_path)
            config = RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            )
            metadata = build_robustness_metadata(
                phase="attack",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={
                    "baseline_results_file": baseline_identity["path"],
                    "baseline_results_sha256": baseline_identity["sha256"],
                },
                is_partial=True,
                attack_strategies=["add_distraction_sentence"],
                baseline_results_file=baseline_identity["path"],
            )
            write_baseline(resume_path, metadata=metadata)
            changed_baseline = read_json(baseline_path)
            changed_baseline["summary"] = {"changed": True}
            baseline_path.write_text(json.dumps(changed_baseline), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                strategies=None,
                testee_model=None,
                max_samples=None,
                log_dir=str(root),
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_attack, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_attack.main(), 1)

    def test_fixed_attack_resume_rejects_out_of_scope_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            resume_path = root / "attack.inprogress.json"
            write_baseline(baseline_path)
            baseline_identity = robustness_file_identity(baseline_path)
            config = RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            )
            metadata = build_robustness_metadata(
                phase="attack",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={
                    "baseline_results_file": baseline_identity["path"],
                    "baseline_results_sha256": baseline_identity["sha256"],
                },
                is_partial=True,
                attack_strategies=["add_distraction_sentence"],
                baseline_results_file=baseline_identity["path"],
            )
            out_of_scope = baseline_result()
            out_of_scope.test_case_id = "not-in-baseline"
            write_baseline(
                resume_path,
                metadata=metadata,
                results=[out_of_scope.to_dict()],
            )
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                strategies=None,
                testee_model=None,
                max_samples=None,
                log_dir=str(root),
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_attack, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_attack.main(), 1)

    def test_orchestrator_resume_rejects_out_of_scope_result_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            resume_path = root / "orchestrator.inprogress.json"
            write_baseline(baseline_path)
            baseline_identity = robustness_file_identity(baseline_path)
            config = RobustnessConfig(
                testee_model="fake",
                attacker_strategies={"add_distraction_sentence": {}},
            )
            metadata = build_robustness_metadata(
                phase="orchestrator_attack",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={
                    "baseline_results_file": baseline_identity["path"],
                    "baseline_results_sha256": baseline_identity["sha256"],
                },
                is_partial=True,
                baseline_results_file=baseline_identity["path"],
                orchestrator_model="gpt-4o",
                tools_model="gpt-4o",
                max_iterations=5,
                max_planner_attempts=2,
                attack_mode="orchestrator_progressive",
                extra={"tool_policy_version": POLICY_VERSION},
            )
            out_of_scope = baseline_result()
            out_of_scope.test_case_id = "not-in-baseline"
            write_baseline(
                resume_path,
                metadata=metadata,
                results=[out_of_scope.to_dict()],
            )
            args = argparse.Namespace(
                baseline_results=str(baseline_path),
                config=None,
                testee_model=None,
                orchestrator_model="gpt-4o",
                tools_model="gpt-4o",
                max_iterations=5,
                max_planner_attempts=2,
                max_samples=None,
                log_dir=str(root),
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_orchestrator, "parse_args", return_value=args), patch.object(
                run_orchestrator,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_orchestrator, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_orchestrator.main(), 1)

    def test_strict_loader_rejects_partial_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "partial.json"
            config = RobustnessConfig(testee_model="fake", attacker_strategies={})
            metadata = build_robustness_metadata(
                phase="baseline",
                config=config,
                target_model="fake",
                dataset_info={"path": "memory"},
                source={"dataset_path": "memory"},
                is_partial=True,
            )
            write_baseline(baseline_path, metadata=metadata)

            with self.assertRaises(SystemExit):
                run_attack.load_baseline_data(str(baseline_path), quiet=True)

    def test_medqa_zero_limit_returns_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_path = Path(tmp) / "medqa.jsonl"
            dataset_path.write_text(
                '{"question":"Q1","options":{"A":"a"},"answer_idx":"A"}\n'
                '{"question":"Q2","options":{"A":"a"},"answer_idx":"A"}\n',
                encoding="utf-8",
            )
            rows, info = load_medqa(str(dataset_path), limit=0)
            self.assertEqual(rows, [])
            self.assertEqual(info["total_loaded"], 0)
            self.assertEqual(info["limit_applied"], 0)

    def test_medqa_fallback_ids_are_content_stable(self):
        sample = {
            "question": "Which option is correct?",
            "options": {"B": "second", "A": "first"},
            "answer_idx": "A",
        }

        first = sample_to_test_case(sample)
        second = sample_to_test_case(dict(sample))
        changed = sample_to_test_case({**sample, "answer_idx": "B"})

        self.assertEqual(first.id, second.id)
        self.assertRegex(first.id, r"^medqa_[0-9a-f]{12}$")
        self.assertNotEqual(first.id, changed.id)

    def test_attack_eligibility_requires_baseline_correct_status(self):
        from med_red_team.robustness.data import is_baseline_result_eligible_for_attack

        eligible = baseline_result(correct=True)
        stale_status = baseline_result(correct=True)
        stale_status.status = "attacked_correct"
        skipped = baseline_result(correct=True)
        skipped.skipped = True
        incorrect = baseline_result(correct=False)

        self.assertTrue(is_baseline_result_eligible_for_attack(eligible))
        self.assertFalse(is_baseline_result_eligible_for_attack(stale_status))
        self.assertFalse(is_baseline_result_eligible_for_attack(skipped))
        self.assertFalse(is_baseline_result_eligible_for_attack(incorrect))

    def test_library_attack_rejects_empty_chain_when_work_exists(self):
        pipeline = RobustnessPipeline(
            testee=None,
            grader=None,
            attack_strategies=[],
            verbose=False,
        )
        with self.assertRaises(ValueError):
            pipeline.run_attack([baseline_result(correct=True)])

    def test_orchestrator_uses_canonical_attack_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline_path = Path(tmp) / "baseline.json"
            stale = baseline_result(correct=True)
            stale.test_case_id = "stale-attacked-row"
            stale.status = "attacked_correct"
            write_baseline(
                baseline_path,
                results=[baseline_result(correct=True).to_dict(), stale.to_dict()],
            )

            _, _, correct_results, _ = run_orchestrator.load_baseline_data(
                str(baseline_path),
                quiet=True,
            )
            self.assertEqual(
                [result.test_case_id for result in correct_results],
                ["case-1"],
            )

    def test_attack_generation_zero_work_finishes_without_model_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"question":"Q1","options":{"A":"a"},"answer_idx":"A"}\n',
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            config = RobustnessConfig(
                testee_model="fake",
                dataset_path=str(dataset_path),
                max_samples=0,
                attacker_strategies={},
            )
            args = argparse.Namespace(
                config=None,
                strategies=None,
                attacker_model=None,
                dataset=None,
                max_samples=0,
                output_dir=str(output_dir),
                resume=None,
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_generate_attacks, "parse_args", return_value=args), patch.object(
                run_generate_attacks,
                "create_config_from_args",
                return_value=config,
            ), patch.object(run_generate_attacks, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_generate_attacks.main(), 0)

            outputs = list(output_dir.glob("*.json"))
            self.assertEqual(len(outputs), 1)
            payload = read_json(outputs[0])
            self.assertFalse(payload["metadata"]["is_partial"])
            self.assertEqual(payload["metadata"]["source"]["dataset_sha256"], robustness_file_identity(dataset_path)["sha256"])
            self.assertEqual(payload["attacked_test_cases"], [])

    def test_attack_replay_rejects_partial_attack_generation_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.jsonl"
            dataset_path.write_text(
                '{"question":"Q1","options":{"A":"a"},"answer_idx":"A"}\n',
                encoding="utf-8",
            )
            identity = robustness_file_identity(dataset_path)
            baseline_config = RobustnessConfig(
                testee_model="fake",
                dataset_path=str(dataset_path),
                attacker_strategies={},
            )
            baseline_metadata = build_robustness_metadata(
                phase="baseline",
                config=baseline_config,
                target_model="fake",
                dataset_info={"path": identity["path"], "total_loaded": 1},
                source={
                    "dataset_path": identity["path"],
                    "dataset_sha256": identity["sha256"],
                },
                is_partial=False,
                evaluation_mode="original",
            )
            baseline_path = root / "baseline.json"
            write_baseline(
                baseline_path,
                metadata=baseline_metadata,
                results=[baseline_result().to_dict()],
            )

            attack_config = RobustnessConfig(
                testee_model="unused-generator-target",
                dataset_path=str(dataset_path),
                attacker_strategies={"add_none_of_the_above": {}},
            )
            metadata = build_robustness_metadata(
                phase="attack_generation",
                config=attack_config,
                target_model=attack_config.testee_model,
                dataset_info={"path": identity["path"]},
                source={
                    "dataset_path": identity["path"],
                    "dataset_sha256": identity["sha256"],
                },
                is_partial=True,
                attack_strategies=["add_none_of_the_above"],
            )
            attacked_path = root / "attacked.json"
            attacked_path.write_text(
                json.dumps({
                    "metadata": metadata,
                    "attacked_test_cases": [],
                }),
                encoding="utf-8",
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(
                run_attack_replay,
                "ModelPool",
                forbidden_model_pool,
            ):
                self.assertEqual(
                    run_attack_replay.main([
                        "--baseline-results",
                        str(baseline_path),
                        "--attacked-dataset",
                        str(attacked_path),
                        "--quiet",
                    ]),
                    1,
                )


if __name__ == "__main__":
    unittest.main()
