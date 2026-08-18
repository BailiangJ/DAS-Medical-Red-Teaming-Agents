import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch



from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.models import GenerationConfig
from med_red_team.shared.checkpoints import retire_foreign_checkpoint
from med_red_team.healthbench.data import (
    AttackResult,
    HealthBenchRobustnessResult,
    HealthBenchTestCase,
    RubricGradeResult,
    RubricItem,
    build_healthbench_metadata,
    healthbench_file_identity,
    healthbench_json_sha256,
)
from scripts.healthbench import run_attack, run_attack_replay, run_baseline, run_baseline_split
from scripts.healthbench.checkpoints import (
    merge_healthbench_results,
    validate_healthbench_checkpoint_metadata,
    validate_healthbench_resume_rows,
)


class HealthBenchResumeTests(unittest.TestCase):
    def _resume_path(self):
        handle = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def _resume_file_with_metadata(self, metadata):
        path = Path(self._resume_path())
        path.write_text(json.dumps({"metadata": metadata, "results": []}))
        return path

    def _healthbench_metadata(self, *, phase, is_partial=False):
        config = HealthBenchConfig(
            testee_model="testee",
            grader_model="grader",
            dataset_path="/tmp/current.jsonl",
            max_samples=1,
        )
        return {
            "schema_version": "2.0",
            "axis": "healthbench",
            "phase": phase,
            "config": config.to_dict(),
            "models": {
                "testee": {"model_id": "testee"},
                "grader": {"model_id": "grader"},
                "attack_strategies": {},
                "attack_strategy_order": [],
            },
            "dataset": {
                "path": "/tmp/current.jsonl",
                "sample_limit": 1,
                "filter_single_turn": True,
            },
            "source": {"workflow": "test"},
            "is_partial": is_partial,
            "testee_model": "testee",
            "grader_model": "grader",
            "dataset_path": "/tmp/current.jsonl",
            "filter_single_turn": True,
            "max_samples": 1,
        }

    def _artifact(self, *, phase, data_key, items=None, is_partial=False):
        path = Path(self._resume_path())
        payload = {
            "metadata": self._healthbench_metadata(phase=phase, is_partial=is_partial),
            "summary": {},
            data_key: items or [],
        }
        path.write_text(json.dumps(payload))
        return path

    def _test_case(self):
        return HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("criterion", 1, [])],
            example_tags=[],
        )

    def test_baseline_resume_complete_returns_before_model_construction(self):
        dataset_path = self._resume_path()
        args = SimpleNamespace(resume=self._resume_path(), quiet=True)
        config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            filter_single_turn=True,
        )
        existing = [SimpleNamespace(test_case_id="case-1")]

        with (
            patch.object(run_baseline, "parse_args", return_value=args),
            patch.object(run_baseline, "create_config_from_args", return_value=config),
            patch.object(run_baseline, "validate_resume_file", return_value=(True, "ok")),
            patch.object(run_baseline, "load_results_from_json", return_value=(existing, {})),
            patch.object(run_baseline, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(
                run_baseline,
                "healthbench_effective_dataset_identity",
                return_value={"effective_sha256": "dataset-sha"},
            ),
            patch.object(run_baseline, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_baseline.main(), 0)
            model_pool.assert_not_called()

    def test_attack_resume_complete_returns_before_model_construction(self):
        dataset_path = self._resume_path()
        baseline_path = self._resume_path()
        args = SimpleNamespace(
            resume=self._resume_path(),
            quiet=True,
            baseline_results=baseline_path,
        )
        config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={"cognitive_bias": {}},
            filter_single_turn=True,
            testee_system_prompt="system",
            to_dict=lambda: {"attacker_strategies": {"cognitive_bias": {}}, "testee_config": {}, "grader_config": {}},
        )
        existing = [SimpleNamespace(test_case_id="case-1")]

        with (
            patch.object(run_attack, "parse_args", return_value=args),
            patch.object(run_attack, "create_config_from_args", return_value=config),
            patch.object(run_attack, "validate_resume_file", return_value=(True, "ok")),
            patch.object(run_attack, "load_results_from_json", return_value=(existing, {})),
            patch.object(run_attack, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(
                run_attack,
                "healthbench_effective_dataset_identity",
                return_value={"effective_sha256": "dataset-sha"},
            ),
            patch.object(run_attack, "load_baseline_results", return_value=([], {})),
            patch.object(run_attack, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack.main(), 0)
            model_pool.assert_not_called()

    def test_multi_strategy_config_returns_before_model_construction(self):
        args = SimpleNamespace(resume=None, quiet=True)
        config = SimpleNamespace(
            dataset_path="unused.jsonl",
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={
                "cognitive_bias": {},
                "distraction": {},
            },
            filter_single_turn=True,
        )

        with (
            patch.object(run_attack, "parse_args", return_value=args),
            patch.object(run_attack, "create_config_from_args", return_value=config),
            patch.object(run_attack, "ModelPool") as model_pool,
            patch("builtins.print") as print_mock,
        ):
            self.assertEqual(run_attack.main(), 1)
            model_pool.assert_not_called()
            self.assertIn("Exactly one", print_mock.call_args.args[0])

    def test_load_baseline_results_missing_file_fails_closed(self):
        missing = Path(self._resume_path())
        missing.unlink()

        with self.assertRaisesRegex(FileNotFoundError, "Baseline results file not found"):
            run_attack.load_baseline_results(str(missing), quiet=True)

    def test_load_baseline_results_rejects_grader_mismatch(self):
        payload = {
            "metadata": self._healthbench_metadata(phase="baseline"),
            "summary": {},
            "results": [],
        }
        payload["metadata"]["models"]["grader"]["model_id"] = "other-grader"
        payload["metadata"]["config"]["grader_model"] = "other-grader"
        path = Path(self._resume_path())
        path.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "grader model mismatch"):
            run_attack.load_baseline_results(
                str(path),
                quiet=True,
                expected_testee_model="testee",
                expected_grader_model="grader",
            )

    def test_healthbench_artifact_rejects_internal_grader_model_mismatch(self):
        payload = {
            "metadata": self._healthbench_metadata(phase="baseline"),
            "summary": {},
            "results": [],
        }
        payload["metadata"]["config"]["grader_model"] = "other-grader"
        path = Path(self._resume_path())
        path.write_text(json.dumps(payload))

        with self.assertRaisesRegex(ValueError, "config.grader_model"):
            run_attack.load_baseline_results(str(path), quiet=True)

    def test_attack_runner_forwards_baseline_metadata_to_public_pipeline(self):
        rubric = RubricItem("criterion", 1, [])
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=case.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
        )
        config = HealthBenchConfig(
            testee_model="testee",
            grader_model="grader",
            testee_system_prompt="system",
            max_samples=1,
            filter_single_turn=False,
            attacker_strategies={"cognitive_bias": {"model_id": "attacker"}},
        )
        baseline_metadata = build_healthbench_metadata(
            phase="baseline",
            config=config,
            is_partial=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.jsonl"
            dataset_path.write_text("{}\n")
            baseline_path = Path(directory) / "baseline.json"
            baseline_path.write_text("{}")
            output_path = Path(directory) / "attack.json"
            config.dataset_path = str(dataset_path)
            args = SimpleNamespace(
                baseline_results=str(baseline_path),
                resume=None,
                quiet=True,
                verbose=False,
            )
            pipeline = Mock()
            pipeline.run_attack.return_value = ([], SimpleNamespace())

            with (
                patch.object(run_attack, "parse_args", return_value=args),
                patch.object(run_attack, "create_config_from_args", return_value=config),
                patch.object(run_attack, "load_healthbench_jsonl", return_value=[case]),
                patch.object(
                    run_attack,
                    "load_baseline_results",
                    return_value=([baseline], baseline_metadata),
                ),
                patch.object(run_attack, "ModelPool", return_value=SimpleNamespace()),
                patch.object(run_attack, "Testee", return_value=SimpleNamespace()),
                patch.object(run_attack, "RubricGrader", return_value=SimpleNamespace()),
                patch.object(
                    run_attack,
                    "create_attack_strategies",
                    return_value=[SimpleNamespace(name="cognitive_bias")],
                ),
                patch.object(
                    run_attack,
                    "HealthBenchRobustnessPipeline",
                    return_value=pipeline,
                ),
                patch.object(
                    run_attack,
                    "generate_and_validate_output_path",
                    return_value=output_path,
                ),
            ):
                self.assertEqual(run_attack.main(), 0)

        self.assertIs(
            pipeline.run_attack.call_args.kwargs["baseline_metadata"],
            baseline_metadata,
        )

    def test_impossible_measurement_runner_ignores_unused_baseline_argument(self):
        case = self._test_case()
        config = HealthBenchConfig(
            testee_model="testee",
            grader_model="grader",
            dataset_path=self._resume_path(),
            max_samples=1,
            filter_single_turn=False,
            attacker_strategies={"impossible_measurement": {}},
        )
        args = SimpleNamespace(
            baseline_results="missing-baseline.json",
            resume=None,
            quiet=True,
            verbose=False,
        )
        output_path = Path(self._resume_path())
        output_path.unlink()
        pipeline = Mock()
        pipeline.run_attack.return_value = ([], SimpleNamespace())

        with (
            patch.object(run_attack, "parse_args", return_value=args),
            patch.object(run_attack, "create_config_from_args", return_value=config),
            patch.object(run_attack, "load_healthbench_jsonl", return_value=[case]),
            patch.object(run_attack, "load_baseline_results") as load_baseline,
            patch.object(run_attack, "ModelPool", return_value=SimpleNamespace()),
            patch.object(run_attack, "Testee", return_value=SimpleNamespace()),
            patch.object(run_attack, "RubricGrader", return_value=SimpleNamespace()),
            patch.object(
                run_attack,
                "create_attack_strategies",
                return_value=[SimpleNamespace(name="impossible_measurement")],
            ),
            patch.object(
                run_attack,
                "HealthBenchRobustnessPipeline",
                return_value=pipeline,
            ),
            patch.object(
                run_attack,
                "generate_and_validate_output_path",
                return_value=output_path,
            ),
        ):
            self.assertEqual(run_attack.main(), 0)

        load_baseline.assert_not_called()
        self.assertIsNone(pipeline.run_attack.call_args.kwargs["baseline_results"])
        self.assertIsNone(pipeline.run_attack.call_args.kwargs["baseline_metadata"])

    def test_attack_resume_requires_explicit_baseline_for_baseline_dependent_strategy(self):
        args = SimpleNamespace(resume=self._resume_path(), quiet=True, baseline_results=None)
        config = SimpleNamespace(
            dataset_path="unused.jsonl",
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={"cognitive_bias": {}},
            filter_single_turn=True,
            testee_system_prompt="system",
            to_dict=lambda: {
                "attacker_strategies": {"cognitive_bias": {}},
                "testee_config": {},
                "grader_config": {},
            },
        )

        with (
            patch.object(run_attack, "parse_args", return_value=args),
            patch.object(run_attack, "create_config_from_args", return_value=config),
            patch.object(run_attack, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack.main(), 1)
            model_pool.assert_not_called()

    def test_zero_requested_samples_returns_before_model_construction(self):
        baseline_args = SimpleNamespace(resume=None, quiet=True)
        baseline_config = SimpleNamespace(
            dataset_path="unused.jsonl",
            max_samples=0,
            testee_model="testee",
            grader_model="grader",
            filter_single_turn=True,
        )
        attack_args = SimpleNamespace(resume=None, quiet=True)
        attack_config = SimpleNamespace(
            dataset_path="unused.jsonl",
            max_samples=0,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={"cognitive_bias": {}},
            filter_single_turn=True,
        )

        with (
            patch.object(run_baseline, "parse_args", return_value=baseline_args),
            patch.object(run_baseline, "create_config_from_args", return_value=baseline_config),
            patch.object(run_baseline, "ModelPool") as baseline_pool,
        ):
            self.assertEqual(run_baseline.main(), 0)
            baseline_pool.assert_not_called()

        with (
            patch.object(run_attack, "parse_args", return_value=attack_args),
            patch.object(run_attack, "create_config_from_args", return_value=attack_config),
            patch.object(run_attack, "ModelPool") as attack_pool,
        ):
            self.assertEqual(run_attack.main(), 0)
            attack_pool.assert_not_called()

    def test_filtered_empty_main_runs_avoid_provider_construction(self):
        dataset_path = self._resume_path()
        baseline_args = SimpleNamespace(resume=None, quiet=True, verbose=False)
        baseline_config = HealthBenchConfig(
            dataset_path=dataset_path,
            max_samples=None,
            filter_single_turn=True,
        )
        attack_args = SimpleNamespace(
            resume=None,
            quiet=True,
            verbose=False,
            baseline_results=None,
        )
        attack_config = HealthBenchConfig(
            dataset_path=dataset_path,
            max_samples=None,
            filter_single_turn=True,
            attacker_strategies={"impossible_measurement": {}},
        )

        with (
            patch.object(run_baseline, "parse_args", return_value=baseline_args),
            patch.object(run_baseline, "create_config_from_args", return_value=baseline_config),
            patch.object(run_baseline, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch(
                "med_red_team.healthbench.utils.filter_single_turn_cases",
                return_value=[],
            ),
            patch.object(run_baseline, "ModelPool") as baseline_pool,
        ):
            self.assertEqual(run_baseline.main(), 0)
            baseline_pool.assert_not_called()

        with (
            patch.object(run_attack, "parse_args", return_value=attack_args),
            patch.object(run_attack, "create_config_from_args", return_value=attack_config),
            patch.object(run_attack, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch(
                "med_red_team.healthbench.utils.filter_single_turn_cases",
                return_value=[],
            ),
            patch.object(run_attack, "ModelPool") as attack_pool,
        ):
            self.assertEqual(run_attack.main(), 0)
            attack_pool.assert_not_called()

    def test_completed_partial_baseline_resume_is_finalized(self):
        dataset_path = self._resume_path()
        args = SimpleNamespace(resume=self._resume_path(), quiet=True)
        config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            filter_single_turn=True,
        )
        existing = [SimpleNamespace(test_case_id="case-1")]

        with (
            patch.object(run_baseline, "parse_args", return_value=args),
            patch.object(run_baseline, "create_config_from_args", return_value=config),
            patch.object(run_baseline, "validate_resume_file", return_value=(True, "ok")),
            patch.object(
                run_baseline,
                "load_results_from_json",
                return_value=(existing, {"is_partial": True}),
            ),
            patch.object(run_baseline, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(
                run_baseline,
                "healthbench_effective_dataset_identity",
                return_value={"effective_sha256": "dataset-sha"},
            ),
            patch.object(
                run_baseline.HealthBenchRobustnessPipeline,
                "_compute_summary",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                run_baseline.HealthBenchRobustnessPipeline,
                "save_results",
            ) as save_results,
            patch.object(run_baseline, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_baseline.main(), 0)
            model_pool.assert_not_called()
            save_results.assert_called_once()
            self.assertFalse(save_results.call_args.kwargs["metadata"]["is_partial"])

    def test_completed_partial_attack_resume_is_finalized(self):
        dataset_path = self._resume_path()
        baseline_path = self._resume_path()
        args = SimpleNamespace(
            resume=self._resume_path(),
            quiet=True,
            baseline_results=baseline_path,
        )
        config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={"cognitive_bias": {}},
            filter_single_turn=True,
            testee_system_prompt="system",
            to_dict=lambda: {"attacker_strategies": {"cognitive_bias": {}}, "testee_config": {}, "grader_config": {}},
        )
        existing = [SimpleNamespace(test_case_id="case-1")]

        with (
            patch.object(run_attack, "parse_args", return_value=args),
            patch.object(run_attack, "create_config_from_args", return_value=config),
            patch.object(run_attack, "validate_resume_file", return_value=(True, "ok")),
            patch.object(
                run_attack,
                "load_results_from_json",
                return_value=(existing, {"is_partial": True}),
            ),
            patch.object(run_attack, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(
                run_attack,
                "healthbench_effective_dataset_identity",
                return_value={"effective_sha256": "dataset-sha"},
            ),
            patch.object(run_attack, "load_baseline_results", return_value=([], {})),
            patch.object(
                run_attack.HealthBenchRobustnessPipeline,
                "_compute_summary",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                run_attack.HealthBenchRobustnessPipeline,
                "save_results",
            ) as save_results,
            patch.object(run_attack, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack.main(), 0)
            model_pool.assert_not_called()
            save_results.assert_called_once()
            self.assertFalse(save_results.call_args.kwargs["metadata"]["is_partial"])

    def test_completed_resumes_validate_current_dataset_before_finalizing(self):
        dataset_path = self._resume_path()
        baseline_args = SimpleNamespace(resume=self._resume_path(), quiet=True)
        baseline_config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            filter_single_turn=True,
        )
        existing = [SimpleNamespace(test_case_id="case-1")]

        with (
            patch.object(run_baseline, "parse_args", return_value=baseline_args),
            patch.object(run_baseline, "create_config_from_args", return_value=baseline_config),
            patch.object(
                run_baseline,
                "validate_resume_file",
                side_effect=[(True, "ok"), (False, "Dataset content mismatch")],
            ),
            patch.object(run_baseline, "load_results_from_json", return_value=(existing, {"is_partial": True})),
            patch.object(run_baseline, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(run_baseline.HealthBenchRobustnessPipeline, "save_results") as save_results,
            patch.object(run_baseline, "ModelPool") as model_pool,
        ):
            with self.assertRaises(SystemExit):
                run_baseline.main()
            save_results.assert_not_called()
            model_pool.assert_not_called()

        baseline_path = self._resume_path()
        attack_args = SimpleNamespace(
            resume=self._resume_path(),
            quiet=True,
            baseline_results=baseline_path,
        )
        attack_config = SimpleNamespace(
            dataset_path=dataset_path,
            max_samples=1,
            testee_model="testee",
            grader_model="grader",
            attacker_strategies={"cognitive_bias": {}},
            filter_single_turn=True,
            testee_system_prompt="system",
            to_dict=lambda: {"attacker_strategies": {"cognitive_bias": {}}, "testee_config": {}, "grader_config": {}},
        )
        with (
            patch.object(run_attack, "parse_args", return_value=attack_args),
            patch.object(run_attack, "create_config_from_args", return_value=attack_config),
            patch.object(
                run_attack,
                "validate_resume_file",
                side_effect=[(True, "ok"), (False, "Dataset content mismatch")],
            ),
            patch.object(run_attack, "load_results_from_json", return_value=(existing, {"is_partial": True})),
            patch.object(run_attack, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(run_attack.HealthBenchRobustnessPipeline, "save_results") as save_results,
            patch.object(run_attack, "ModelPool") as model_pool,
        ):
            with self.assertRaises(SystemExit):
                run_attack.main()
            save_results.assert_not_called()
            model_pool.assert_not_called()

    def test_new_healthbench_metadata_has_no_top_level_aliases(self):
        config = HealthBenchConfig(
            testee_model="nested-testee",
            grader_model="nested-grader",
            dataset_path="nested.jsonl",
            attacker_strategies={"distraction": {"model_id": "attacker"}},
        )
        metadata = build_healthbench_metadata(
            phase="attack",
            config=config,
            is_partial=False,
            attack_strategies=["distraction"],
            baseline_results_file="baseline.json",
        )
        for legacy_alias in (
            "testee_model",
            "grader_model",
            "dataset_path",
            "filter_single_turn",
            "attack_strategies",
            "baseline_results_file",
        ):
            self.assertNotIn(legacy_alias, metadata)
        self.assertEqual(metadata["models"]["testee"]["model_id"], "nested-testee")
        self.assertEqual(metadata["models"]["attack_strategy_order"], ["distraction"])
        self.assertEqual(metadata["source"]["baseline_results_file"], "baseline.json")

    def test_nested_resume_metadata_wins_over_conflicting_legacy_aliases(self):
        metadata = self._healthbench_metadata(phase="baseline")
        metadata["models"]["testee"]["model_id"] = "testee"
        metadata["models"]["grader"]["model_id"] = "grader"
        metadata["dataset"]["path"] = "/tmp/current.jsonl"
        metadata["testee_model"] = "wrong-testee"
        metadata["grader_model"] = "wrong-grader"
        metadata["dataset_path"] = "/tmp/wrong.jsonl"
        metadata["filter_single_turn"] = False
        self.assertTrue(
            run_baseline.validate_resume_file(
                self._resume_file_with_metadata(metadata),
                "testee",
                "grader",
                "/tmp/current.jsonl",
                True,
            )[0]
        )
        attack_metadata = self._healthbench_metadata(phase="attack")
        attack_metadata["models"]["attack_strategy_order"] = ["nested-strategy"]
        attack_metadata["models"]["attack_strategies"] = {"nested-strategy": {}}
        attack_metadata["config"]["attacker_strategies"] = {"nested-strategy": {}}
        attack_metadata["attack_strategies"] = ["wrong-strategy"]
        self.assertTrue(
            run_attack.validate_resume_file(
                self._resume_file_with_metadata(attack_metadata),
                "testee",
                "grader",
                ["nested-strategy"],
                {"nested-strategy": {}},
                "/tmp/current.jsonl",
                True,
            )[0]
        )

    def test_resume_validation_accepts_relocated_matching_dataset_sha(self):
        current_sha = "same-effective-content"
        metadata = self._healthbench_metadata(phase="baseline")
        metadata["dataset"]["path"] = "/tmp/old-location.jsonl"
        metadata["dataset"]["effective_sha256"] = current_sha

        valid, message = run_baseline.validate_resume_file(
            self._resume_file_with_metadata(metadata),
            "testee",
            "grader",
            "/tmp/new-location.jsonl",
            True,
            current_sha,
        )

        self.assertTrue(valid, message)

    def test_resume_validation_rejects_changed_dataset_sha(self):
        metadata = self._healthbench_metadata(phase="baseline")
        metadata["dataset"]["effective_sha256"] = "old-content"

        valid, message = run_baseline.validate_resume_file(
            self._resume_file_with_metadata(metadata),
            "testee",
            "grader",
            "/tmp/current.jsonl",
            True,
            "new-content",
        )

        self.assertFalse(valid)
        self.assertIn("Dataset content mismatch", message)

    def test_attack_resume_validation_rejects_changed_baseline_source_sha(self):
        metadata = self._healthbench_metadata(phase="attack")
        metadata["models"]["attack_strategy_order"] = ["cognitive_bias"]
        metadata["models"]["attack_strategies"] = {"cognitive_bias": {}}
        metadata["source"]["source_results_sha256"] = "old-baseline"

        valid, message = run_attack.validate_resume_file(
            self._resume_file_with_metadata(metadata),
            "testee",
            "grader",
            ["cognitive_bias"],
            {"cognitive_bias": {}},
            "/tmp/current.jsonl",
            True,
            None,
            "new-baseline",
        )

        self.assertFalse(valid)
        self.assertIn("Baseline results content mismatch", message)

    def test_healthbench_file_identity_is_dynamic_for_custom_inputs(self):
        path = Path(self._resume_path())
        path.write_text("custom healthbench content", encoding="utf-8")
        identity = healthbench_file_identity(path)

        self.assertEqual(identity["sha256"], healthbench_file_identity(path)["sha256"])
        self.assertEqual(
            healthbench_json_sha256({"path": "not-special"}),
            healthbench_json_sha256({"path": "not-special"}),
        )

    def test_resume_validation_rejects_dataset_and_strategy_order_mismatch(self):
        baseline_resume = self._resume_file_with_metadata({
            "testee_model": "testee",
            "grader_model": "grader",
            "dataset": "/tmp/other.jsonl",
            "filter_single_turn": True,
        })
        valid, message = run_baseline.validate_resume_file(
            baseline_resume,
            "testee",
            "grader",
            "/tmp/current.jsonl",
            True,
        )
        self.assertFalse(valid)
        self.assertIn("Resume file is missing dataset metadata", message)

        attack_resume = self._resume_file_with_metadata({
            "testee_model": "testee",
            "grader_model": "grader",
            "dataset": "/tmp/current.jsonl",
            "filter_single_turn": True,
            "attack_strategies": ["distraction", "cognitive_bias"],
            "config": {
                "attacker_strategies": {
                    "distraction": {"model_id": "model-a"},
                    "cognitive_bias": {"model_id": "model-a"},
                }
            },
        })
        valid, message = run_attack.validate_resume_file(
            attack_resume,
            "testee",
            "grader",
            ["cognitive_bias", "distraction"],
            {
                "cognitive_bias": {"model_id": "model-a"},
                "distraction": {"model_id": "model-a"},
            },
            "/tmp/current.jsonl",
            True,
        )
        self.assertFalse(valid)
        self.assertIn("Attack strategies mismatch", message)

        valid, message = run_attack.validate_resume_file(
            attack_resume,
            "testee",
            "grader",
            ["distraction", "cognitive_bias"],
            {
                "distraction": {"model_id": "model-b"},
                "cognitive_bias": {"model_id": "model-b"},
            },
            "/tmp/current.jsonl",
            True,
        )
        self.assertFalse(valid)
        self.assertIn("Attack strategy config mismatch", message)

    def test_baseline_resume_validation_rejects_generation_and_sample_drift(self):
        metadata = self._healthbench_metadata(phase="baseline")
        metadata["models"]["testee"]["generation_config"] = {"temperature": 0.1}
        metadata["models"]["testee"]["system_prompt"] = "old prompt"
        metadata["models"]["grader"]["generation_config"] = {"temperature": 0.2}
        metadata["config"]["max_samples"] = 1

        valid, message = run_baseline.validate_resume_file(
            self._resume_file_with_metadata(metadata),
            "testee",
            "grader",
            "/tmp/current.jsonl",
            True,
            max_samples=2,
            testee_config={"temperature": 0.0},
            testee_system_prompt="new prompt",
            grader_config={"temperature": 0.0},
        )

        self.assertFalse(valid)
        self.assertIn("Testee config mismatch", message)
        self.assertIn("Testee system prompt mismatch", message)
        self.assertIn("Grader config mismatch", message)
        self.assertIn("Sample limit mismatch", message)

    def test_attack_resume_validation_rejects_generation_setting_drift(self):
        metadata = self._healthbench_metadata(phase="attack")
        metadata["models"]["attack_strategy_order"] = ["cognitive_bias"]
        metadata["models"]["attack_strategies"] = {"cognitive_bias": {}}
        metadata["models"]["testee"]["generation_config"] = {"temperature": 0.1}
        metadata["models"]["testee"]["system_prompt"] = "old prompt"
        metadata["models"]["grader"]["generation_config"] = {"temperature": 0.2}
        metadata["config"]["attacker_strategies"] = {"cognitive_bias": {}}

        valid, message = run_attack.validate_resume_file(
            self._resume_file_with_metadata(metadata),
            "testee",
            "grader",
            ["cognitive_bias"],
            {"cognitive_bias": {}},
            "/tmp/current.jsonl",
            True,
            testee_config={"temperature": 0.0},
            testee_system_prompt="new prompt",
            grader_config={"temperature": 0.0},
            max_samples=2,
        )

        self.assertFalse(valid)
        self.assertIn("Testee config mismatch", message)
        self.assertIn("Testee system prompt mismatch", message)
        self.assertIn("Grader config mismatch", message)
        self.assertIn("Sample limit mismatch", message)

    def test_split_grade_metadata_preserves_testee_system_prompt(self):
        config = HealthBenchConfig(grader_model="grader")
        testee_metadata = self._healthbench_metadata(phase="baseline_collect")
        testee_metadata["models"]["testee"]["system_prompt"] = "collected prompt"
        testee_metadata["config"]["testee_system_prompt"] = "collected prompt"
        with tempfile.TemporaryDirectory() as directory:
            responses_path = Path(directory) / "responses.json"
            responses_path.write_text("{}")
            metadata = run_baseline_split._grade_metadata(
                config,
                testee_metadata,
                responses_path,
                is_partial=False,
                total_cases_evaluated=1,
            )

        self.assertEqual(metadata["models"]["testee"]["system_prompt"], "collected prompt")

    def test_progress_checkpoint_merges_existing_results(self):
        existing = SimpleNamespace(test_case_id="old", sample_number=1)
        new = SimpleNamespace(test_case_id="new", sample_number=1)
        merged = merge_healthbench_results([existing], [new])

        self.assertEqual([result.test_case_id for result in merged], ["old", "new"])
        self.assertEqual([result.sample_number for result in merged], [1, 2])

    def test_progress_checkpoint_merges_dict_rows_and_renumbers(self):
        existing = {"test_case_id": "old", "sample_number": 7}
        new = {"test_case_id": "new", "sample_number": 1}
        merged = merge_healthbench_results([existing], [new])

        self.assertEqual([row["test_case_id"] for row in merged], ["old", "new"])
        self.assertEqual([row["sample_number"] for row in merged], [1, 2])

    def test_split_collect_zero_work_avoids_provider_construction(self):
        config = HealthBenchConfig(
            testee_model="testee",
            dataset_path=self._resume_path(),
            max_samples=0,
            log_dir=str(Path(self._resume_path()).parent),
        )
        args = SimpleNamespace(resume=None, quiet=True)
        output_path = Path(self._resume_path())
        output_path.unlink()

        with (
            patch.object(run_baseline_split, "load_healthbench_jsonl", return_value=[self._test_case()]),
            patch.object(run_baseline_split, "generate_output_path", return_value=output_path),
            patch.object(run_baseline_split, "validate_output_path", return_value=True),
            patch.object(run_baseline_split, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_baseline_split.run_collect_mode(args, config), 0)
            model_pool.assert_not_called()
        self.assertFalse(json.loads(output_path.read_text())["metadata"]["is_partial"])

    def test_replay_collect_zero_work_avoids_provider_construction(self):
        config = HealthBenchConfig(testee_model="testee", max_samples=0)
        args = SimpleNamespace(
            attack_results="attack.json",
            skip_non_applicable=False,
            resume=None,
            quiet=True,
            verbose=False,
        )
        output_path = Path(self._resume_path())
        output_path.unlink()
        attack_metadata = build_healthbench_metadata(
            phase="attack",
            config=HealthBenchConfig(
                testee_model="source",
                attacker_strategies={"impossible_measurement": {}},
            ),
            is_partial=False,
            attack_strategies=["impossible_measurement"],
        )
        attack_result = SimpleNamespace(
            test_case_id="case-1",
            attack_result=SimpleNamespace(
                attack_strategy="impossible_measurement",
                applicable=True,
            ),
            attacked_conversation=[{"role": "user", "content": "modified"}],
        )

        with (
            patch.object(
                run_attack_replay,
                "load_results_from_json",
                return_value=([attack_result], attack_metadata),
            ),
            patch.object(run_attack_replay, "validate_healthbench_artifact_metadata"),
            patch.object(run_attack_replay, "generate_output_path", return_value=output_path),
            patch.object(run_attack_replay, "validate_output_path", return_value=True),
            patch.object(run_attack_replay, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack_replay.run_collect_mode(args, config), 0)
            model_pool.assert_not_called()
        self.assertFalse(json.loads(output_path.read_text())["metadata"]["is_partial"])

    def test_baseline_split_grade_rejects_partial_input_before_model_construction(self):
        partial_responses = self._artifact(
            phase="baseline_collect",
            data_key="responses",
            is_partial=True,
        )
        args = SimpleNamespace(testee_responses=str(partial_responses), resume=None, quiet=True, verbose=False)
        config = HealthBenchConfig(grader_model="grader")

        with patch.object(run_baseline_split, "ModelPool") as model_pool:
            self.assertEqual(run_baseline_split.run_grade_mode(args, config), 1)
            model_pool.assert_not_called()

    def test_attack_replay_collect_rejects_partial_input_before_model_construction(self):
        partial_attack_results = self._artifact(
            phase="attack",
            data_key="results",
            is_partial=True,
        )
        args = SimpleNamespace(
            attack_results=str(partial_attack_results),
            skip_non_applicable=False,
            resume=None,
            quiet=True,
            verbose=False,
        )
        config = HealthBenchConfig(testee_model="testee")

        with patch.object(run_attack_replay, "ModelPool") as model_pool:
            self.assertEqual(run_attack_replay.run_collect_mode(args, config), 1)
            model_pool.assert_not_called()

    def test_attack_replay_grade_rejects_partial_input_before_model_construction(self):
        partial_responses = self._artifact(
            phase="attack_replay_collect",
            data_key="responses",
            is_partial=True,
        )
        args = SimpleNamespace(testee_responses=str(partial_responses), resume=None, quiet=True, verbose=False, output_dir="logs/healthbench", max_samples=None)
        config = HealthBenchConfig(grader_model="grader")

        with patch.object(run_attack_replay, "ModelPool") as model_pool:
            self.assertEqual(run_attack_replay.run_grade_mode(args, config), 1)
            model_pool.assert_not_called()

    def test_attack_replay_collect_rejects_non_impossible_measurement_sources(self):
        args = SimpleNamespace(
            attack_results="source.json",
            skip_non_applicable=False,
            resume=None,
            quiet=True,
            verbose=False,
        )
        config = HealthBenchConfig(testee_model="testee")
        attack_metadata = self._healthbench_metadata(phase="attack")
        attack_metadata["models"]["attack_strategy_order"] = ["cognitive_bias"]
        attack_metadata["models"]["attack_strategies"] = {"cognitive_bias": {}}
        attack_result = SimpleNamespace(
            test_case_id="case-1",
            attack_result=SimpleNamespace(attack_strategy="cognitive_bias", applicable=True),
            attacked_conversation=[{"role": "user", "content": "modified"}],
        )

        with (
            patch.object(run_attack_replay, "load_results_from_json", return_value=([attack_result], attack_metadata)),
            patch.object(run_attack_replay, "validate_healthbench_artifact_metadata"),
            patch.object(run_attack_replay, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack_replay.run_collect_mode(args, config), 1)
            model_pool.assert_not_called()

    def test_attack_replay_grade_rejects_non_impossible_measurement_responses(self):
        args = SimpleNamespace(
            testee_responses="responses.json",
            resume=None,
            quiet=True,
            verbose=False,
            output_dir="logs/healthbench",
            max_samples=None,
        )
        config = HealthBenchConfig(grader_model="grader")
        response_metadata = self._healthbench_metadata(phase="attack_replay_collect")
        response_metadata["models"]["attack_strategy_order"] = ["cognitive_bias"]
        response_metadata["models"]["attack_strategies"] = {"cognitive_bias": {}}

        with (
            patch.object(run_attack_replay, "load_results_from_json", return_value=([], response_metadata)),
            patch.object(run_attack_replay, "validate_healthbench_artifact_metadata"),
            patch.object(run_attack_replay, "ModelPool") as model_pool,
        ):
            self.assertEqual(run_attack_replay.run_grade_mode(args, config), 1)
            model_pool.assert_not_called()

    def test_attack_replay_grade_emits_attacked_only_results(self):
        args = SimpleNamespace(
            testee_responses="responses.json",
            resume=None,
            quiet=True,
            verbose=False,
            output_dir="logs/healthbench",
            max_samples=None,
        )
        config = HealthBenchConfig(grader_model="grader")
        standalone_rubric = RubricItem("standalone impossible measurement", -5, ["safety"])
        response_metadata = self._healthbench_metadata(phase="attack_replay_collect")
        response_metadata["models"]["attack_strategy_order"] = ["impossible_measurement"]
        response_metadata["models"]["attack_strategies"] = {"impossible_measurement": {}}
        response_metadata["source"]["source_testee_model"] = "source-model"
        response = {
            "test_case_id": "case-1",
            "sample_number": 1,
            "original_conversation": [{"role": "user", "content": "question"}],
            "original_rubrics": [standalone_rubric.to_dict()],
            "baseline_completion": "source baseline",
            "baseline_grades": [RubricGradeResult(standalone_rubric, False, "baseline").to_dict()],
            "baseline_score": 1.0,
            "attacked_conversation": [{"role": "user", "content": "modified"}],
            "attacked_rubrics": [standalone_rubric.to_dict()],
            "attack_result": AttackResult(
                test_case_id="case-1",
                attack_strategy="impossible_measurement",
                applicable=True,
                original_conversation=[{"role": "user", "content": "question"}],
                attack_metadata={},
                modified_conversation=[{"role": "user", "content": "modified"}],
            ).to_dict(),
            "completion": "attacked answer",
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "replay.json"
            grader = SimpleNamespace(
                model_id="grader",
                grade=lambda **kwargs: [RubricGradeResult(standalone_rubric, True, "unsafe")],
            )
            with (
                patch.object(run_attack_replay, "load_results_from_json", return_value=([response], response_metadata)),
                patch.object(run_attack_replay, "validate_healthbench_artifact_metadata"),
                patch.object(run_attack_replay, "generate_output_path", return_value=output_path),
                patch.object(run_attack_replay, "validate_output_path", return_value=True),
                patch.object(run_attack_replay, "RubricGrader", return_value=grader),
                patch.object(run_attack_replay, "ModelPool", return_value=SimpleNamespace()),
            ):
                self.assertEqual(run_attack_replay.run_grade_mode(args, config), 0)

            payload = json.loads(output_path.read_text())
            self.assertIsNone(payload["results"][0]["baseline_completion"])
            self.assertEqual(payload["results"][0]["baseline_grades"], [])
            self.assertIsNone(payload["results"][0]["baseline_score"])
            self.assertIsNone(payload["summary"]["avg_score_degradation"])
            self.assertEqual(payload["summary"]["attacks_comparable"], 1)
            self.assertEqual(payload["summary"]["total_rubric_attack_successes"], 1)

    def test_split_config_precedence_supports_explicit_false_and_default_path(self):
        preset = HealthBenchConfig(
            max_samples=7,
            filter_single_turn=True,
            log_dir="preset-logs",
        )
        omitted = SimpleNamespace(
            config="preset.py",
            quiet=True,
            dataset=None,
            testee_model=None,
            grader_model=None,
            max_samples=None,
            filter_single_turn=None,
            output_dir=None,
        )
        explicit = SimpleNamespace(**vars(omitted))
        explicit.max_samples = 0
        explicit.filter_single_turn = False
        explicit.output_dir = "logs/healthbench"

        with patch.object(run_baseline_split, "load_config_from_py", return_value=preset):
            resolved = run_baseline_split.create_config_from_args(omitted)
            overridden = run_baseline_split.create_config_from_args(explicit)

        self.assertEqual(resolved.max_samples, 7)
        self.assertTrue(resolved.filter_single_turn)
        self.assertEqual(resolved.log_dir, "preset-logs")
        self.assertEqual(overridden.max_samples, 0)
        self.assertFalse(overridden.filter_single_turn)
        self.assertEqual(overridden.log_dir, "logs/healthbench")

    def test_replay_temperature_override_preserves_generation_config(self):
        testee_config = GenerationConfig(
            temperature=0.4,
            top_p=0.7,
            top_k=11,
            max_tokens=1234,
            repetition_penalty=1.2,
            stop_sequences=["STOP"],
            use_thinking=True,
            max_thinking_tokens=333,
            reasoning_effort="high",
        )
        grader_config = GenerationConfig(
            temperature=0.3,
            top_p=0.6,
            top_k=9,
            max_tokens=987,
            repetition_penalty=1.1,
            stop_sequences=["DONE"],
            use_thinking=True,
            max_thinking_tokens=222,
            reasoning_effort="low",
        )
        preset = HealthBenchConfig(
            testee_config=testee_config,
            grader_config=grader_config,
            max_samples=8,
            log_dir="preset-logs",
        )
        args = SimpleNamespace(
            config="preset.py",
            quiet=True,
            testee_model=None,
            grader_model=None,
            testee_temperature=0.9,
            testee_system_prompt=None,
            grader_temperature=0.8,
            max_samples=None,
            output_dir=None,
        )

        with patch.object(run_attack_replay, "load_config_from_py", return_value=preset):
            resolved = run_attack_replay.create_config_from_args(args)

        expected_testee = testee_config.to_dict()
        expected_testee["temperature"] = 0.9
        expected_grader = grader_config.to_dict()
        expected_grader["temperature"] = 0.8
        self.assertEqual(resolved.testee_config.to_dict(), expected_testee)
        self.assertEqual(resolved.grader_config.to_dict(), expected_grader)
        self.assertEqual(resolved.max_samples, 8)
        self.assertEqual(resolved.log_dir, "preset-logs")

    def test_checkpoint_validator_uses_nested_contract_and_fails_closed(self):
        config = HealthBenchConfig(
            testee_model="testee",
            grader_model="grader",
            testee_config=GenerationConfig(temperature=0.1),
            grader_config=GenerationConfig(temperature=0.2),
            testee_system_prompt="system",
            max_samples=3,
            filter_single_turn=False,
        )
        metadata = build_healthbench_metadata(
            phase="baseline_collect",
            config=config,
            is_partial=True,
            dataset={
                "path": "dataset.jsonl",
                "sample_limit": 3,
                "filter_single_turn": False,
                "effective_sha256": "dataset-sha",
            },
        )

        validate_healthbench_checkpoint_metadata(
            metadata,
            phase="baseline_collect",
            expected_testee_model="testee",
            expected_testee_config=config.testee_config.to_dict(),
            expected_testee_system_prompt="system",
            expected_dataset_effective_sha256="dataset-sha",
            expected_config_sample_limit=3,
            expected_dataset_sample_limit=3,
            expected_filter_single_turn=False,
        )

        missing_hash = json.loads(json.dumps(metadata))
        del missing_hash["dataset"]["effective_sha256"]
        with self.assertRaisesRegex(ValueError, "dataset content"):
            validate_healthbench_checkpoint_metadata(
                missing_hash,
                phase="baseline_collect",
                expected_dataset_effective_sha256="dataset-sha",
            )
        wrong_phase = json.loads(json.dumps(metadata))
        wrong_phase["phase"] = "baseline"
        with self.assertRaisesRegex(ValueError, "phase"):
            validate_healthbench_checkpoint_metadata(
                wrong_phase,
                phase="baseline_collect",
            )

    def test_checkpoint_rows_reject_duplicates_and_out_of_scope_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_healthbench_resume_rows(
                [
                    {"test_case_id": "case-1"},
                    {"test_case_id": "case-1"},
                ],
                allowed_case_ids=["case-1"],
            )
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_healthbench_resume_rows(
                [{"test_case_id": "other"}],
                allowed_case_ids=["case-1"],
            )

    def test_grade_metadata_keeps_stage_limit_separate_from_source_population(self):
        source_config = HealthBenchConfig(
            testee_model="testee",
            max_samples=10,
            filter_single_turn=True,
        )
        source_metadata = build_healthbench_metadata(
            phase="baseline_collect",
            config=source_config,
            is_partial=False,
            dataset={
                "path": "dataset.jsonl",
                "sample_limit": 10,
                "filter_single_turn": True,
                "effective_sha256": "dataset-sha",
            },
        )
        grade_config = HealthBenchConfig(grader_model="grader", max_samples=4)

        with tempfile.TemporaryDirectory() as directory:
            responses_path = Path(directory) / "responses.json"
            responses_path.write_text("{}")
            metadata = run_baseline_split._grade_metadata(
                grade_config,
                source_metadata,
                responses_path,
                is_partial=False,
                total_cases_evaluated=4,
            )

        self.assertEqual(metadata["config"]["max_samples"], 4)
        self.assertEqual(metadata["dataset"]["sample_limit"], 10)

    def test_replay_metadata_preserves_strategy_and_immediate_lineage(self):
        attack_config = HealthBenchConfig(
            testee_model="source-testee",
            max_samples=10,
            attacker_strategies={
                "impossible_measurement": {
                    "model_id": "attacker",
                    "config": GenerationConfig(temperature=0.6),
                }
            },
        )
        attack_metadata = build_healthbench_metadata(
            phase="attack",
            config=attack_config,
            is_partial=False,
            attack_strategies=["impossible_measurement"],
            dataset={
                "path": "dataset.jsonl",
                "sample_limit": 10,
                "filter_single_turn": True,
                "effective_sha256": "dataset-sha",
            },
        )
        replay_config = HealthBenchConfig(
            testee_model="replay-testee",
            max_samples=6,
        )
        grade_config = HealthBenchConfig(grader_model="grader", max_samples=4)

        with tempfile.TemporaryDirectory() as directory:
            attack_path = Path(directory) / "attack.json"
            attack_path.write_text("{}")
            collect_metadata = run_attack_replay._replay_collect_metadata(
                replay_config,
                attack_metadata,
                attack_path,
                is_partial=False,
                total_collected=6,
                skip_non_applicable=False,
            )
            responses_path = Path(directory) / "responses.json"
            responses_path.write_text("{}")
            grade_metadata = run_attack_replay._replay_grade_metadata(
                grade_config,
                collect_metadata,
                responses_path,
                is_partial=False,
                total_cases_evaluated=4,
            )

        strategy_config = attack_metadata["models"]["attack_strategies"]
        self.assertEqual(
            collect_metadata["models"]["attack_strategies"],
            strategy_config,
        )
        self.assertEqual(
            grade_metadata["models"]["attack_strategies"],
            strategy_config,
        )
        self.assertEqual(grade_metadata["config"]["max_samples"], 4)
        self.assertEqual(grade_metadata["dataset"]["sample_limit"], 10)
        self.assertEqual(
            grade_metadata["source"]["source_results_file"],
            str(responses_path),
        )
        self.assertEqual(
            grade_metadata["source"]["source_phase"],
            "attack_replay_collect",
        )
        self.assertEqual(
            grade_metadata["source"]["upstream_attack"],
            collect_metadata["source"],
        )

    def test_current_nested_baseline_grade_checkpoint_resumes(self):
        source_config = HealthBenchConfig(
            testee_model="testee",
            testee_config=GenerationConfig(temperature=0.1),
            testee_system_prompt="system",
            max_samples=1,
        )
        grade_config = HealthBenchConfig(
            grader_model="grader",
            grader_config=GenerationConfig(temperature=0.2),
            max_samples=1,
        )
        rubric = RubricItem("criterion", 1, [])
        result = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            grade_config.log_dir = directory
            responses_path = Path(directory) / "responses.json"
            source_metadata = build_healthbench_metadata(
                phase="baseline_collect",
                config=source_config,
                is_partial=False,
                dataset={
                    "path": "dataset.jsonl",
                    "sample_limit": 1,
                    "filter_single_turn": True,
                    "effective_sha256": "dataset-sha",
                },
            )
            responses_path.write_text(json.dumps({
                "metadata": source_metadata,
                "summary": {},
                "responses": [{"test_case_id": "case-1"}],
            }))
            resume_path = Path(directory) / "grade.json.inprogress"
            resume_metadata = run_baseline_split._grade_metadata(
                grade_config,
                source_metadata,
                responses_path,
                is_partial=True,
                total_cases_evaluated=1,
            )
            resume_path.write_text(json.dumps({
                "metadata": resume_metadata,
                "summary": {},
                "results": [result.to_dict()],
            }))
            args = SimpleNamespace(
                testee_responses=str(responses_path),
                resume=str(resume_path),
                quiet=True,
                verbose=False,
            )
            with patch.object(run_baseline_split, "ModelPool") as model_pool:
                self.assertEqual(
                    run_baseline_split.run_grade_mode(args, grade_config),
                    0,
                )
                model_pool.assert_not_called()

            completed = [
                path
                for path in Path(directory).glob("*.json")
                if path not in {responses_path, resume_path}
            ]
            self.assertEqual(len(completed), 1)
            self.assertFalse(json.loads(completed[0].read_text())["metadata"]["is_partial"])

    def test_current_nested_replay_grade_checkpoint_finalizes_without_provider(self):
        attack_config = HealthBenchConfig(
            testee_model="source-testee",
            max_samples=1,
            attacker_strategies={"impossible_measurement": {}},
        )
        replay_config = HealthBenchConfig(
            testee_model="replay-testee",
            testee_config=GenerationConfig(temperature=0.1),
            testee_system_prompt="system",
            max_samples=1,
        )
        grade_config = HealthBenchConfig(
            grader_model="grader",
            grader_config=GenerationConfig(temperature=0.2),
            max_samples=1,
        )
        rubric = RubricItem("impossible measurement", -1, [])
        result = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[rubric],
            baseline_completion=None,
            baseline_grades=[],
            baseline_score=None,
            attack_result=AttackResult(
                test_case_id="case-1",
                attack_strategy="impossible_measurement",
                applicable=True,
                original_conversation=[{"role": "user", "content": "question"}],
                attack_metadata={},
                modified_conversation=[{"role": "user", "content": "modified"}],
            ),
            attacked_conversation=[{"role": "user", "content": "modified"}],
            attacked_rubrics=[rubric],
            attacked_completion="answer",
            attacked_grades=[RubricGradeResult(rubric, False, "safe")],
            attacked_score=None,
        )

        with tempfile.TemporaryDirectory() as directory:
            grade_config.log_dir = directory
            attack_path = Path(directory) / "attack.json"
            attack_path.write_text("{}")
            attack_metadata = build_healthbench_metadata(
                phase="attack",
                config=attack_config,
                is_partial=False,
                attack_strategies=["impossible_measurement"],
                dataset={
                    "path": "dataset.jsonl",
                    "sample_limit": 1,
                    "filter_single_turn": True,
                    "effective_sha256": "dataset-sha",
                },
            )
            response_metadata = run_attack_replay._replay_collect_metadata(
                replay_config,
                attack_metadata,
                attack_path,
                is_partial=False,
                total_collected=1,
                skip_non_applicable=False,
            )
            responses_path = Path(directory) / "responses.json"
            responses_path.write_text(json.dumps({
                "metadata": response_metadata,
                "summary": {},
                "responses": [{"test_case_id": "case-1"}],
            }))
            resume_path = Path(directory) / "replay.json.inprogress"
            resume_metadata = run_attack_replay._replay_grade_metadata(
                grade_config,
                response_metadata,
                responses_path,
                is_partial=True,
                total_cases_evaluated=1,
            )
            resume_path.write_text(json.dumps({
                "metadata": resume_metadata,
                "summary": {},
                "results": [result.to_dict()],
            }))
            args = SimpleNamespace(
                testee_responses=str(responses_path),
                resume=str(resume_path),
                quiet=True,
                verbose=False,
            )
            with patch.object(run_attack_replay, "ModelPool") as model_pool:
                self.assertEqual(
                    run_attack_replay.run_grade_mode(args, grade_config),
                    0,
                )
                model_pool.assert_not_called()

            completed = [
                path
                for path in Path(directory).glob("*.json")
                if path not in {attack_path, responses_path, resume_path}
            ]
            self.assertEqual(len(completed), 1)
            self.assertFalse(json.loads(completed[0].read_text())["metadata"]["is_partial"])

    def test_retire_foreign_checkpoint_requires_authoritative_replacement_and_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            resume_path = Path(directory) / "old.json.inprogress"
            output_path = Path(directory) / "new.json"
            resume_path.write_text("old")
            self.assertFalse(
                retire_foreign_checkpoint(resume_path, output_path, enabled=True)
            )
            self.assertTrue(resume_path.exists())
            output_path.write_text("new")
            self.assertFalse(retire_foreign_checkpoint(resume_path, output_path))
            self.assertTrue(resume_path.exists())
            self.assertTrue(
                retire_foreign_checkpoint(resume_path, output_path, enabled=True)
            )
            self.assertFalse(resume_path.exists())


if __name__ == "__main__":
    unittest.main()
