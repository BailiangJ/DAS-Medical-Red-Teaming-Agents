import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch



from med_red_team.privacy.config import PrivacyConfig
from med_red_team.privacy.data import (
    EXPECTED_PHI_IDENTIFIER_KEYS,
    PRIVACY_OUTCOME_SAFE,
    PRIVACY_STATUS_COMPLETE,
    PrivacyResult,
    PrivacyTestCase,
    build_privacy_metadata,
    validate_privacy_resume_metadata,
    validate_privacy_stage_compatibility,
)
from med_red_team.privacy.results_io import create_attack_summary, create_baseline_summary
from med_red_team.privacy.pipeline import PrivacyPipeline
from med_red_team.shared.checkpoints import checkpoint_path, merge_unique
from scripts.privacy import run_attack, run_baseline
from scripts.utils import load_results_from_json


class NoopComponent:
    model_id = "noop"


def full_patient_info(prefix="Sentinel"):
    patient_info = {key: f"{prefix} {key}" for key in EXPECTED_PHI_IDENTIFIER_KEYS}
    patient_info["patient_name"] = f"{prefix} Patient"
    return patient_info


class PrivacyResultIOTests(unittest.TestCase):
    def make_pipeline(self):
        return PrivacyPipeline(
            testee=NoopComponent(),
            grader=NoopComponent(),
            generator=NoopComponent(),
            verbose=False,
        )

    def test_result_round_trip_preserves_evaluated_prompts(self):
        original = PrivacyResult(
            test_case_id="case_001",
            sample_number=1,
            original_prompt="canonical",
            baseline_evaluated_prompt="canonical + suffix",
            baseline_attempts=[
                {
                    "attempt_number": 1,
                    "violation_score": 0.0,
                    "evaluated_prompt": "canonical + suffix",
                }
            ],
            attack_strategy="implicit_disguise",
            modified_prompt="modified canonical",
            attack_evaluated_prompt="modified canonical + suffix",
            attack_violation_score=0.5,
            attack_attempts=[
                {
                    "attempt_number": 1,
                    "violation_score": 0.5,
                    "evaluated_prompt": "modified canonical + suffix",
                }
            ],
        )

        restored = PrivacyResult.from_dict(original.to_dict())

        self.assertEqual(
            restored.baseline_evaluated_prompt,
            "canonical + suffix",
        )
        self.assertEqual(
            restored.attack_evaluated_prompt,
            "modified canonical + suffix",
        )
        self.assertEqual(
            restored.attack_attempts[0]["evaluated_prompt"],
            "modified canonical + suffix",
        )

    def test_result_round_trip_preserves_complete_phi_payload(self):
        original = PrivacyResult(
            test_case_id="case_001",
            sample_number=1,
            original_prompt="Prompt with {GENERATED PHI}",
            phi_generated=True,
            generated_phi_text="SENTINEL PHI",
            patient_info=full_patient_info(),
            baseline_evaluated_prompt="Prompt with SENTINEL PHI",
            attack_strategy="implicit_disguise",
            modified_prompt="Prompt with SENTINEL PHI and disguise",
            attack_evaluated_prompt="Prompt with SENTINEL PHI and disguise + suffix",
            attack_attempts=[
                {"evaluated_prompt": "Prompt with SENTINEL PHI and disguise + suffix"}
            ],
        )

        restored = PrivacyResult.from_dict(original.to_dict())

        self.assertTrue(restored.phi_generated)
        self.assertEqual(restored.generated_phi_text, "SENTINEL PHI")
        self.assertEqual(restored.patient_info, full_patient_info())
        self.assertEqual(restored.attack_evaluated_prompt, original.attack_evaluated_prompt)

    def test_attack_save_and_load_preserves_eligibility(self):
        baseline = PrivacyResult(
            test_case_id="case_001",
            sample_number=1,
            violation_score=0.0,
            baseline_status=PRIVACY_STATUS_COMPLETE,
            baseline_outcome=PRIVACY_OUTCOME_SAFE,
        )
        attack = PrivacyResult(
            test_case_id="case_001",
            sample_number=1,
            violation_score=0.0,
            baseline_status=PRIVACY_STATUS_COMPLETE,
            baseline_outcome=PRIVACY_OUTCOME_SAFE,
            attack_strategy="implicit_disguise",
            attack_violation_score=0.0,
            attack_status=PRIVACY_STATUS_COMPLETE,
            attack_outcome=PRIVACY_OUTCOME_SAFE,
            attack_attempts=[{"violation_score": 0.0}],
        )
        summary = create_attack_summary(
            [baseline],
            [attack],
            ["implicit_disguise"],
        )
        eligibility = {"intersection_safe_case_ids": ["case_001"]}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attack.json"
            self.make_pipeline().save_results(
                results=[attack],
                summary=summary,
                output_path=str(path),
                metadata={"config": {"num_attempts": 1}},
                baseline_results=[baseline],
                eligibility=eligibility,
            )

            baseline_loaded, metadata = load_results_from_json(
                path,
                data_key="baseline_results",
                quiet=True,
                reconstruct_fn=PrivacyResult.from_dict,
            )
            attack_loaded, _ = load_results_from_json(
                path,
                data_key="attack_results",
                quiet=True,
                reconstruct_fn=PrivacyResult.from_dict,
            )
            payload = json.loads(path.read_text())

        self.assertEqual(baseline_loaded[0].test_case_id, "case_001")
        self.assertEqual(attack_loaded[0].attack_strategy, "implicit_disguise")
        self.assertEqual(metadata["config"]["num_attempts"], 1)
        self.assertEqual(metadata["axis"], "privacy")
        self.assertEqual(metadata["phase"], "attack")
        self.assertIsInstance(metadata["models"], dict)
        self.assertIsInstance(metadata["dataset"], dict)
        self.assertIsInstance(metadata["source"], dict)
        for legacy_alias in (
            "evaluation_type",
            "mode",
            "testee_model",
            "testee_system_prompt",
            "testee_system_prompt_mode",
            "grader_model",
            "generator_model",
            "max_samples",
            "num_attempts",
        ):
            self.assertNotIn(legacy_alias, metadata)
        self.assertEqual(payload["eligibility"], eligibility)

    def test_result_round_trip_preserves_typed_status_and_attack_trace(self):
        original = PrivacyResult(
            test_case_id="case_001",
            sample_number=17,
            original_prompt="Prompt",
            violation_score=0.0,
            baseline_status="complete",
            baseline_outcome="safe",
            attack_strategy="implicit_disguise",
            attack_violation_score=None,
            attack_status="retryable",
            attack_outcome="unknown",
            attack_failure_category="attacker_parser_error",
            attack_trace={"steps": [{"strategy": "implicit_disguise"}]},
            skipped=False,
            skip_reason="retry later",
        )

        restored = PrivacyResult.from_dict(original.to_dict())

        self.assertEqual(restored.test_case_id, "case_001")
        self.assertEqual(restored.sample_number, 17)
        self.assertEqual(restored.attack_status, "retryable")
        self.assertEqual(restored.attack_failure_category, "attacker_parser_error")
        self.assertEqual(restored.attack_trace, original.attack_trace)
        self.assertEqual(restored.skip_reason, "retry later")

    def test_attack_source_metadata_binds_result_file_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "baseline.json"
            source.write_text("first", encoding="utf-8")
            first = run_attack._attack_source_metadata(source, {})
            source.write_text("second", encoding="utf-8")
            second = run_attack._attack_source_metadata(source, {})

        self.assertEqual(first["source_results_path"], second["source_results_path"])
        self.assertNotEqual(
            first["source_results_sha256"],
            second["source_results_sha256"],
        )

    def test_combined_stage_uses_combined_default_strategy_config(self):
        source = PrivacyConfig(
            testee_model="testee",
            grader_model="grader",
            generator_model="generator",
            attacker_strategies={"implicit_disguise": {}},
        )
        args = SimpleNamespace(
            config=None,
            quiet=True,
            testee_model=None,
            grader_model=None,
            num_attempts=None,
            grader_max_structured_output_retries=None,
            log_dir=None,
            strategies=None,
        )

        combined = run_attack.create_config_from_args(args, source, "combined")

        self.assertEqual(
            list(combined.attacker_strategies),
            ["combined_disguise"],
        )

    def test_stage_compatibility_allows_only_attack_strategy_changes(self):
        source = PrivacyConfig(
            testee_model="testee",
            grader_model="grader",
            generator_model="generator",
            data_file="privacy.xlsx",
            max_samples=12,
            num_attempts=3,
            attacker_strategies={},
        )
        attack_data = source.to_dict()
        attack_data["attacker_strategies"] = {"implicit_disguise": {}}
        attack = PrivacyConfig.from_dict(attack_data)

        validate_privacy_stage_compatibility(source, attack)

        drifted_data = attack.to_dict()
        drifted_data["grader_config"]["max_tokens"] += 1
        drifted = PrivacyConfig.from_dict(drifted_data)
        with self.assertRaisesRegex(ValueError, "grader_config"):
            validate_privacy_stage_compatibility(source, drifted)

    def test_resume_validator_rejects_legacy_top_level_aliases(self):
        config = PrivacyConfig(
            testee_model="testee",
            grader_model="grader",
            generator_model="generator",
            data_file="memory.xlsx",
            attacker_strategies={"implicit_disguise": {}},
        )
        expected = build_privacy_metadata(
            phase="attack",
            config=config,
            is_partial=True,
            source={"source_results_file": "baseline.json"},
            attack_stage="individual",
            attack_strategies=["implicit_disguise"],
        )
        legacy = {
            "schema_version": "2.0",
            "evaluation_type": "privacy",
            "mode": "attack",
            "config": config.to_dict(),
            "testee_model": "testee",
            "grader_model": "grader",
            "generator_model": "generator",
            "data_file": "memory.xlsx",
            "attack_strategies": ["implicit_disguise"],
            "attack_stage": "individual",
        }
        with self.assertRaisesRegex(ValueError, "models"):
            validate_privacy_resume_metadata(expected, legacy, compare_attack=True)

    def test_checkpoint_keeps_each_case_strategy_pair(self):
        baseline = PrivacyResult(
            test_case_id="case_001",
            sample_number=1,
            violation_score=0.0,
            baseline_status=PRIVACY_STATUS_COMPLETE,
            baseline_outcome=PRIVACY_OUTCOME_SAFE,
        )
        attacks = [
            PrivacyResult(
                test_case_id="case_001",
                sample_number=1,
                baseline_status=PRIVACY_STATUS_COMPLETE,
                baseline_outcome=PRIVACY_OUTCOME_SAFE,
                attack_strategy=strategy,
                attack_violation_score=0.0,
                attack_status=PRIVACY_STATUS_COMPLETE,
                attack_outcome=PRIVACY_OUTCOME_SAFE,
            )
            for strategy in ["implicit_disguise", "well_intention"]
        ]
        pipeline = self.make_pipeline()
        merged = merge_unique(
            [],
            attacks,
            key=lambda result: (result.test_case_id, result.attack_strategy),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"
            partial_path = checkpoint_path(output_path)
            pipeline.save_results(
                results=merged,
                summary=create_attack_summary(
                    [baseline],
                    merged,
                    ["implicit_disguise", "well_intention"],
                ),
                output_path=str(partial_path),
                metadata={"config": {}, "is_partial": True},
                baseline_results=[baseline],
            )
            payload = json.loads(partial_path.read_text())

        self.assertEqual(len(payload["attack_results"]), 2)
        self.assertEqual(
            {result["attack_strategy"] for result in payload["attack_results"]},
            {"implicit_disguise", "well_intention"},
        )


class PrivacyRunnerResumeTests(unittest.TestCase):
    def _baseline_metadata(self, config, is_partial=False):
        return build_privacy_metadata(
            phase="baseline",
            config=config,
            is_partial=is_partial,
            dataset={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
                "prompt_column": config.prompt_column,
                "max_samples": config.max_samples,
            },
            source={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
                "prompt_column": config.prompt_column,
            },
        )

    def test_baseline_resume_all_complete_finalizes_without_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = PrivacyConfig(
                testee_model="fake-testee",
                grader_model="fake-grader",
                generator_model="fake-generator",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                attacker_strategies={},
            )
            resume_path = tmp / "baseline.json.inprogress"
            resume_payload = {
                "metadata": self._baseline_metadata(config, is_partial=True),
                "summary": {},
                "results": [
                    PrivacyResult(
                        test_case_id="case_001",
                        sample_number=1,
                        violation_score=0.0,
                        baseline_status=PRIVACY_STATUS_COMPLETE,
                        baseline_outcome=PRIVACY_OUTCOME_SAFE,
                    ).to_dict()
                ],
            }
            resume_path.write_text(json.dumps(resume_payload), encoding="utf-8")
            args = argparse.Namespace(
                config=None,
                data_file=None,
                sheet_name=None,
                prompt_column=None,
                testee_model=None,
                grader_model=None,
                generator_model=None,
                system_prompt_mode=None,
                max_samples=None,
                num_attempts=None,
                grader_max_structured_output_retries=None,
                log_dir=None,
                output_name="final.json",
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_baseline, "parse_args", return_value=args), \
                 patch.object(run_baseline, "BASELINE_CONFIG", config), \
                 patch.object(
                     run_baseline,
                     "load_privacy_test_cases",
                     return_value=[PrivacyTestCase(
                         case_id="case_001",
                         original_prompt="Prompt",
                         category="test",
                         diagnosis="test",
                     )],
                 ), \
                 patch.object(run_baseline, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_baseline.main(), 0)

            final_payload = json.loads((tmp / "final.json").read_text())
            self.assertFalse(final_payload["metadata"]["is_partial"])
            self.assertEqual(len(final_payload["results"]), 1)
            self.assertTrue(resume_path.exists())

    def test_attack_resume_skips_completed_case_strategy_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            strategy_names = ["implicit_disguise", "well_intention"]
            config = PrivacyConfig(
                testee_model="fake-testee",
                grader_model="fake-grader",
                generator_model="fake-generator",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                num_attempts=1,
                attacker_strategies={name: {} for name in strategy_names},
            )
            source_path = tmp / "baseline.json"
            source_metadata = self._baseline_metadata(config, is_partial=False)
            baseline_result = PrivacyResult(
                test_case_id="case_001",
                sample_number=1,
                original_prompt="Prompt",
                violation_score=0.0,
                baseline_attempts=[{"attempt_number": 1, "violation_score": 0.0, "valid_grading": True}],
                baseline_status=PRIVACY_STATUS_COMPLETE,
                baseline_outcome=PRIVACY_OUTCOME_SAFE,
            )
            source_path.write_text(json.dumps({
                "metadata": source_metadata,
                "summary": {},
                "results": [baseline_result.to_dict()],
            }), encoding="utf-8")

            attack_metadata = build_privacy_metadata(
                phase="attack",
                config=config,
                is_partial=True,
                dataset=run_attack._attack_dataset_metadata("individual", source_metadata, config),
                source=run_attack._attack_source_metadata(source_path, source_metadata),
                attack_stage="individual",
                attack_strategies=strategy_names,
                extra={"source_results_file": str(source_path)},
            )
            resume_path = tmp / "attack.json.inprogress"
            completed = PrivacyResult(
                test_case_id="case_001",
                sample_number=1,
                original_prompt="Prompt",
                violation_score=0.0,
                baseline_status=PRIVACY_STATUS_COMPLETE,
                baseline_outcome=PRIVACY_OUTCOME_SAFE,
                attack_strategy="implicit_disguise",
                attack_violation_score=0.0,
                attack_attempts=[{"attempt_number": 1, "violation_score": 0.0}],
                attack_status=PRIVACY_STATUS_COMPLETE,
                attack_outcome=PRIVACY_OUTCOME_SAFE,
            )
            resume_path.write_text(json.dumps({
                "metadata": attack_metadata,
                "summary": {},
                "baseline_results": [baseline_result.to_dict()],
                "attack_results": [completed.to_dict()],
            }), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(source_path),
                individual_attack_results=None,
                config=None,
                strategies=strategy_names,
                testee_model=None,
                grader_model=None,
                num_attempts=None,
                grader_max_structured_output_retries=None,
                log_dir=None,
                output_name="attack-final.json",
                resume=str(resume_path),
                quiet=True,
            )
            captured = {}

            def fake_run_attack(self, baseline_results, num_attempts, progress_callback, save_every, completed_attack_keys, baseline_expected_attempts):
                captured["completed_attack_keys"] = set(completed_attack_keys)
                return [
                    PrivacyResult(
                        test_case_id="case_001",
                        sample_number=1,
                        original_prompt="Prompt",
                        violation_score=0.0,
                        baseline_status=PRIVACY_STATUS_COMPLETE,
                        baseline_outcome=PRIVACY_OUTCOME_SAFE,
                        attack_strategy="well_intention",
                        attack_violation_score=0.0,
                        attack_attempts=[{"attempt_number": 1, "violation_score": 0.0}],
                        attack_status=PRIVACY_STATUS_COMPLETE,
                        attack_outcome=PRIVACY_OUTCOME_SAFE,
                    )
                ], create_attack_summary(baseline_results, [], strategy_names)

            with patch.object(run_attack, "parse_args", return_value=args), \
                 patch.object(run_attack, "ModelPool", return_value=SimpleNamespace()), \
                 patch.object(run_attack, "Testee", return_value=SimpleNamespace(model_id="fake-testee")), \
                 patch.object(run_attack, "PrivacyGrader", return_value=SimpleNamespace(model_id="fake-grader")), \
                 patch.object(run_attack, "PHIGenerator", return_value=SimpleNamespace(model_id="fake-generator")), \
                 patch.object(
                     run_attack,
                     "create_attack_strategies_from_config",
                     return_value=[SimpleNamespace(name=name) for name in strategy_names],
                 ), \
                 patch.object(PrivacyPipeline, "run_attack", fake_run_attack):
                self.assertEqual(run_attack.main(), 0)

            self.assertEqual(
                captured["completed_attack_keys"],
                {("case_001", "implicit_disguise")},
            )
            final_payload = json.loads((tmp / "attack-final.json").read_text())
            self.assertFalse(final_payload["metadata"]["is_partial"])
            self.assertEqual(len(final_payload["attack_results"]), 2)
            self.assertEqual(
                {row["attack_strategy"] for row in final_payload["attack_results"]},
                set(strategy_names),
            )
            self.assertTrue(resume_path.exists())

    def test_attack_resume_rejects_changed_phi_payload_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            strategy_names = ["implicit_disguise"]
            config = PrivacyConfig(
                testee_model="fake-testee",
                grader_model="fake-grader",
                generator_model="fake-generator",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                num_attempts=1,
                attacker_strategies={"implicit_disguise": {}},
            )
            source_path = tmp / "baseline.json"
            source_metadata = self._baseline_metadata(config, is_partial=False)
            baseline = PrivacyResult(
                test_case_id="case_001",
                sample_number=1,
                original_prompt="Prompt with {GENERATED PHI}",
                phi_generated=True,
                generated_phi_text="SENTINEL PHI",
                patient_info=full_patient_info(),
                violation_score=0.0,
                baseline_attempts=[{"attempt_number": 1, "violation_score": 0.0, "valid_grading": True}],
                baseline_status=PRIVACY_STATUS_COMPLETE,
                baseline_outcome=PRIVACY_OUTCOME_SAFE,
            )
            source_path.write_text(json.dumps({
                "metadata": source_metadata,
                "summary": {},
                "results": [baseline.to_dict()],
            }), encoding="utf-8")

            attack_metadata = build_privacy_metadata(
                phase="attack",
                config=config,
                is_partial=True,
                dataset=run_attack._attack_dataset_metadata("individual", source_metadata, config),
                source=run_attack._attack_source_metadata(source_path, source_metadata),
                attack_stage="individual",
                attack_strategies=strategy_names,
                extra={"source_results_file": str(source_path)},
            )
            resume_path = tmp / "attack.json.inprogress"
            altered = PrivacyResult(
                test_case_id="case_001",
                sample_number=1,
                original_prompt="Prompt with {GENERATED PHI}",
                phi_generated=True,
                generated_phi_text="ALTERED PHI",
                patient_info=full_patient_info(),
                violation_score=0.0,
                baseline_status=PRIVACY_STATUS_COMPLETE,
                baseline_outcome=PRIVACY_OUTCOME_SAFE,
                attack_strategy="implicit_disguise",
                attack_violation_score=0.0,
                attack_attempts=[{"attempt_number": 1, "violation_score": 0.0}],
                attack_status=PRIVACY_STATUS_COMPLETE,
                attack_outcome=PRIVACY_OUTCOME_SAFE,
            )
            resume_path.write_text(json.dumps({
                "metadata": attack_metadata,
                "summary": {},
                "baseline_results": [baseline.to_dict()],
                "attack_results": [altered.to_dict()],
            }), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(source_path),
                individual_attack_results=None,
                config=None,
                strategies=strategy_names,
                testee_model=None,
                grader_model=None,
                num_attempts=None,
                grader_max_structured_output_retries=None,
                log_dir=None,
                output_name="attack-final.json",
                resume=str(resume_path),
                quiet=True,
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_attack, "parse_args", return_value=args), patch.object(
                run_attack, "ModelPool", forbidden_model_pool
            ):
                self.assertEqual(run_attack.main(), 1)


if __name__ == "__main__":
    unittest.main()
