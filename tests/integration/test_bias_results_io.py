import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from med_red_team.bias import (
    BiasConfig,
    BiasGrader,
    BiasPipeline,
    BiasReferenceResult,
    BiasResult,
    BiasTestCase,
)
from med_red_team.bias.data import (
    build_bias_metadata,
    create_attack_summary,
    create_baseline_summary,
    load_baseline_cache,
    reclassify_bias_attack_result,
    reclassify_bias_reference_result,
    save_baseline_cache,
    validate_bias_resume_metadata,
)
from med_red_team.shared.checkpoints import checkpoint_path
from scripts.bias import run_attack, run_baseline


class NoopTestee:
    model_id = "noop-testee"



def make_case():
    return BiasTestCase(
        case_id="case_001",
        clinical_summary="A 45-year-old {patient} presents.",
        vital_signs="BP 120/80",
        patient_self_report="Symptoms for two days.",
        questions=["Question? A: One B: Two C: Three D: Four", None, None, None],
        category="cardiology",
    )



def make_baseline_result(vote_outcome="valid"):
    case = make_case()
    return BiasReferenceResult(
        case_id=case.case_id,
        question_idx=0,
        category=case.category,
        case_text=case.format_full_case(0, patient_placeholder="patient"),
        ref_responses=["A"] * 5 if vote_outcome == "valid" else [""] * 5,
        ref_majority_vote="A" if vote_outcome == "valid" else "",
        ref_vote_entropy=0.0,
        unbiased_model_choice="A: One" if vote_outcome == "valid" else "Invalid vote",
        test_case=case,
        ref_vote_outcome=vote_outcome,
        ref_valid_vote_count=(5 if vote_outcome == "valid" else 0),
        ref_invalid_vote_count=(0 if vote_outcome == "valid" else 5),
        ref_vote_counts=({"A": 5} if vote_outcome == "valid" else {}),
        ref_configured_total_attempts=5,
        ref_required_votes=3,
    )



def make_attack_result(strategy_name):
    return BiasResult(
        sample_number=1,
        test_case_id="case_001_q0",
        case_id="case_001",
        question_idx=0,
        category="cardiology",
        original_case=make_case().format_full_case(0, patient_placeholder="patient"),
        ref_responses=["A"] * 5,
        ref_majority_vote="A",
        ref_vote_entropy=0.0,
        unbiased_model_choice="A: One",
        ref_vote_outcome="valid",
        ref_valid_vote_count=5,
        ref_invalid_vote_count=0,
        ref_vote_counts={"A": 5},
        ref_configured_total_attempts=5,
        ref_required_votes=3,
        attack_strategy=strategy_name,
        manipulated_responses=["A"] * 5,
        manipulated_majority_vote="A",
        manipulated_vote_entropy=0.0,
        manipulated_vote_outcome="valid",
        manipulated_valid_vote_count=5,
        manipulated_invalid_vote_count=0,
        manipulated_vote_counts={"A": 5},
        manipulated_configured_total_attempts=5,
        manipulated_required_votes=3,
        bias_detected=False,
        evaluation_outcome="no_bias",
        comparison_reason="Baseline: A, Attack: A → No bias",
    )


class BiasResultIOTests(unittest.TestCase):
    def make_pipeline(self):
        return BiasPipeline(
            testee=NoopTestee(),
            grader=BiasGrader(verbose=False),
            attack_strategies=[],
            verbose=False,
        )

    def test_save_results_uses_stable_envelope_metadata(self):
        config = BiasConfig(
            testee_model="fake-testee",
            data_file="memory.xlsx",
            log_dir="logs",
            attacker_strategies={},
        )
        result = make_baseline_result()
        summary = create_baseline_summary([result])
        metadata = build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=False,
            dataset={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
                "max_samples": config.max_samples,
                "question_population": 1,
            },
            source={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            self.make_pipeline().save_results(
                results=[result],
                summary=summary,
                output_path=str(path),
                metadata=metadata,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["metadata"]["axis"], "bias")
        self.assertEqual(payload["metadata"]["phase"], "baseline")
        self.assertIsInstance(payload["metadata"]["models"], dict)
        self.assertIsInstance(payload["metadata"]["dataset"], dict)
        self.assertIsInstance(payload["metadata"]["source"], dict)
        self.assertEqual(payload["results"][0]["ref_vote_outcome"], "valid")
        for legacy_alias in (
            "target_model",
            "testee_model",
            "grader_model",
            "dataset_path",
            "attack_strategies",
            "baseline_results_file",
        ):
            self.assertNotIn(legacy_alias, payload["metadata"])

    def test_result_round_trip_preserves_strict_majority_fields(self):
        baseline = BiasReferenceResult.from_dict(make_baseline_result().to_dict())
        attack = BiasResult.from_dict(make_attack_result("cognitive_bias").to_dict())

        self.assertEqual(baseline.ref_configured_total_attempts, 5)
        self.assertEqual(baseline.ref_required_votes, 3)
        self.assertEqual(attack.ref_required_votes, 3)
        self.assertEqual(attack.manipulated_configured_total_attempts, 5)
        self.assertEqual(attack.manipulated_required_votes, 3)

    def test_reclassifies_legacy_plurality_results_under_strict_majority(self):
        baseline = make_baseline_result()
        baseline.ref_responses = ["A", "A", "B", "C", "noise"]
        baseline.ref_majority_vote = "A"
        baseline.ref_vote_outcome = "valid"
        self.assertTrue(
            reclassify_bias_reference_result(
                baseline,
                configured_total_attempts=5,
            )
        )
        self.assertEqual(baseline.ref_vote_outcome, "no_winner")
        self.assertEqual(baseline.ref_majority_vote, "")

        attack = make_attack_result("cognitive_bias")
        attack.manipulated_responses = ["B", "B", "A", "C", "noise"]
        attack.manipulated_majority_vote = "B"
        attack.evaluation_outcome = "bias_detected"
        self.assertTrue(
            reclassify_bias_attack_result(
                attack,
                configured_total_attempts=5,
                question_text=make_case().questions[0],
            )
        )
        self.assertEqual(attack.manipulated_vote_outcome, "no_winner")
        self.assertEqual(attack.evaluation_outcome, "attack_no_winner")
        self.assertTrue(attack.skipped)

    def test_completed_plurality_attack_is_reclassified_before_resume_acceptance(self):
        baseline = make_baseline_result()
        attack = make_attack_result("cognitive_bias")
        attack.manipulated_responses = ["B", "B", "A", "C", "noise"]
        attack.manipulated_majority_vote = "B"
        attack.evaluation_outcome = "bias_detected"
        attack.bias_detected = True
        attack.attack_status = "complete"

        self.assertTrue(
            run_attack._reclassify_resume_attack_result(
                attack,
                baseline_by_key={(baseline.case_id, baseline.question_idx): baseline},
                configured_total_attempts=5,
            )
        )
        self.assertEqual(attack.evaluation_outcome, "attack_no_winner")
        self.assertEqual(attack.manipulated_majority_vote, "")
        self.assertFalse(attack.bias_detected)

    def test_attack_resume_rejects_original_case_drift(self):
        baseline = make_baseline_result()
        attack = make_attack_result("cognitive_bias")
        attack.original_case = "Stale prompt"

        self.assertFalse(
            run_attack._reclassify_resume_attack_result(
                attack,
                baseline_by_key={(baseline.case_id, baseline.question_idx): baseline},
                configured_total_attempts=5,
            )
        )

    def test_incomplete_completed_attack_history_remains_pending(self):
        baseline = make_baseline_result()
        attack = make_attack_result("cognitive_bias")
        attack.manipulated_responses = ["B", "B", "B"]
        attack.attack_status = "complete"

        self.assertFalse(
            run_attack._reclassify_resume_attack_result(
                attack,
                baseline_by_key={(baseline.case_id, baseline.question_idx): baseline},
                configured_total_attempts=5,
            )
        )

    def test_contradictory_terminal_attack_row_remains_pending(self):
        baseline = make_baseline_result()
        attack = make_attack_result("cognitive_bias")
        attack.manipulated_responses = []
        attack.manipulated_majority_vote = "B"
        attack.manipulated_vote_outcome = "valid"
        attack.manipulated_valid_vote_count = 3
        attack.manipulated_vote_counts = {"B": 3}
        attack.manipulated_configured_total_attempts = 0
        attack.manipulated_required_votes = 0
        attack.attack_status = "manipulation_failed"
        attack.attack_failure_category = "not_applicable"
        attack.evaluation_outcome = "manipulation_failed"
        attack.skipped = True
        attack.bias_detected = False

        self.assertFalse(
            run_attack._reclassify_resume_attack_result(
                attack,
                baseline_by_key={(baseline.case_id, baseline.question_idx): baseline},
                configured_total_attempts=5,
            )
        )

    def test_reference_reclassification_rejects_incomplete_history(self):
        baseline = make_baseline_result()
        baseline.ref_responses = ["A"] * 3

        self.assertFalse(
            reclassify_bias_reference_result(
                baseline,
                configured_total_attempts=5,
            )
        )

    def test_attack_summary_reports_coverage_end_to_end_and_combined_union(self):
        shifted = make_attack_result("race_socioeconomic_label")
        shifted.case_id = "case_001"
        shifted.question_idx = 0
        shifted.evaluation_outcome = "bias_detected"
        shifted.bias_detected = True
        shifted.manipulated_majority_vote = "B"

        no_bias = make_attack_result("language_manipulation")
        no_bias.case_id = "case_001"
        no_bias.question_idx = 0

        excluded = make_attack_result("emotion_manipulation")
        excluded.case_id = "case_002"
        excluded.question_idx = 0
        excluded.evaluation_outcome = "attack_excluded"
        excluded.skipped = True
        excluded.attack_status = "not_applicable"

        failed = make_attack_result("cognitive_bias")
        failed.case_id = "case_002"
        failed.question_idx = 0
        failed.evaluation_outcome = "manipulation_failed"
        failed.skipped = True
        failed.attack_status = "manipulation_failed"

        strategies = [
            "race_socioeconomic_label",
            "language_manipulation",
            "emotion_manipulation",
            "cognitive_bias",
        ]
        summary = create_attack_summary(
            [shifted, no_bias, excluded, failed],
            strategies,
            attack_population=8,
            strategy_populations={name: 2 for name in strategies},
            question_population=2,
            baseline_population=3,
            baseline_valid_count=2,
        )

        self.assertAlmostEqual(summary.bias_rate, 0.5)
        self.assertEqual(summary.baseline_population, 3)
        self.assertEqual(summary.baseline_valid_count, 2)
        self.assertAlmostEqual(summary.baseline_valid_coverage, 2 / 3)
        self.assertEqual(summary.applicable_population, 3)
        self.assertAlmostEqual(summary.applicable_coverage, 3 / 8)
        self.assertAlmostEqual(summary.comparable_coverage, 2 / 3)
        self.assertAlmostEqual(summary.end_to_end_bias_rate, 1 / 8)
        self.assertEqual(summary.combined_susceptible_count, 1)
        self.assertAlmostEqual(summary.combined_susceptibility_rate, 1.0)
        self.assertAlmostEqual(summary.combined_end_to_end_rate, 0.5)
        self.assertEqual(
            summary.strategy_summaries["race_socioeconomic_label"]["population"],
            2,
        )
        self.assertAlmostEqual(
            summary.strategy_summaries["emotion_manipulation"]["applicable_coverage"],
            0.0,
        )

    def test_summary_serialization_and_printing_are_phase_aware(self):
        baseline = create_baseline_summary([make_baseline_result()], expected_population=1)
        baseline_data = baseline.to_dict()
        self.assertEqual(baseline_data["phase"], "baseline")
        self.assertNotIn("attack_population", baseline_data)
        self.assertNotIn("bias_detected_count", baseline_data)
        baseline_output = io.StringIO()
        with contextlib.redirect_stdout(baseline_output):
            baseline.print_summary()
        self.assertIn("BIAS BASELINE SUMMARY", baseline_output.getvalue())
        self.assertNotIn("Strategies used", baseline_output.getvalue())

        attack = create_attack_summary(
            [],
            ["cognitive_bias"],
            attack_population=1,
            strategy_populations={"cognitive_bias": 1},
            question_population=1,
            baseline_population=1,
            baseline_valid_count=1,
        )
        attack_data = attack.to_dict()
        self.assertEqual(attack_data["phase"], "attack")
        self.assertEqual(attack_data["comparable_attack_artifact_count"], 0)
        self.assertIn("attack_population", attack_data)

    def test_save_results_rejects_summary_phase_mismatch(self):
        config = BiasConfig(attacker_strategies={})
        metadata = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=True,
            dataset={"source_dataset": {}, "baseline_question_count": 0},
            source={"source_results_file": "baseline.json"},
            attack_strategies=[],
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "summary phase"):
                self.make_pipeline().save_results(
                    results=[],
                    summary=create_baseline_summary([]),
                    output_path=str(Path(directory) / "mismatch.json"),
                    metadata=metadata,
                )

    def test_save_results_rejects_summary_count_mismatch(self):
        config = BiasConfig(attacker_strategies={})
        metadata = build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=False,
            dataset={"data_file": "bias.xlsx", "sheet_name": "Bias", "question_population": 0},
            source={"data_file": "bias.xlsx"},
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "total_samples"):
                self.make_pipeline().save_results(
                    results=[make_baseline_result()],
                    summary=create_baseline_summary([]),
                    output_path=str(Path(directory) / "count-mismatch.json"),
                    metadata=metadata,
                )

    def test_baseline_cache_requires_matching_content_bound_metadata(self):
        result = make_baseline_result()
        metadata = {
            "testee_model": "fake-testee",
            "testee_config": {"temperature": 0.0},
            "testee_system_prompt": "answer with a letter",
            "vote_num_baseline": 5,
            "source": {
                "dataset": {"data_file": "memory.xlsx", "sheet_name": "Bias"},
                "source": {"data_file_sha256": "abc123"},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline-cache.json"
            save_baseline_cache([result], path, metadata=metadata)

            compatible = load_baseline_cache(path, expected_metadata=metadata)
            changed = load_baseline_cache(
                path,
                expected_metadata={
                    **metadata,
                    "source": {
                        **metadata["source"],
                        "source": {"data_file_sha256": "changed"},
                    },
                },
            )

        self.assertEqual(set(compatible), {"case_001_q0"})
        self.assertEqual(changed, {})

    def test_baseline_cache_rejects_same_path_content_drift(self):
        result = make_baseline_result()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "bias.xlsx"
            source_path.write_bytes(b"original")
            identity = run_attack.bias_file_identity(source_path)
            metadata = {
                "testee_model": "fake-testee",
                "testee_config": {},
                "testee_system_prompt": "answer",
                "vote_num_baseline": 5,
                "source": {
                    "dataset": {"data_file": str(source_path)},
                    "source": {
                        "data_file_path": identity["path"],
                        "data_file_sha256": identity["sha256"],
                    },
                },
            }
            cache_path = Path(directory) / "baseline-cache.json"
            save_baseline_cache([result], cache_path, metadata=metadata)
            source_path.write_bytes(b"changed")

            self.assertEqual(
                load_baseline_cache(cache_path, expected_metadata=metadata),
                {},
            )

    def test_legacy_flat_baseline_cache_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy-cache.json"
            path.write_text(
                json.dumps({"case_001_q0": make_baseline_result().to_dict()}),
                encoding="utf-8",
            )
            self.assertEqual(load_baseline_cache(path), {})

    def test_attack_resume_rejects_changed_top_level_attacker_defaults(self):
        config = BiasConfig(
            attacker_model="attacker-a",
            attacker_strategies={"cognitive_bias": {}},
        )
        expected = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=False,
            dataset={"source_dataset": {}, "baseline_question_count": 1},
            source={"source_results_sha256": "abc"},
            attack_strategies=["cognitive_bias"],
        )
        changed = BiasConfig.from_dict(config.to_dict())
        changed.attacker_model = "attacker-b"
        actual = build_bias_metadata(
            phase="attack",
            config=changed,
            is_partial=False,
            dataset=expected["dataset"],
            source=expected["source"],
            attack_strategies=["cognitive_bias"],
        )

        with self.assertRaisesRegex(ValueError, "models|config"):
            validate_bias_resume_metadata(expected, actual, compare_attack=True)

    def test_attack_resume_rejects_changed_prompt_fingerprint(self):
        config = BiasConfig(
            attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
        )
        expected = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=False,
            dataset={"source_dataset": {}, "baseline_question_count": 1},
            source={"source_results_sha256": "abc"},
            attack_strategies=["cognitive_bias"],
        )
        actual = json.loads(json.dumps(expected))
        actual["source"]["attack_prompt_fingerprints"]["cognitive_bias"]["sha256"] = "changed"

        with self.assertRaisesRegex(ValueError, "source"):
            validate_bias_resume_metadata(expected, actual, compare_attack=True)

    def test_attack_resume_ignores_equivalent_baseline_artifact_identity_fields(self):
        config = BiasConfig(
            attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
        )
        expected = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=False,
            dataset={"source_dataset": {}, "baseline_question_count": 1},
            source={
                "source_results_file": "/tmp/baseline-a.json",
                "source_results_path": "/tmp/baseline-a.json",
                "source_results_sha256": "abc",
            },
            attack_strategies=["cognitive_bias"],
            baseline_results_file="/tmp/baseline-a.json",
        )
        actual = json.loads(json.dumps(expected))
        actual["source"].update({
            "source_results_file": "/tmp/baseline-b.json",
            "source_results_path": "/tmp/baseline-b.json",
            "source_results_sha256": "def",
            "baseline_results_file": "/tmp/baseline-b.json",
        })

        validate_bias_resume_metadata(expected, actual, compare_attack=True)

    def test_attack_resume_rejects_changed_source_dataset_and_count(self):
        config = BiasConfig(
            testee_model="fake-testee",
            attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
        )
        expected = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=False,
            dataset={
                "source_dataset": {"data_file": "bias.xlsx", "sheet_name": "Bias"},
                "baseline_question_count": 2,
            },
            source={"source_results_sha256": "abc"},
            attack_strategies=["cognitive_bias"],
        )
        actual = json.loads(json.dumps(expected))
        actual["dataset"]["baseline_question_count"] = 1

        with self.assertRaisesRegex(ValueError, "dataset"):
            validate_bias_resume_metadata(expected, actual, compare_attack=True)

    def test_baseline_resume_ignores_surplus_attacker_model_metadata(self):
        config = BiasConfig(attacker_strategies={"cognitive_bias": {}})
        expected = build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=False,
            dataset={"data_file": "bias.xlsx", "sheet_name": "Bias"},
            source={"data_file": "bias.xlsx"},
        )
        actual = json.loads(json.dumps(expected))
        actual["models"].update({
            "attacker": {"model_id": "legacy-attacker"},
            "attack_strategies": {"cognitive_bias": {"model_id": "legacy-attacker"}},
            "attack_strategy_order": ["cognitive_bias"],
        })

        validate_bias_resume_metadata(expected, actual)

    def test_attack_source_metadata_canonicalizes_path_aliases(self):
        config = BiasConfig(attacker_strategies={"cognitive_bias": {}})
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "baseline.json"
            source.write_text("{}", encoding="utf-8")
            relative = Path(os.path.relpath(source, Path.cwd()))
            first = run_attack._attack_source_metadata(relative, config, {}, {})
            second = run_attack._attack_source_metadata(source.resolve(), config, {}, {})
            self.assertEqual(first["source_results_file"], second["source_results_file"])
            self.assertEqual(first["source_results_path"], second["source_results_path"])
            metadata = build_bias_metadata(
                phase="attack",
                config=config,
                is_partial=True,
                source=first,
                baseline_results_file=str(relative),
                attack_strategies=["cognitive_bias"],
            )
            self.assertEqual(
                metadata["source"]["baseline_results_file"],
                str(source.resolve()),
            )

    def test_resume_validator_rejects_legacy_top_level_aliases(self):
        config = BiasConfig(
            testee_model="nested-testee",
            data_file="memory.xlsx",
            attacker_strategies={"cognitive_bias": {}},
        )
        metadata = build_bias_metadata(
            phase="attack",
            config=config,
            is_partial=True,
            dataset={"data_file": "memory.xlsx", "sheet_name": "Sheet1"},
            source={"source_results_file": "nested-baseline.json"},
            attack_strategies=["cognitive_bias"],
        )
        legacy_metadata = {
            "schema_version": "2.0",
            "evaluation_type": "bias",
            "mode": "attack",
            "config": config.to_dict(),
            "target_model": "nested-testee",
            "data_file": "memory.xlsx",
            "sheet_name": "Sheet1",
            "attack_strategies": ["cognitive_bias"],
        }
        with self.assertRaisesRegex(ValueError, "models"):
            validate_bias_resume_metadata(metadata, legacy_metadata, compare_attack=True)


class BiasRunnerResumeTests(unittest.TestCase):
    def test_baseline_config_failure_returns_controlled_error(self):
        args = argparse.Namespace(
            config="missing.py",
            testee_model=None,
            data_file=None,
            max_samples=None,
            vote_num_baseline=None,
            no_cache=False,
            cache_file=None,
            log_dir=None,
            quiet=True,
            resume=None,
        )
        output = io.StringIO()
        with patch.object(run_baseline, "parse_args", return_value=args), \
             patch.object(run_baseline, "create_config_from_args", side_effect=FileNotFoundError("missing.py")), \
             contextlib.redirect_stdout(output):
            self.assertEqual(run_baseline.main(), 1)
        self.assertIn("[ERROR]", output.getvalue())

    def test_attack_malformed_json_returns_controlled_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.json"
            path.write_text("{not-json", encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(path),
                config=None,
                strategies=None,
                testee_model=None,
                attacker_model=None,
                log_dir=None,
                quiet=True,
                resume=None,
            )
            output = io.StringIO()
            with patch.object(run_attack, "parse_args", return_value=args), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(run_attack.main(), 1)
            self.assertIn("[ERROR]", output.getvalue())

    def _baseline_metadata(self, config, is_partial=False):
        return build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=is_partial,
            dataset={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
                "max_samples": config.max_samples,
                "question_population": 1,
            },
            source={
                "data_file": config.data_file,
                "sheet_name": config.sheet_name,
            },
        )

    def test_baseline_resume_all_complete_finalizes_without_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = BiasConfig(
                testee_model="fake-testee",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                baseline_cache_file=str(tmp / "baseline-cache.json"),
                attacker_strategies={},
            )
            resume_path = tmp / "baseline.json.inprogress"
            resume_payload = {
                "metadata": self._baseline_metadata(config, is_partial=True),
                "summary": {},
                "results": [make_baseline_result().to_dict()],
            }
            resume_path.write_text(json.dumps(resume_payload), encoding="utf-8")
            args = argparse.Namespace(
                config=None,
                testee_model=None,
                data_file=None,
                max_samples=None,
                vote_num_baseline=None,
                no_cache=False,
                cache_file=None,
                log_dir=None,
                quiet=True,
                resume=str(resume_path),
            )

            def forbidden_model_pool():
                raise AssertionError("ModelPool should not be constructed")

            with patch.object(run_baseline, "parse_args", return_value=args), \
                 patch.object(run_baseline, "BASELINE_CONFIG", config), \
                 patch.object(run_baseline, "load_bias_test_cases", return_value=[make_case()]), \
                 patch.object(run_baseline, "generate_and_validate_output_path", return_value=tmp / "final.json"), \
                 patch.object(run_baseline, "ModelPool", forbidden_model_pool):
                self.assertEqual(run_baseline.main(), 0)

            final_payload = json.loads((tmp / "final.json").read_text(encoding="utf-8"))
            self.assertFalse(final_payload["metadata"]["is_partial"])
            self.assertEqual(len(final_payload["results"]), 1)
            cache_payload = json.loads(
                Path(config.baseline_cache_file).read_text(encoding="utf-8")
            )
            self.assertEqual(set(cache_payload["results"]), {"case_001_q0"})
            self.assertTrue(resume_path.exists())

    def test_zero_work_baseline_finalizes_without_loader_or_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = BiasConfig(
                testee_model="fake-testee",
                data_file="missing.xlsx",
                max_samples=0,
                log_dir=str(tmp),
                baseline_cache_file=str(tmp / "zero-cache.json"),
                attacker_strategies={},
            )
            args = argparse.Namespace(
                config=None,
                testee_model=None,
                data_file=None,
                max_samples=None,
                vote_num_baseline=None,
                no_cache=False,
                cache_file=None,
                log_dir=None,
                quiet=True,
                resume=None,
            )

            with patch.object(run_baseline, "parse_args", return_value=args), \
                 patch.object(run_baseline, "BASELINE_CONFIG", config), \
                 patch.object(run_baseline, "load_bias_test_cases", side_effect=AssertionError("loader must not run")), \
                 patch.object(run_baseline, "generate_and_validate_output_path", return_value=tmp / "zero.json"), \
                 patch.object(run_baseline, "ModelPool", side_effect=AssertionError("provider must not run")):
                self.assertEqual(run_baseline.main(), 0)

            payload = json.loads((tmp / "zero.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["results"], [])
            self.assertEqual(payload["summary"]["baseline_population"], 0)
            self.assertEqual(payload["summary"]["baseline_valid_coverage"], 0.0)
            self.assertEqual(payload["metadata"]["dataset"]["question_population"], 0)

    def test_attack_rejects_changed_target_config_before_provider_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            baseline_config = BiasConfig(
                testee_model="fake-testee",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
            )
            source_path = tmp / "baseline.json"
            source_path.write_text(json.dumps({
                "metadata": self._baseline_metadata(baseline_config, is_partial=False),
                "summary": {},
                "results": [make_baseline_result().to_dict()],
            }), encoding="utf-8")
            changed_config = BiasConfig.from_dict(baseline_config.to_dict())
            changed_config.testee_config.temperature = 0.9
            args = argparse.Namespace(
                baseline_results=str(source_path),
                config=None,
                strategies=["cognitive_bias"],
                testee_model=None,
                attacker_model=None,
                log_dir=None,
                quiet=True,
                resume=None,
            )

            with patch.object(run_attack, "parse_args", return_value=args), \
                 patch.object(run_attack, "create_config_from_args", return_value=changed_config), \
                 patch.object(run_attack, "load_bias_test_cases", return_value=[make_case()]), \
                 patch.object(run_attack, "ModelPool", side_effect=AssertionError("provider must not run")):
                self.assertEqual(run_attack.main(), 1)

    def test_all_nonvalid_baselines_finalize_without_provider_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            strategy_names = ["cognitive_bias"]
            config = BiasConfig(
                testee_model="fake-testee",
                data_file="memory.xlsx",
                log_dir=str(tmp / "configured-logs"),
                attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
            )
            baseline_result = make_baseline_result("invalid")
            source_path = tmp / "baseline.json"
            source_path.write_text(json.dumps({
                "metadata": self._baseline_metadata(config, is_partial=False),
                "summary": {},
                "results": [baseline_result.to_dict()],
            }), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(source_path),
                config=None,
                strategies=strategy_names,
                testee_model=None,
                attacker_model=None,
                log_dir=None,
                quiet=True,
                resume=None,
            )
            captured = {}

            def fake_output_path(**kwargs):
                captured["log_dir"] = kwargs["log_dir"]
                return tmp / "attack-final.json"

            with patch.object(run_attack, "parse_args", return_value=args), \
                 patch.object(run_attack, "DEFAULT_CONFIG", config), \
                 patch.object(run_attack, "generate_and_validate_output_path", side_effect=fake_output_path), \
                 patch.object(run_attack, "load_bias_test_cases", return_value=[make_case()]), \
                 patch.object(run_attack, "ModelPool", side_effect=AssertionError("provider must not run")):
                self.assertEqual(run_attack.main(), 0)

            payload = json.loads((tmp / "attack-final.json").read_text(encoding="utf-8"))
            self.assertEqual(captured["log_dir"], config.log_dir)
            self.assertEqual(payload["results"][0]["evaluation_outcome"], "baseline_invalid")
            self.assertEqual(payload["results"][0]["attack_status"], "complete")
            self.assertEqual(
                payload["metadata"]["source"]["source_schema_version"],
                "2.0",
            )

    def test_retryable_attack_result_keeps_only_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            config = BiasConfig(
                testee_model="fake-testee",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
            )
            baseline_result = make_baseline_result()
            source_path = tmp / "baseline.json"
            source_path.write_text(json.dumps({
                "metadata": self._baseline_metadata(config, is_partial=False),
                "summary": {},
                "results": [baseline_result.to_dict()],
            }), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(source_path),
                config=None,
                strategies=["cognitive_bias"],
                testee_model=None,
                attacker_model=None,
                log_dir=None,
                quiet=True,
                resume=None,
            )

            retryable = make_attack_result("cognitive_bias")
            retryable.attack_status = "retryable"
            retryable.evaluation_outcome = "error"
            retryable.manipulated_responses = []
            retryable.manipulated_configured_total_attempts = 0

            def fake_run_attack(self, *args, **kwargs):
                return [retryable], create_attack_summary(
                    [retryable],
                    ["cognitive_bias"],
                    attack_population=1,
                    strategy_populations={"cognitive_bias": 1},
                    question_population=1,
                    baseline_population=1,
                    baseline_valid_count=1,
                )

            final_path = tmp / "attack-final.json"
            with patch.object(run_attack, "parse_args", return_value=args), \
                 patch.object(run_attack, "DEFAULT_CONFIG", config), \
                 patch.object(run_attack, "generate_and_validate_output_path", return_value=final_path), \
                 patch.object(run_attack, "load_bias_test_cases", return_value=[make_case()]), \
                 patch.object(run_attack, "ModelPool", return_value=SimpleNamespace()), \
                 patch.object(run_attack, "Testee", return_value=SimpleNamespace(model_id="fake-testee")), \
                 patch.object(run_attack, "create_attack_strategies_from_config", return_value=[SimpleNamespace(name="cognitive_bias")]), \
                 patch.object(BiasPipeline, "run_attack", fake_run_attack):
                self.assertEqual(run_attack.main(), 1)

            partial_path = checkpoint_path(final_path)
            self.assertTrue(partial_path.exists())
            self.assertFalse(final_path.exists())
            payload = json.loads(partial_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["metadata"]["is_partial"])
            self.assertEqual(payload["results"][0]["attack_status"], "retryable")
            self.assertEqual(payload["summary"]["total_samples"], len(payload["results"]))

    def test_attack_resume_skips_completed_case_question_strategy_triples(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            strategy_names = ["race_socioeconomic_label", "language_manipulation"]
            config = BiasConfig(
                testee_model="fake-testee",
                data_file="memory.xlsx",
                log_dir=str(tmp),
                attacker_strategies={name: {"model_id": "fake-attacker"} for name in strategy_names},
            )
            source_path = tmp / "baseline.json"
            baseline_result = make_baseline_result()
            source_path.write_text(json.dumps({
                "metadata": self._baseline_metadata(config, is_partial=False),
                "summary": {},
                "results": [baseline_result.to_dict()],
            }), encoding="utf-8")

            resume_path = tmp / "attack.json.inprogress"
            attack_metadata = build_bias_metadata(
                phase="attack",
                config=config,
                is_partial=True,
                dataset=run_attack._attack_dataset_metadata(
                    {
                        "data_file": config.data_file,
                        "sheet_name": config.sheet_name,
                        "max_samples": config.max_samples,
                        "question_population": 1,
                    },
                    [baseline_result],
                ),
                source=run_attack._attack_source_metadata(
                    source_path,
                    config,
                    {
                        "data_file": config.data_file,
                        "sheet_name": config.sheet_name,
                        "max_samples": config.max_samples,
                        "question_population": 1,
                    },
                    self._baseline_metadata(config, is_partial=False),
                ),
                attack_strategies=strategy_names,
                baseline_results_file=str(source_path),
            )
            completed = make_attack_result("race_socioeconomic_label")
            retryable = make_attack_result("language_manipulation")
            retryable.attack_status = "retryable"
            retryable.evaluation_outcome = "error"
            retryable.manipulated_responses = []
            retryable.manipulated_configured_total_attempts = 0
            resume_path.write_text(json.dumps({
                "metadata": attack_metadata,
                "summary": {},
                "results": [completed.to_dict(), retryable.to_dict()],
            }), encoding="utf-8")
            args = argparse.Namespace(
                baseline_results=str(source_path),
                config=None,
                strategies=strategy_names,
                testee_model=None,
                attacker_model=None,
                log_dir=None,
                quiet=True,
                resume=str(resume_path),
            )
            captured = {}

            def fake_run_attack(
                self,
                baseline_results,
                strategies,
                progress_callback,
                save_every,
                completed_attack_keys,
                existing_results,
                **summary_kwargs,
            ):
                captured["completed_attack_keys"] = set(completed_attack_keys)
                current = make_attack_result("language_manipulation")
                merged = [
                    result
                    for result in existing_results
                    if result.attack_strategy != current.attack_strategy
                ] + [current]
                return merged, create_attack_summary(
                    merged,
                    strategy_names,
                    **summary_kwargs,
                )

            with patch.object(run_attack, "parse_args", return_value=args), \
                 patch.object(run_attack, "DEFAULT_CONFIG", config), \
                 patch.object(run_attack, "generate_and_validate_output_path", return_value=tmp / "attack-final.json"), \
                 patch.object(run_attack, "load_bias_test_cases", return_value=[make_case()]), \
                 patch.object(run_attack, "ModelPool", return_value=SimpleNamespace()), \
                 patch.object(run_attack, "Testee", return_value=SimpleNamespace(model_id="fake-testee")), \
                 patch.object(run_attack, "create_attack_strategies_from_config", return_value=[SimpleNamespace(name=name) for name in strategy_names]), \
                 patch.object(BiasPipeline, "run_attack", fake_run_attack):
                self.assertEqual(run_attack.main(), 0)

            self.assertEqual(
                captured["completed_attack_keys"],
                {("case_001", 0, "race_socioeconomic_label")},
            )
            final_payload = json.loads((tmp / "attack-final.json").read_text(encoding="utf-8"))
            self.assertFalse(final_payload["metadata"]["is_partial"])
            self.assertEqual(len(final_payload["results"]), 2)
            self.assertEqual(
                {row["attack_strategy"] for row in final_payload["results"]},
                set(strategy_names),
            )
            self.assertTrue(resume_path.exists())


if __name__ == "__main__":
    unittest.main()
