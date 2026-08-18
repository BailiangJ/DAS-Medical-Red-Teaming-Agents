import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from unittest.mock import patch

from configs.paper.bias.attack import CONFIG as SCRIPT_DEFAULT_CONFIG
from configs.paper.bias.baseline import CONFIG as BASELINE_CONFIG
from med_red_team.bias import (
    BiasConfig,
    BiasReferenceResult,
    BiasTestCase,
    load_bias_test_cases,
)
from configs.examples.bias.comprehensive import CONFIG as COMPREHENSIVE_CONFIG
from configs.examples.bias.default import CONFIG as EXAMPLE_DEFAULT_CONFIG
from configs.examples.bias.dev import CONFIG as DEV_CONFIG
from configs.examples.bias.fast import CONFIG as FAST_CONFIG
from med_red_team.bias.data import build_bias_metadata
from med_red_team.bias.grader import BiasGrader
from med_red_team.bias.pipeline import BiasPipeline
from med_red_team.models import GenerationConfig
from scripts.bias.run_attack import create_config_from_args, load_baseline_data
from scripts.bias.run_baseline import create_config_from_args as create_baseline_config_from_args
from scripts.utils import create_attack_strategies_from_config


class FakeTestee:
    model_id = "fake-testee"

    def answer(self, prompt, system_prompt=None, config=None):
        raise AssertionError("This test must not call a model")



def make_case():
    return BiasTestCase(
        case_id="case-1",
        clinical_summary="A 45-year-old {patient} presents.",
        vital_signs="BP 120/80",
        patient_self_report="Symptoms for two days.",
        questions=["Question? A: One B: Two C: Three D: Four", None, None, None],
        category="cardiology",
    )



def make_result(include_test_case=True, vote_outcome="valid"):
    case = make_case()
    return BiasReferenceResult(
        case_id=case.case_id,
        question_idx=0,
        category=case.category,
        case_text=case.format_full_case(0, patient_placeholder="patient"),
        ref_responses=["A"] * 5,
        ref_majority_vote="A" if vote_outcome == "valid" else "",
        ref_vote_entropy=0.0,
        unbiased_model_choice="A: One" if vote_outcome == "valid" else "Invalid vote",
        test_case=case if include_test_case else None,
        ref_vote_outcome=vote_outcome,
        ref_valid_vote_count=(5 if vote_outcome == "valid" else 0),
        ref_invalid_vote_count=(0 if vote_outcome == "valid" else 5),
        ref_vote_counts=({"A": 5} if vote_outcome == "valid" else {}),
    )



def make_attack_args(**overrides):
    values = {
        "config": None,
        "strategies": None,
        "attacker_model": None,
        "testee_model": None,
        "log_dir": None,
        "quiet": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)



def make_baseline_args(**overrides):
    values = {
        "config": None,
        "testee_model": None,
        "data_file": None,
        "max_samples": None,
        "vote_num_baseline": None,
        "no_cache": False,
        "cache_file": None,
        "log_dir": None,
        "quiet": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BiasVoteDefaultsTests(unittest.TestCase):
    def test_normal_defaults_use_five_votes(self):
        configs = [
            BiasConfig(),
            EXAMPLE_DEFAULT_CONFIG,
            COMPREHENSIVE_CONFIG,
            BASELINE_CONFIG,
            SCRIPT_DEFAULT_CONFIG,
        ]

        for config in configs:
            with self.subTest(config=repr(config)):
                self.assertEqual(config.vote_num_baseline, 5)
                self.assertEqual(config.vote_num_attack, 5)

    def test_fast_and_dev_presets_use_minimum_strict_panel(self):
        self.assertEqual((FAST_CONFIG.vote_num_baseline, FAST_CONFIG.vote_num_attack), (5, 5))
        self.assertEqual((DEV_CONFIG.vote_num_baseline, DEV_CONFIG.vote_num_attack), (5, 5))
        self.assertEqual(FAST_CONFIG.testee_model, "gpt-4.1-mini")
        self.assertEqual(FAST_CONFIG.attacker_model, "gpt-4.1-mini")

    def test_presets_live_in_configs_not_source_package(self):
        import med_red_team.bias.config as bias_config_module

        for name in ("DEFAULT_CONFIG", "FAST_CONFIG", "DEV_CONFIG", "COMPREHENSIVE_CONFIG"):
            self.assertFalse(hasattr(bias_config_module, name))

        for config in (EXAMPLE_DEFAULT_CONFIG, FAST_CONFIG, DEV_CONFIG, COMPREHENSIVE_CONFIG):
            self.assertIsInstance(config, BiasConfig)

    def test_pipeline_defaults_use_five_votes(self):
        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[],
            verbose=False,
        )
        self.assertEqual(pipeline.vote_num_baseline, 5)
        self.assertEqual(pipeline.vote_num_attack, 5)


class BiasVoteValidationTests(unittest.TestCase):
    def test_accepts_arbitrary_odd_vote_panels(self):
        config = BiasConfig(vote_num_baseline=7, vote_num_attack=9)
        restored = BiasConfig.from_dict(config.to_dict())

        self.assertEqual(restored.vote_num_baseline, 7)
        self.assertEqual(restored.vote_num_attack, 9)

    def test_rejects_invalid_production_vote_panels(self):
        for value in (1, 3, 4, 6, True, 5.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    BiasConfig(vote_num_baseline=value)
                with self.assertRaises(ValueError):
                    BiasConfig(vote_num_attack=value)


class BiasStrategyResolutionTests(unittest.TestCase):
    def test_empty_baseline_config_resolves_standard_four_strategies(self):
        baseline_config = BiasConfig(
            testee_model="sentinel-model",
            attacker_strategies={},
        )

        resolved = create_config_from_args(
            make_attack_args(),
            baseline_config,
            quiet=True,
        )

        self.assertEqual(resolved.testee_model, "sentinel-model")
        self.assertEqual(
            set(resolved.attacker_strategies),
            {
                "race_socioeconomic_label",
                "language_manipulation",
                "emotion_manipulation",
                "cognitive_bias",
            },
        )

    def test_explicit_strategy_uses_standard_fallback(self):
        baseline_config = BiasConfig(attacker_strategies={})

        resolved = create_config_from_args(
            make_attack_args(strategies=["emotion_manipulation"]),
            baseline_config,
            quiet=True,
        )

        self.assertEqual(set(resolved.attacker_strategies), {"emotion_manipulation"})

    def test_attacker_model_only_updates_seeded_standard_strategies(self):
        baseline_config = BiasConfig(attacker_strategies={})

        resolved = create_config_from_args(
            make_attack_args(attacker_model="sentinel-attacker"),
            baseline_config,
            quiet=True,
        )

        self.assertEqual(len(resolved.attacker_strategies), 4)
        self.assertEqual(resolved.attacker_model, "sentinel-attacker")
        self.assertTrue(all(
            kwargs["model_id"] == "sentinel-attacker"
            for kwargs in resolved.attacker_strategies.values()
        ))

    def test_registry_forwards_quiet_flag_to_bias_strategies(self):
        class FakeModel:
            def generate(self, **kwargs):
                raise AssertionError("model should not be called")

        class FakePool:
            def get_model(self, model_id):
                return FakeModel()

        config = BiasConfig(
            attacker_strategies={"cognitive_bias": {"model_id": "fake-attacker"}},
        )
        strategy = create_attack_strategies_from_config(
            config,
            FakePool(),
            verbose=False,
        )[0]

        self.assertFalse(strategy.verbose)

    def test_strategy_construction_uses_top_level_defaults_and_preserves_overrides(self):
        default_config = GenerationConfig(temperature=0.2, max_tokens=321)
        override_config = GenerationConfig(temperature=0.7, max_tokens=654)
        config = BiasConfig(
            attacker_model="default-attacker",
            attacker_config=default_config,
            attacker_strategies={
                "language_manipulation": {},
                "cognitive_bias": {
                    "model_id": "override-attacker",
                    "config": override_config,
                },
            },
        )
        captured = []

        def strategy_getter(**kwargs):
            captured.append(kwargs)
            return kwargs

        create_attack_strategies_from_config(
            config,
            model_pool=object(),
            strategy_getter=strategy_getter,
            verbose=False,
        )

        by_name = {item["strategy_name"]: item for item in captured}
        self.assertEqual(
            by_name["language_manipulation"]["model_id"],
            "default-attacker",
        )
        self.assertIs(by_name["language_manipulation"]["config"], default_config)
        self.assertFalse(by_name["language_manipulation"]["verbose"])
        self.assertEqual(
            by_name["cognitive_bias"]["model_id"],
            "override-attacker",
        )
        self.assertIs(by_name["cognitive_bias"]["config"], override_config)

    def test_explicit_empty_attack_config_fails(self):
        config_path = Path(__file__).parents[2] / "configs/paper/bias/baseline.py"

        with self.assertRaisesRegex(ValueError, "No Bias attack strategies configured"):
            create_config_from_args(
                make_attack_args(config=str(config_path)),
                BiasConfig(),
                quiet=True,
            )

    def test_explicit_attack_config_still_applies_cli_overrides(self):
        config_path = Path(__file__).parents[2] / "configs/paper/bias/attack.py"

        resolved = create_config_from_args(
            make_attack_args(config=str(config_path), attacker_model="override-attacker"),
            BiasConfig(),
            quiet=True,
        )

        self.assertEqual(resolved.attacker_model, "override-attacker")
        self.assertTrue(all(
            kwargs["model_id"] == "override-attacker"
            for kwargs in resolved.attacker_strategies.values()
        ))


class BiasBaselineConfigTests(unittest.TestCase):
    def test_zero_max_samples_is_valid(self):
        config = BiasConfig(max_samples=0)
        self.assertEqual(config.max_samples, 0)

    def test_zero_subject_loader_does_not_read_workbook(self):
        with patch(
            "med_red_team.bias.data.pd.read_excel",
            side_effect=AssertionError("workbook must not be read"),
        ):
            self.assertEqual(load_bias_test_cases("missing.xlsx", n_subjects=0), [])

    def test_loader_rejects_invalid_limits_before_reading_workbook(self):
        with patch(
            "med_red_team.bias.data.pd.read_excel",
            side_effect=AssertionError("workbook must not be read"),
        ):
            for value in (-1, True, 1.5):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        load_bias_test_cases("missing.xlsx", n_subjects=value)
            with self.assertRaises(ValueError):
                load_bias_test_cases("missing.xlsx", loading_type="invalid")

    def test_loader_rejects_duplicate_case_ids_before_sampling(self):
        frame = pd.DataFrame({
            "case id": ["case-1", "case-1"],
            "patient self-report": ["report", "report"],
            "vital sign": ["normal", "normal"],
            "clinical summary": ["summary", "summary"],
            "question 1": ["Question? A: One B: Two", "Question? A: One B: Two"],
            "question 2": [float("nan"), float("nan")],
            "question 3": [float("nan"), float("nan")],
            "question 4": [float("nan"), float("nan")],
            "category": ["general", "general"],
        })
        with patch("med_red_team.bias.data.pd.read_excel", return_value=frame):
            with self.assertRaisesRegex(ValueError, "duplicate case IDs"):
                load_bias_test_cases("memory.xlsx", n_subjects=1)

    def test_loader_normalizes_required_fields_and_category(self):
        frame = pd.DataFrame({
            "case id": [" case-1 "],
            "patient self-report": [float("nan")],
            "vital sign": [" BP 120/80 "],
            "clinical summary": [" A 45-year-old {patient} presents. "],
            "question 1": [" Question? A: One B: Two C: Three D: Four "],
            "question 2": [float("nan")],
            "question 3": [float("nan")],
            "question 4": [float("nan")],
            "category": [" Psychiatric "],
        })
        with patch("med_red_team.bias.data.pd.read_excel", return_value=frame):
            [case] = load_bias_test_cases("memory.xlsx", n_subjects=1)

        self.assertEqual(case.case_id, "case-1")
        self.assertEqual(case.clinical_summary, "A 45-year-old {patient} presents.")
        self.assertEqual(case.vital_signs, "BP 120/80")
        self.assertEqual(case.patient_self_report, "")
        self.assertEqual(case.questions[0], "Question? A: One B: Two C: Three D: Four")
        self.assertEqual(case.category, "Psychiatric")

    def test_loader_rejects_blank_required_fields_and_questionless_rows(self):
        base = {
            "case id": ["case-1"],
            "patient self-report": ["report"],
            "vital sign": ["normal"],
            "clinical summary": ["summary"],
            "question 1": ["Question? A: One B: Two"],
            "question 2": [float("nan")],
            "question 3": [float("nan")],
            "question 4": [float("nan")],
            "category": ["general"],
        }
        mutations = {
            "clinical summary": ("clinical summary", [float("nan")], "clinical summary"),
            "vital sign": ("vital sign", ["   "], "vital sign"),
            "category": ("category", [float("nan")], "category"),
            "questions": ("question 1", ["   "], "nonblank questions"),
        }

        for name, (field, value, pattern) in mutations.items():
            with self.subTest(name=name):
                frame = pd.DataFrame({**base, field: value})
                with patch("med_red_team.bias.data.pd.read_excel", return_value=frame):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_bias_test_cases("memory.xlsx", n_subjects=1)

    def test_explicit_baseline_config_still_applies_cli_overrides(self):
        config_path = Path(__file__).parents[2] / "configs/paper/bias/baseline.py"

        resolved = create_baseline_config_from_args(
            make_baseline_args(config=str(config_path), testee_model="override-model", max_samples=7),
        )

        self.assertEqual(resolved.testee_model, "override-model")
        self.assertEqual(resolved.max_samples, 7)


class BiasBaselineMetadataTests(unittest.TestCase):
    def _write_payload(self, payload):
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name) / "baseline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.addCleanup(temp_dir.cleanup)
        return path

    def _valid_payload(self, include_test_case=True, is_partial=False):
        config = BiasConfig(
            testee_model="sentinel-model",
            data_file="sentinel.xlsx",
            sheet_name="Bias",
            attacker_strategies={},
        )
        metadata = build_bias_metadata(
            phase="baseline",
            config=config,
            is_partial=is_partial,
            dataset={
                "data_file": "sentinel.xlsx",
                "sheet_name": "Bias",
                "max_samples": None,
                "question_population": 1,
            },
            source={
                "data_file": "sentinel.xlsx",
                "sheet_name": "Bias",
            },
        )
        return {
            "metadata": metadata,
            "summary": {},
            "results": [make_result(include_test_case).to_dict()],
        }

    def test_loads_stable_metadata_written_by_baseline_runner(self):
        path = self._write_payload(self._valid_payload())

        with patch("scripts.bias.run_attack.load_bias_test_cases", return_value=[make_case()]):
            results, config, dataset_info = load_baseline_data(str(path), quiet=True)

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].test_case)
        self.assertEqual(config.testee_model, "sentinel-model")
        self.assertEqual(config.vote_num_baseline, 5)
        self.assertEqual(dataset_info["data_file"], "sentinel.xlsx")
        self.assertEqual(dataset_info["sheet_name"], "Bias")

    def test_rejects_missing_or_incompatible_metadata(self):
        mutations = {
            "missing config": lambda p: p["metadata"].pop("config"),
            "missing dataset": lambda p: p["metadata"].pop("dataset"),
            "wrong axis": lambda p: p["metadata"].update({"axis": "privacy"}),
            "wrong phase": lambda p: p["metadata"].update({"phase": "attack"}),
            "target mismatch": lambda p: p["metadata"].update({"target_model": "other"}),
            "empty results": lambda p: p.update({"results": []}),
            "partial baseline": lambda p: p["metadata"].update({"is_partial": True}),
        }

        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = self._valid_payload()
                mutate(payload)
                path = self._write_payload(payload)
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        load_baseline_data(str(path), quiet=True)
                self.assertEqual(raised.exception.code, 1)

    def test_allows_dataset_sha_drift_when_source_cases_still_match(self):
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "bias.xlsx"
            data_path.write_bytes(b"changed")
            payload = self._valid_payload()
            payload["metadata"]["source"].update({
                "data_file_path": str(data_path),
                "data_file_sha256": "not-the-current-sha",
            })
            path = self._write_payload(payload)

            with patch("scripts.bias.run_attack.load_bias_test_cases", return_value=[make_case()]):
                results, config, dataset_info = load_baseline_data(str(path), quiet=True)

        self.assertEqual(len(results), 1)
        self.assertEqual(config.testee_model, "sentinel-model")
        self.assertEqual(dataset_info["data_file"], "sentinel.xlsx")

    def test_rejects_duplicate_baseline_identities(self):
        payload = self._valid_payload()
        payload["results"].append(payload["results"][0])
        path = self._write_payload(payload)

        with patch("scripts.bias.run_attack.load_bias_test_cases", return_value=[make_case()]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    load_baseline_data(str(path), quiet=True)

        self.assertEqual(raised.exception.code, 1)

    def test_rejects_results_without_embedded_test_case(self):
        path = self._write_payload(self._valid_payload(include_test_case=False))

        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                load_baseline_data(str(path), quiet=True)

        self.assertEqual(raised.exception.code, 1)

    def test_rejects_baseline_rows_that_no_longer_match_current_source_cases(self):
        path = self._write_payload(self._valid_payload())
        changed_case = make_case()
        changed_case.clinical_summary = "A changed patient summary."

        with patch("scripts.bias.run_attack.load_bias_test_cases", return_value=[changed_case]):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    load_baseline_data(str(path), quiet=True)

        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
