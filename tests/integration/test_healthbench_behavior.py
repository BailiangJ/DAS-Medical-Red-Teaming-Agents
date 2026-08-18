import unittest
from types import SimpleNamespace



import numpy as np

from med_red_team.healthbench.attacker import CognitiveBiasStrategy
from med_red_team.healthbench.config import HealthBenchConfig
from med_red_team.healthbench.data import (
    AttackResult,
    HealthBenchRobustnessResult,
    HealthBenchTestCase,
    RubricGradeResult,
    RubricItem,
    build_healthbench_metadata,
    validate_cached_baseline_result,
)
from med_red_team.healthbench.grader import RubricGrader
from med_red_team.healthbench.pipeline import HealthBenchRobustnessPipeline
from med_red_team.healthbench.prompts import COGNITIVE_BIAS_SYSTEM_PROMPT
from med_red_team.healthbench.utils import (
    add_distraction_to_conversation,
    calculate_score,
    compute_bootstrap_std,
    validate_attack_result_state,
    validate_grading_results_match_rubrics,
    validate_selected_rubric_index,
)
from med_red_team.models import GenerationConfig, ModelGenerationError
from med_red_team.utils import normalize_bool, parse_json_response
from scripts.utils import reconstruct_healthbench_result


class ExplodingTestee:
    model_id = "fake-testee"

    def answer(self, *args, **kwargs):
        raise AssertionError("max_samples=0 must not call the testee")


class ExplodingGrader:
    model_id = "fake-grader"

    def grade(self, *args, **kwargs):
        raise AssertionError("max_samples=0 must not call the grader")


class ExplodingStrategy:
    name = "impossible_measurement"

    def apply_to_conversation(self, *args, **kwargs):
        raise AssertionError("max_samples=0 must not call the attacker")


class NonApplicableStrategy:
    name = "cognitive_bias"

    def apply_to_conversation(self, conversation, attackable_rubrics_with_context, test_case_id):
        return AttackResult(
            test_case_id=test_case_id,
            attack_strategy=self.name,
            applicable=False,
            original_conversation=conversation,
            attack_metadata={"selected_rubric_index": -1},
            modified_conversation=None,
            error="not applicable",
        )


class ExplodingBaselineStrategy:
    name = "cognitive_bias"

    def apply_to_conversation(self, *args, **kwargs):
        raise AssertionError("cached baseline validation must finish before attacks run")


class ContradictoryStrategy:
    name = "impossible_measurement"

    def apply_to_conversation(self, conversation, test_case_id):
        return AttackResult(
            test_case_id=test_case_id,
            attack_strategy=self.name,
            applicable=False,
            original_conversation=conversation,
            attack_metadata={},
            modified_conversation=[{"role": "user", "content": "modified"}],
            error="attacker produced contradictory state",
        )


class FailingStrategy:
    name = "impossible_measurement"

    def apply_to_conversation(self, *args, **kwargs):
        raise ModelGenerationError("boom")


class StaticModel:
    def __init__(self, final_answer):
        self.final_answer = final_answer

    def generate(self, *args, **kwargs):
        return SimpleNamespace(final_answer=self.final_answer)


class StaticModelPool:
    def __init__(self, final_answer):
        self.model = StaticModel(final_answer)

    def get_model(self, model_id):
        return self.model


class StaticAnswerTestee:
    model_id = "static-testee"

    def __init__(self, answer="attacked"):
        self.answer_text = answer

    def answer(self, *args, **kwargs):
        return SimpleNamespace(final_answer=self.answer_text, metadata={})


class StaticGrader:
    model_id = "static-grader"

    def __init__(self, grades):
        self.grades = grades

    def grade(self, *args, **kwargs):
        return self.grades


def make_baseline_metadata(
    *,
    testee_model="fake-testee",
    grader_model="fake-grader",
    testee_config=None,
    grader_config=None,
    system_prompt="system",
    phase="baseline",
    is_partial=False,
):
    config = HealthBenchConfig(
        testee_model=testee_model,
        grader_model=grader_model,
        testee_config=testee_config or GenerationConfig(temperature=0.1),
        grader_config=grader_config or GenerationConfig(temperature=0.2),
        testee_system_prompt=system_prompt,
    )
    return build_healthbench_metadata(
        phase=phase,
        config=config,
        is_partial=is_partial,
    )


def make_result(points, baseline_met, attacked_met, strategy="cognitive_bias"):
    rubric = RubricItem("target criterion", points, ["test"])
    baseline_grades = [] if baseline_met is None else [
        RubricGradeResult(rubric, baseline_met, "baseline")
    ]
    return HealthBenchRobustnessResult(
        test_case_id=f"case-{points}-{baseline_met}-{attacked_met}",
        sample_number=1,
        original_conversation=[{"role": "user", "content": "question"}],
        original_rubrics=[] if baseline_met is None else [rubric],
        baseline_completion="baseline" if baseline_met is not None else None,
        baseline_grades=baseline_grades,
        baseline_score=1.0 if baseline_met is not None else None,
        attack_result=AttackResult(
            test_case_id="case",
            attack_strategy=strategy,
            applicable=True,
            original_conversation=[{"role": "user", "content": "question"}],
            attack_metadata={},
            modified_conversation=[{"role": "user", "content": "modified question"}],
        ),
        attacked_conversation=[{"role": "user", "content": "modified question"}],
        attacked_rubrics=[rubric],
        attacked_completion="attacked",
        attacked_grades=[RubricGradeResult(rubric, attacked_met, "attacked")],
        attacked_score=None,
    )


class HealthBenchSuccessTests(unittest.TestCase):
    def test_positive_rubric_success(self):
        result = make_result(5, True, False)
        self.assertTrue(result.attack_successful)
        self.assertEqual(result.rubric_attack_opportunities, 1)
        self.assertEqual(result.rubric_attack_successes, 1)

    def test_negative_rubric_success(self):
        result = make_result(-5, False, True)
        self.assertTrue(result.attack_successful)
        self.assertEqual(result.rubric_attack_successes, 1)

    def test_non_successful_transitions(self):
        self.assertFalse(make_result(5, True, True).attack_successful)
        self.assertFalse(make_result(-5, False, False).attack_successful)

    def test_impossible_measurement_negative_rubric_success(self):
        result = make_result(-5, None, True, strategy="impossible_measurement")
        self.assertIsNone(result.attacked_score)
        self.assertTrue(result.attack_successful)
        self.assertEqual(result.rubric_attack_opportunities, 1)

    def test_duplicate_rubric_text_uses_original_rubric_index(self):
        rubric = RubricItem("duplicate", 5, ["test"])
        result = HealthBenchRobustnessResult(
            test_case_id="duplicate-case",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[rubric, rubric],
            baseline_completion="baseline",
            baseline_grades=[
                RubricGradeResult(rubric, True, "first"),
                RubricGradeResult(rubric, False, "second"),
            ],
            baseline_score=0.5,
            attack_result=AttackResult(
                test_case_id="duplicate-case",
                attack_strategy="cognitive_bias",
                applicable=True,
                original_conversation=[{"role": "user", "content": "question"}],
                attack_metadata={"original_rubric_index": 0},
                modified_conversation=[{"role": "user", "content": "modified"}],
            ),
            attacked_conversation=[{"role": "user", "content": "modified"}],
            attacked_rubrics=[rubric],
            attacked_completion="attacked",
            attacked_grades=[RubricGradeResult(rubric, False, "attacked")],
            attacked_score=0.0,
        )

        self.assertTrue(result.attack_successful)
        self.assertEqual(result.rubric_attack_successes, 1)

    def test_selected_rubric_score_change_uses_selected_baseline_rubric_only(self):
        rubric_a = RubricItem("first", 5, ["test"])
        rubric_b = RubricItem("target", 5, ["test"])
        baseline_result = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[rubric_a, rubric_b],
            baseline_completion="baseline",
            baseline_grades=[
                RubricGradeResult(rubric_a, False, "not met"),
                RubricGradeResult(rubric_b, True, "met"),
            ],
            baseline_score=0.5,
        )
        attack_result = AttackResult(
            test_case_id="case-1",
            attack_strategy="cognitive_bias",
            applicable=True,
            original_conversation=baseline_result.original_conversation,
            attack_metadata={
                "selected_rubric_index": 0,
                "original_rubric_index": 1,
                "attackable_rubric_items": [rubric_b.to_dict()],
            },
            modified_conversation=[{"role": "user", "content": "modified"}],
        )
        pipeline = HealthBenchRobustnessPipeline(
            StaticAnswerTestee(),
            StaticGrader([RubricGradeResult(rubric_b, False, "attacked")]),
            [],
            verbose=False,
        )
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=baseline_result.original_conversation,
            rubrics=[rubric_a, rubric_b],
            example_tags=[],
        )

        result = pipeline._evaluate_attacked(case, attack_result, baseline_result)

        self.assertEqual(result.baseline_score, 1.0)
        self.assertEqual(result.attacked_score, 0.0)
        self.assertEqual(result.score_change, 1.0)
        self.assertTrue(result.attack_successful)

    def test_summary_uses_polarity_aware_counts(self):
        results = [
            make_result(5, True, False),
            make_result(-5, False, True),
            make_result(-5, None, True, strategy="impossible_measurement"),
        ]
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [], verbose=False
        )
        summary = pipeline._compute_summary(results, baseline_only=False)

        self.assertEqual(summary.attacks_applicable, 3)
        self.assertEqual(summary.attacks_comparable, 3)
        self.assertEqual(summary.attacks_successful, 3)
        self.assertEqual(summary.total_attackable_rubrics, 3)
        self.assertEqual(summary.total_rubric_attack_successes, 3)
        self.assertEqual(summary.total_rubrics_degraded, 3)
        self.assertEqual(summary.case_attack_success_rate, 1.0)
        self.assertEqual(summary.rubric_attack_success_rate, 1.0)

    def test_summary_keeps_applicable_but_skipped_attacks_in_denominator(self):
        comparable = make_result(5, True, False)
        skipped = make_result(5, True, False)
        skipped.attacked_grades = []
        skipped.attacked_score = None
        skipped.skipped = True
        skipped.skip_reason = "parse error"
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [], verbose=False
        )

        summary = pipeline._compute_summary([comparable, skipped], baseline_only=False)

        self.assertEqual(summary.attacks_attempted, 2)
        self.assertEqual(summary.attacks_applicable, 2)
        self.assertEqual(summary.attacks_comparable, 1)
        self.assertEqual(summary.attacks_successful, 1)
        self.assertEqual(summary.case_attack_success_rate, 1.0)


class HealthBenchParsingTests(unittest.TestCase):
    def test_boolean_normalization(self):
        self.assertTrue(normalize_bool(True))
        self.assertFalse(normalize_bool(False))
        self.assertTrue(normalize_bool(" TRUE "))
        self.assertFalse(normalize_bool("false"))
        for value in (0, 1, "yes", "0", None, []):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_bool(value)

    def test_parse_json_response_handles_string_false(self):
        parsed = parse_json_response(
            '{"criteria_met": "false"}',
            expected_fields={"criteria_met": bool},
        )
        self.assertIs(parsed["criteria_met"], False)

    def test_invalid_boolean_is_omitted_without_fallback(self):
        parsed = parse_json_response(
            '{"criteria_met": "maybe"}',
            expected_fields={"criteria_met": bool},
            fallback={},
        )
        self.assertNotIn("criteria_met", parsed)

    def test_selected_rubric_index_validation(self):
        self.assertEqual(validate_selected_rubric_index(0, 1), 0)
        self.assertEqual(validate_selected_rubric_index(-1, 1), -1)
        for value in ("0", 0.0, True, False, -2, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_selected_rubric_index(value, 1)

    def test_add_distraction_copies_conversation_before_editing(self):
        conversation = [{"role": "user", "content": "original"}]

        modified = add_distraction_to_conversation(conversation, "distractor")

        self.assertEqual(conversation, [{"role": "user", "content": "original"}])
        self.assertEqual(
            modified,
            [{"role": "user", "content": "original\n\ndistractor"}],
        )

    def test_invalid_attacker_index_returns_non_applicable_result(self):
        response = """{
            "applicable": true,
            "selected_rubric_index": "0",
            "modified_conversation": "USER: modified"
        }"""
        strategy = CognitiveBiasStrategy(
            model_id="fake",
            model_pool=StaticModelPool(response),
        )
        result = strategy.apply_to_conversation(
            conversation=[{"role": "user", "content": "original"}],
            attackable_rubrics_with_context=[{
                "rubric": {"criterion": "target", "points": 1, "tags": []},
                "grader_explanation": "met",
                "rubric_index": 0,
            }],
            test_case_id="case",
        )
        self.assertFalse(result.applicable)
        self.assertIn("must be an integer", result.error)
        self.assertEqual(result.attack_metadata["selected_rubric_index_raw"], "0")

    def test_non_string_modified_conversation_is_rejected(self):
        response = """{
            "applicable": true,
            "selected_rubric_index": 0,
            "modified_conversation": [{"role": "user", "content": "modified"}]
        }"""
        strategy = CognitiveBiasStrategy(
            model_id="fake",
            model_pool=StaticModelPool(response),
        )
        result = strategy.apply_to_conversation(
            conversation=[{"role": "user", "content": "original"}],
            attackable_rubrics_with_context=[{
                "rubric": {"criterion": "target", "points": 1, "tags": []},
                "grader_explanation": "met",
                "rubric_index": 0,
            }],
            test_case_id="case",
        )
        self.assertTrue(result.applicable)
        self.assertIsNone(result.modified_conversation)
        self.assertIn("must be a non-empty string", result.error)

    def test_cognitive_bias_json_fields_are_comma_separated(self):
        self.assertIn(
            'explain selection. ≤100 words>",\n\n  "selected_rubric_index"',
            COGNITIVE_BIAS_SYSTEM_PROMPT,
        )

    def test_reconstruction_normalizes_legacy_booleans_and_indices(self):
        payload = make_result(-5, False, True).to_dict()
        payload["attack_result"]["applicable"] = "false"
        payload["attack_result"]["attack_metadata"] = {
            "selected_rubric_index": "0",
            "original_rubric_index": "0",
            "attackable_rubric_items": [
                {"criterion": "target criterion", "points": -5, "tags": ["test"]}
            ],
        }
        payload["baseline_grades"][0]["criteria_met"] = "false"
        payload["attacked_grades"][0]["criteria_met"] = "true"

        reconstructed = reconstruct_healthbench_result(payload)

        self.assertIs(reconstructed.attack_result.applicable, False)
        self.assertIs(reconstructed.baseline_grades[0].criteria_met, False)
        self.assertIs(reconstructed.attacked_grades[0].criteria_met, True)
        self.assertEqual(
            reconstructed.attack_result.attack_metadata["selected_rubric_index"], -1
        )
        self.assertEqual(
            reconstructed.attack_result.attack_metadata["selected_rubric_index_raw"], "0"
        )
        self.assertTrue(reconstructed.skipped)

    def test_reconstruction_preserves_integer_index_without_attackable_list(self):
        payload = make_result(5, True, False).to_dict()
        payload["attack_result"]["attack_metadata"] = {
            "selected_rubric_index": 0,
            "original_rubric_index": 0,
        }
        reconstructed = reconstruct_healthbench_result(payload)
        self.assertEqual(
            reconstructed.attack_result.attack_metadata["selected_rubric_index"], 0
        )

    def test_reconstruction_rebases_selected_rubric_baseline_score(self):
        rubric_a = RubricItem("first", 5, ["test"])
        rubric_b = RubricItem("target", 5, ["test"])
        payload = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[rubric_a, rubric_b],
            baseline_completion="baseline",
            baseline_grades=[
                RubricGradeResult(rubric_a, False, "no"),
                RubricGradeResult(rubric_b, True, "yes"),
            ],
            baseline_score=0.5,
            attack_result=AttackResult(
                test_case_id="case-1",
                attack_strategy="cognitive_bias",
                applicable=True,
                original_conversation=[{"role": "user", "content": "question"}],
                attack_metadata={
                    "selected_rubric_index": 0,
                    "original_rubric_index": 1,
                    "attackable_rubric_items": [rubric_b.to_dict()],
                },
                modified_conversation=[{"role": "user", "content": "modified"}],
            ),
            attacked_conversation=[{"role": "user", "content": "modified"}],
            attacked_rubrics=[rubric_b],
            attacked_completion="attacked",
            attacked_grades=[RubricGradeResult(rubric_b, False, "failed")],
            attacked_score=0.0,
        ).to_dict()

        reconstructed = reconstruct_healthbench_result(payload)

        self.assertEqual(reconstructed.baseline_score, 1.0)
        self.assertEqual(reconstructed.attacked_score, 0.0)
        self.assertEqual(reconstructed.score_change, 1.0)


class HealthBenchScoringAndValidationTests(unittest.TestCase):
    def test_compute_bootstrap_std_is_deterministic_and_clipped(self):
        values = [-0.5, 1.5, 0.25]
        seed = 17
        observed = compute_bootstrap_std(values, n_bootstrap=8, seed=seed)

        rng = np.random.default_rng(seed)
        means = []
        array = np.asarray(values, dtype=float)
        for _ in range(8):
            sample = rng.choice(array, size=len(array), replace=True)
            means.append(np.clip(np.mean(sample), 0, 1))
        expected = float(np.std(means))

        self.assertEqual(observed, expected)
        self.assertEqual(observed, compute_bootstrap_std(values, n_bootstrap=8, seed=seed))

    def test_validate_attack_result_state_rejects_contradictions(self):
        original = [{"role": "user", "content": "original"}]
        modified = [{"role": "user", "content": "modified"}]
        self.assertIn(
            "applicable=False",
            validate_attack_result_state(
                applicable=False,
                modified_conversation=modified,
                original_conversation=original,
                selected_rubric_index=-1,
            ),
        )
        self.assertIn(
            "selected_rubric_index=-1",
            validate_attack_result_state(
                applicable=True,
                modified_conversation=modified,
                original_conversation=original,
                selected_rubric_index=-1,
            ),
        )

    def test_cached_baseline_validation_requires_exact_match(self):
        rubric = RubricItem("criterion", 1, [])
        test_case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "different"}],
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
        )
        with self.assertRaisesRegex(ValueError, "conversation mismatch"):
            validate_cached_baseline_result(test_case, baseline)

    def test_score_validation_rejects_mismatched_grade_lists(self):
        rubric = RubricItem("criterion", 1, [])
        other = RubricItem("other", 1, [])
        grades = [RubricGradeResult(other, True, "met")]

        self.assertIn(
            "rubric mismatch",
            validate_grading_results_match_rubrics([rubric], grades),
        )
        self.assertIsNone(calculate_score([rubric], grades))
        self.assertIsNone(calculate_score([rubric], []))

    def test_cached_baseline_validation_rejects_mismatched_grade_shape(self):
        rubric = RubricItem("criterion", 1, [])
        test_case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=test_case.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(RubricItem("other", 1, []), True, "met")],
            baseline_score=1.0,
        )

        with self.assertRaisesRegex(ValueError, "cached baseline grades rubric mismatch"):
            validate_cached_baseline_result(test_case, baseline)

    def test_cached_baseline_validation_rejects_attack_results(self):
        rubric = RubricItem("criterion", 1, [])
        test_case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=test_case.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
            attack_result=AttackResult(
                test_case_id="case-1",
                attack_strategy="cognitive_bias",
                applicable=True,
                original_conversation=test_case.conversation,
                attack_metadata={},
                modified_conversation=[{"role": "user", "content": "modified"}],
            ),
            attacked_conversation=[{"role": "user", "content": "modified"}],
        )

        with self.assertRaisesRegex(ValueError, "not baseline-only"):
            validate_cached_baseline_result(test_case, baseline)

    def test_cached_baseline_validation_rejects_stale_attacked_score(self):
        rubric = RubricItem("criterion", 1, [])
        test_case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=test_case.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
            attacked_score=0.0,
        )

        with self.assertRaisesRegex(ValueError, "not baseline-only"):
            validate_cached_baseline_result(test_case, baseline)

    def test_cached_baseline_validation_rejects_score_grade_mismatch(self):
        rubric = RubricItem("criterion", 1, [])
        test_case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=test_case.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=0.0,
        )

        with self.assertRaisesRegex(ValueError, "score does not match"):
            validate_cached_baseline_result(test_case, baseline)

    def test_rubric_grader_parse_exhaustion_is_invalid(self):
        grader = RubricGrader(
            model_id="fake",
            model_pool=StaticModelPool('{"criteria_met": "maybe"}'),
            max_retries=1,
            verbose=False,
        )
        result = grader.grade(
            conversation=[{"role": "user", "content": "question"}],
            completion="answer",
            rubric_items=[RubricItem("criterion", 1, [])],
        )[0]
        self.assertIsNone(result.criteria_met)
        self.assertIn("Failed to parse", result.explanation)


class HealthBenchLimitTests(unittest.TestCase):
    def test_direct_pipeline_rejects_negative_sample_limits(self):
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [ExplodingStrategy()], verbose=False
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pipeline.run_baseline(
                test_cases=[object()],
                max_samples=-1,
                filter_single_turn=False,
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pipeline.run_attack(
                test_cases=[object()],
                max_samples=-1,
                filter_single_turn=False,
            )

    def test_baseline_max_samples_zero_does_no_work(self):
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [], verbose=False
        )
        results, summary = pipeline.run_baseline(
            test_cases=[object()],
            max_samples=0,
            filter_single_turn=False,
        )
        self.assertEqual(results, [])
        self.assertEqual(summary.total_cases, 0)

    def test_attack_max_samples_zero_does_no_work(self):
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [ExplodingStrategy()], verbose=False
        )
        results, summary = pipeline.run_attack(
            test_cases=[object()],
            max_samples=0,
            filter_single_turn=False,
        )
        self.assertEqual(results, [])
        self.assertEqual(summary.total_cases, 0)

    def test_zero_work_ignores_provenance_less_cached_baseline(self):
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [NonApplicableStrategy()], verbose=False
        )
        results, summary = pipeline.run_attack(
            test_cases=[object()],
            max_samples=0,
            filter_single_turn=False,
            baseline_results=[],
        )
        self.assertEqual(results, [])
        self.assertEqual(summary.total_cases, 0)

    def test_impossible_measurement_ignores_unused_baseline_metadata(self):
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("target", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [ContradictoryStrategy()], verbose=False
        )
        results, _ = pipeline.run_attack(
            test_cases=[case],
            baseline_results=[
                SimpleNamespace(test_case_id="duplicate"),
                SimpleNamespace(test_case_id="duplicate"),
            ],
            filter_single_turn=False,
        )
        self.assertTrue(results[0].skipped)

    def test_direct_pipeline_rejects_cached_baseline_without_metadata(self):
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("target", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(), ExplodingGrader(), [NonApplicableStrategy()], verbose=False
        )

        with self.assertRaisesRegex(ValueError, "require metadata"):
            pipeline.run_attack(
                test_cases=[case],
                baseline_results=[],
                filter_single_turn=False,
            )

    def test_direct_pipeline_rejects_incompatible_cached_baseline_metadata(self):
        testee_config = GenerationConfig(temperature=0.1, top_p=0.8)
        grader_config = GenerationConfig(temperature=0.2, top_k=5)
        testee = SimpleNamespace(
            model_id="fake-testee",
            config=testee_config,
            system_prompt="system",
        )
        grader = SimpleNamespace(model_id="fake-grader", config=grader_config)
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("target", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            testee, grader, [NonApplicableStrategy()], verbose=False
        )

        mismatches = (
            (make_baseline_metadata(testee_model="other"), "testee model mismatch"),
            (
                make_baseline_metadata(
                    testee_config=GenerationConfig(temperature=0.9, top_p=0.8),
                    grader_config=grader_config,
                ),
                "testee config mismatch",
            ),
            (
                make_baseline_metadata(
                    testee_config=testee_config,
                    grader_config=grader_config,
                    system_prompt="other",
                ),
                "system prompt mismatch",
            ),
            (
                make_baseline_metadata(
                    grader_model="other",
                    testee_config=testee_config,
                    grader_config=grader_config,
                ),
                "grader model mismatch",
            ),
            (
                make_baseline_metadata(
                    testee_config=testee_config,
                    grader_config=GenerationConfig(temperature=0.9, top_k=5),
                ),
                "grader config mismatch",
            ),
            (
                make_baseline_metadata(
                    testee_config=testee_config,
                    grader_config=grader_config,
                    phase="attack",
                ),
                "phase must be 'baseline'",
            ),
            (
                make_baseline_metadata(
                    testee_config=testee_config,
                    grader_config=grader_config,
                    is_partial=True,
                ),
                "must be complete",
            ),
        )
        for metadata, message in mismatches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.run_attack(
                        test_cases=[case],
                        baseline_results=[],
                        baseline_metadata=metadata,
                        filter_single_turn=False,
                    )

    def test_direct_pipeline_accepts_compatible_cached_baseline_metadata(self):
        rubric = RubricItem("target", 1, [])
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
        testee_config = GenerationConfig(temperature=0.1, top_p=0.8)
        grader_config = GenerationConfig(temperature=0.2, top_k=5)
        pipeline = HealthBenchRobustnessPipeline(
            SimpleNamespace(
                model_id="fake-testee",
                config=testee_config,
                system_prompt="system",
            ),
            SimpleNamespace(model_id="fake-grader", config=grader_config),
            [NonApplicableStrategy()],
            verbose=False,
        )

        results, _ = pipeline.run_attack(
            test_cases=[case],
            baseline_results=[baseline],
            baseline_metadata=make_baseline_metadata(
                testee_config=testee_config,
                grader_config=grader_config,
            ),
            filter_single_turn=False,
        )

        self.assertTrue(results[0].skipped)

    def test_non_applicable_attack_does_not_mutate_cached_baseline(self):
        rubric = RubricItem("target", 1, [])
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
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [NonApplicableStrategy()],
            verbose=False,
        )

        results, _ = pipeline.run_attack(
            test_cases=[case],
            baseline_results=[baseline],
            baseline_metadata=make_baseline_metadata(),
            filter_single_turn=False,
        )

        self.assertIsNone(baseline.attack_result)
        self.assertFalse(baseline.skipped)
        self.assertIsNot(results[0], baseline)
        self.assertTrue(results[0].skipped)

    def test_contradictory_attack_state_is_rejected_before_target_evaluation(self):
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("criterion", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [ContradictoryStrategy()],
            verbose=False,
        )

        results, _ = pipeline.run_attack(
            test_cases=[case],
            filter_single_turn=False,
        )

        self.assertTrue(results[0].skipped)
        self.assertIn("applicable=False", results[0].skip_reason)

    def test_direct_pipeline_rejects_under_covering_cached_baseline(self):
        rubric = RubricItem("target", 1, [])
        first = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question 1"}],
            rubrics=[rubric],
            example_tags=[],
        )
        second = HealthBenchTestCase(
            prompt_id="case-2",
            conversation=[{"role": "user", "content": "question 2"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baseline = HealthBenchRobustnessResult(
            test_case_id="case-1",
            sample_number=1,
            original_conversation=first.conversation,
            original_rubrics=[rubric],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(rubric, True, "met")],
            baseline_score=1.0,
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [NonApplicableStrategy()],
            verbose=False,
        )

        with self.assertRaisesRegex(ValueError, "do not cover all requested HealthBench cases"):
            pipeline.run_attack(
                test_cases=[first, second],
                baseline_results=[baseline],
                baseline_metadata=make_baseline_metadata(),
                filter_single_turn=False,
            )

    def test_direct_pipeline_preflights_all_cached_rows_before_attack_work(self):
        rubric = RubricItem("target", 1, [])
        first = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question 1"}],
            rubrics=[rubric],
            example_tags=[],
        )
        second = HealthBenchTestCase(
            prompt_id="case-2",
            conversation=[{"role": "user", "content": "question 2"}],
            rubrics=[rubric],
            example_tags=[],
        )
        baselines = [
            HealthBenchRobustnessResult(
                test_case_id="case-1",
                sample_number=1,
                original_conversation=first.conversation,
                original_rubrics=[rubric],
                baseline_completion="answer",
                baseline_grades=[RubricGradeResult(rubric, True, "met")],
                baseline_score=1.0,
            ),
            HealthBenchRobustnessResult(
                test_case_id="case-2",
                sample_number=2,
                original_conversation=[{"role": "user", "content": "stale"}],
                original_rubrics=[rubric],
                baseline_completion="answer",
                baseline_grades=[RubricGradeResult(rubric, True, "met")],
                baseline_score=1.0,
            ),
        ]
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [ExplodingBaselineStrategy()],
            verbose=False,
        )

        with self.assertRaisesRegex(ValueError, "conversation mismatch"):
            pipeline.run_attack(
                test_cases=[first, second],
                baseline_results=baselines,
                baseline_metadata=make_baseline_metadata(),
                filter_single_turn=False,
            )

    def test_invalid_scores_are_excluded_from_summary_means(self):
        valid = HealthBenchRobustnessResult(
            test_case_id="valid",
            sample_number=1,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[RubricItem("criterion", 1, [])],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(RubricItem("criterion", 1, []), True, "met")],
            baseline_score=1.0,
            attacked_score=None,
        )
        invalid = HealthBenchRobustnessResult(
            test_case_id="invalid",
            sample_number=2,
            original_conversation=[{"role": "user", "content": "question"}],
            original_rubrics=[RubricItem("criterion", 1, [])],
            baseline_completion="answer",
            baseline_grades=[RubricGradeResult(RubricItem("criterion", 1, []), None, "invalid")],
            baseline_score=None,
            attacked_score=None,
            skipped=True,
            skip_reason="invalid",
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [],
            verbose=False,
            bootstrap_seed=123,
        )
        summary = pipeline._compute_summary([valid, invalid], baseline_only=True)
        self.assertEqual(summary.baseline_avg_score, 1.0)
        self.assertEqual(summary.total_cases, 2)

    def test_direct_pipeline_rejects_multiple_attack_strategies(self):
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("criterion", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [NonApplicableStrategy(), ContradictoryStrategy()],
            verbose=False,
        )
        with self.assertRaisesRegex(ValueError, "Exactly one HealthBench attack strategy"):
            pipeline.run_attack(test_cases=[case], filter_single_turn=False)

    def test_model_execution_errors_propagate(self):
        case = HealthBenchTestCase(
            prompt_id="case-1",
            conversation=[{"role": "user", "content": "question"}],
            rubrics=[RubricItem("criterion", 1, [])],
            example_tags=[],
        )
        pipeline = HealthBenchRobustnessPipeline(
            ExplodingTestee(),
            ExplodingGrader(),
            [FailingStrategy()],
            verbose=False,
        )
        with self.assertRaises(ModelGenerationError):
            pipeline.run_attack(test_cases=[case], filter_single_turn=False)


if __name__ == "__main__":
    unittest.main()
