import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from med_red_team.bias.attacker import (
    EmotionManipulationStrategy,
    LanguageManipulationStrategy,
    RaceSocioeconomicLabelStrategy,
)
from med_red_team.bias.data import BiasReferenceResult, BiasResult, BiasTestCase
from med_red_team.bias.grader import BiasGrader
from med_red_team.bias.pipeline import BiasPipeline
from med_red_team.bias.utils import (
    INVALID_VOTE_OUTCOME,
    NO_WINNER_VOTE_OUTCOME,
    VALID_VOTE_OUTCOME,
    is_mental_health_case,
    normalize_single_vote,
    summarize_votes,
)
from med_red_team.models import GenerationConfig, ModelExecutionError


class FakeTestee:
    model_id = "fake-testee"

    def __init__(self, answer="A"):
        self.final_answer = answer
        self.prompts = []

    def answer(self, prompt, system_prompt=None, config=None):
        self.prompts.append(prompt)
        return SimpleNamespace(final_answer=self.final_answer, metadata={})


class SequenceTestee:
    model_id = "sequence-testee"

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def answer(self, prompt, system_prompt=None, config=None):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError("No more scripted answers")
        next_item = self.answers.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return SimpleNamespace(final_answer=next_item, metadata={})


class StaticStrategy:
    def __init__(self, name, modified_case):
        self.name = name
        self.modified_case = modified_case
        self.calls = 0

    def apply(self, test_case, context):
        self.calls += 1
        return self.modified_case


class ExplodingModel:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("Excluded emotional cases must not call the model")


class FakeModelPool:
    def __init__(self, model):
        self.model = model

    def get_model(self, model_id):
        return self.model


class RecordingModel:
    def __init__(self, final_answer):
        self.final_answer = final_answer
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(final_answer=self.final_answer)


class ProviderFailStrategy:
    name = "race_socioeconomic_label"

    def apply(self, test_case, context):
        raise ModelExecutionError("provider down")



def make_case(
    category="cardiology",
    question="Which option is most appropriate? A: One B: Two C: Three D: Four",
    clinical_summary="A 45-year-old {patient} presents for evaluation.",
):
    return BiasTestCase(
        case_id="case-1",
        clinical_summary=clinical_summary,
        vital_signs="BP 120/80",
        patient_self_report="I have had symptoms for two days.",
        questions=[question, None, None, None],
        category=category,
    )



def make_reference(case, vote_outcome=VALID_VOTE_OUTCOME, vote="A"):
    return BiasReferenceResult(
        case_id=case.case_id,
        question_idx=0,
        category=case.category,
        case_text=case.format_full_case(0, patient_placeholder="patient"),
        ref_responses=[vote] * 5 if vote_outcome == VALID_VOTE_OUTCOME else [""] * 5,
        ref_majority_vote=vote if vote_outcome == VALID_VOTE_OUTCOME else "",
        ref_vote_entropy=0.0,
        unbiased_model_choice=(f"{vote}: One" if vote_outcome == VALID_VOTE_OUTCOME else "Invalid vote"),
        test_case=case,
        ref_vote_outcome=vote_outcome,
        ref_valid_vote_count=(5 if vote_outcome == VALID_VOTE_OUTCOME else 0),
        ref_invalid_vote_count=(0 if vote_outcome == VALID_VOTE_OUTCOME else 5),
        ref_vote_counts=({vote: 5} if vote_outcome == VALID_VOTE_OUTCOME else {}),
        ref_configured_total_attempts=5,
        ref_required_votes=3,
    )


class BiasVoteNormalizationTests(unittest.TestCase):
    def test_normalize_single_vote_accepts_reasonable_formats(self):
        valid_labels = ("A", "B", "C", "D")
        self.assertEqual(normalize_single_vote("A", valid_labels), "A")
        self.assertEqual(normalize_single_vote("a", valid_labels), "A")
        self.assertEqual(normalize_single_vote("A.", valid_labels), "A")
        self.assertEqual(normalize_single_vote("The answer is A", valid_labels), "A")

    def test_summarize_votes_excludes_invalid_and_multiple_votes(self):
        summary = summarize_votes(
            ["A", "a.", "The answer is A", "A,B", "because"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=5,
        )

        self.assertEqual(summary.outcome, VALID_VOTE_OUTCOME)
        self.assertEqual(summary.majority_vote, "A")
        self.assertEqual(summary.valid_votes, ["A", "A", "A"])
        self.assertEqual(len(summary.invalid_votes), 2)

    def test_summarize_votes_represents_no_winner_explicitly(self):
        summary = summarize_votes(
            ["A", "B", "C", "noise", "noise"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=5,
        )

        self.assertEqual(summary.outcome, NO_WINNER_VOTE_OUTCOME)
        self.assertEqual(summary.majority_vote, "")
        self.assertEqual(summary.required_votes, 3)

    def test_strict_majority_rejects_two_one_one_plurality(self):
        summary = summarize_votes(
            ["A", "A", "B", "C", "noise"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=5,
        )

        self.assertEqual(summary.outcome, NO_WINNER_VOTE_OUTCOME)
        self.assertEqual(summary.majority_vote, "")
        self.assertEqual(summary.counts, {"A": 2, "B": 1, "C": 1})

    def test_invalid_votes_do_not_reduce_configured_denominator(self):
        summary = summarize_votes(
            ["A", "A", "noise", "noise", "noise"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=5,
        )

        self.assertEqual(summary.outcome, INVALID_VOTE_OUTCOME)
        self.assertEqual(summary.required_votes, 3)
        self.assertEqual(summary.majority_vote, "")

    def test_strict_majority_generalizes_to_larger_odd_panels(self):
        seven = summarize_votes(
            ["A", "A", "A", "B", "B", "C", "D"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=7,
        )
        nine = summarize_votes(
            ["A", "A", "A", "A", "A", "noise", "noise", "noise", "noise"],
            question_text="Question? A: One B: Two C: Three D: Four",
            configured_total_attempts=9,
        )

        self.assertEqual(seven.outcome, NO_WINNER_VOTE_OUTCOME)
        self.assertEqual(seven.required_votes, 4)
        self.assertEqual(nine.outcome, VALID_VOTE_OUTCOME)
        self.assertEqual(nine.majority_vote, "A")
        self.assertEqual(nine.required_votes, 5)


class BiasFormattingTests(unittest.TestCase):
    def test_format_full_case_replaces_patient_placeholder(self):
        case = make_case()

        neutral = case.format_full_case(0, patient_placeholder="patient")
        template = case.format_full_case(0)

        self.assertIn("45-year-old patient", neutral)
        self.assertNotIn("{patient}", neutral)
        self.assertIn("{patient}", template)

    def test_non_identity_final_prompt_is_neutral(self):
        case = make_case()
        testee = FakeTestee()
        modified_case = make_case()
        modified_case.patient_self_report += " Rewritten."
        strategy = StaticStrategy("language_manipulation", modified_case)
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=1,
            verbose=False,
        )

        results, summary = pipeline.run_attack([make_reference(case)])

        self.assertEqual(len(results), 1)
        self.assertEqual(summary.total_samples, 1)
        self.assertNotIn("{patient}", results[0].manipulated_case)
        self.assertIn("45-year-old patient", results[0].manipulated_case)
        self.assertEqual(testee.prompts, [results[0].manipulated_case])

    def test_identity_final_prompt_preserves_selected_label(self):
        case = make_case()
        labeled_case = make_case(
            clinical_summary="A 45-year-old black patient presents for evaluation."
        )
        testee = FakeTestee()
        strategy = StaticStrategy("race_socioeconomic_label", labeled_case)
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=1,
            verbose=False,
        )

        results, _summary = pipeline.run_attack([make_reference(case)])

        self.assertIn("black patient", results[0].manipulated_case)
        self.assertNotIn("{patient}", results[0].manipulated_case)


class BiasApplicabilityTests(unittest.TestCase):
    def test_identity_without_placeholder_is_not_applicable_before_model_call(self):
        model = ExplodingModel()
        strategy = RaceSocioeconomicLabelStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
        )
        case = make_case(clinical_summary="A 45-year-old patient presents.")

        modified = strategy.apply(
            case,
            {
                "ref_answer": "A",
                "unbiased_model_choice": "A: One",
                "question_idx": 0,
            },
        )

        self.assertTrue(modified.metadata["attack_excluded"])
        self.assertEqual(modified.metadata["status"], "not_applicable")
        self.assertEqual(model.calls, 0)

    def test_identity_prompt_uses_full_baseline_option_text(self):
        model = RecordingModel("black patient")
        strategy = RaceSocioeconomicLabelStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
            verbose=False,
        )
        case = make_case()

        modified = strategy.apply(
            case,
            {
                "ref_answer": "A",
                "unbiased_model_choice": "A: One",
                "question_idx": 0,
            },
        )

        self.assertIn("black patient", modified.clinical_summary)
        self.assertEqual(modified.metadata["ref_answer"], "A")
        self.assertIn("A: One", model.calls[0]["user_prompt"])
        self.assertIn("A: One", model.calls[0]["system_prompt"])

    def test_quiet_identity_strategy_suppresses_raw_attacker_output(self):
        model = RecordingModel("black patient")
        strategy = RaceSocioeconomicLabelStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
            verbose=False,
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            strategy.apply(
                make_case(),
                {
                    "ref_answer": "A",
                    "unbiased_model_choice": "A: One",
                    "question_idx": 0,
                },
            )

        self.assertEqual(output.getvalue(), "")

    def test_language_rewrite_temperature_override_preserves_generation_fields(self):
        model = RecordingModel("rewritten")
        config = GenerationConfig(
            temperature=0.9,
            top_p=0.4,
            top_k=17,
            max_tokens=321,
            repetition_penalty=1.2,
            stop_sequences=["STOP"],
            use_thinking=True,
            max_thinking_tokens=222,
            reasoning_effort="high",
        )
        strategy = LanguageManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
            config=config,
            verbose=False,
        )

        rewritten = strategy._rewrite_text("Original report", "limited english")

        self.assertEqual(rewritten, "rewritten")
        effective = model.calls[0]["config"]
        self.assertEqual(effective.temperature, 0.3)
        self.assertEqual(effective.top_p, config.top_p)
        self.assertEqual(effective.top_k, config.top_k)
        self.assertEqual(effective.max_tokens, config.max_tokens)
        self.assertEqual(effective.repetition_penalty, config.repetition_penalty)
        self.assertEqual(effective.stop_sequences, config.stop_sequences)
        self.assertEqual(effective.use_thinking, config.use_thinking)
        self.assertEqual(effective.max_thinking_tokens, config.max_thinking_tokens)
        self.assertEqual(effective.reasoning_effort, config.reasoning_effort)

    def test_emotion_rewrite_temperature_override_preserves_generation_fields(self):
        model = RecordingModel("rewritten")
        config = GenerationConfig(
            temperature=0.8,
            top_p=0.5,
            top_k=23,
            max_tokens=456,
            repetition_penalty=1.3,
            stop_sequences=["HALT"],
            use_thinking=True,
            max_thinking_tokens=333,
            reasoning_effort="low",
        )
        strategy = EmotionManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
            config=config,
            verbose=False,
        )

        rewritten = strategy._rewrite_text("Original report", "anxious")

        self.assertEqual(rewritten, "rewritten")
        effective = model.calls[0]["config"]
        self.assertEqual(effective.temperature, 0.3)
        self.assertEqual(effective.top_p, config.top_p)
        self.assertEqual(effective.top_k, config.top_k)
        self.assertEqual(effective.max_tokens, config.max_tokens)
        self.assertEqual(effective.repetition_penalty, config.repetition_penalty)
        self.assertEqual(effective.stop_sequences, config.stop_sequences)
        self.assertEqual(effective.use_thinking, config.use_thinking)
        self.assertEqual(effective.max_thinking_tokens, config.max_thinking_tokens)
        self.assertEqual(effective.reasoning_effort, config.reasoning_effort)

    def test_missing_self_report_is_not_applicable_before_language_model_call(self):
        model = ExplodingModel()
        strategy = LanguageManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
        )
        case = make_case()
        case.patient_self_report = "nan"

        modified = strategy.apply(
            case,
            {
                "ref_answer": "A",
                "unbiased_model_choice": "A: One",
                "question_idx": 0,
            },
        )

        self.assertTrue(modified.metadata["attack_excluded"])
        self.assertEqual(modified.metadata["status"], "not_applicable")
        self.assertEqual(model.calls, 0)

    def test_missing_self_report_is_not_applicable_before_emotion_model_call(self):
        model = ExplodingModel()
        strategy = EmotionManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
        )
        case = make_case()
        case.patient_self_report = "   "

        modified = strategy.apply(
            case,
            {
                "ref_answer": "A",
                "unbiased_model_choice": "A: One",
                "question_idx": 0,
            },
        )

        self.assertTrue(modified.metadata["attack_excluded"])
        self.assertEqual(modified.metadata["status"], "not_applicable")
        self.assertEqual(model.calls, 0)

    def test_pipeline_rejects_unchanged_final_prompt_without_target_call(self):
        case = make_case()
        testee = FakeTestee()
        strategy = StaticStrategy("language_manipulation", case)
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=1,
            verbose=False,
        )

        results, summary = pipeline.run_attack([make_reference(case)])

        self.assertEqual(results[0].evaluation_outcome, "manipulation_failed")
        self.assertEqual(results[0].attack_status, "manipulation_failed")
        self.assertEqual(testee.prompts, [])
        self.assertEqual(summary.failed_manipulations, 1)


class BiasEmotionExclusionTests(unittest.TestCase):
    def test_mental_health_helper_is_case_insensitive(self):
        self.assertTrue(is_mental_health_case(" PSYCHIATRIC ", "Routine question"))
        self.assertTrue(is_mental_health_case("Cardiology", " Mental health review "))
        self.assertTrue(is_mental_health_case("Cardiology", "Psych assessment"))
        self.assertFalse(is_mental_health_case("Cardiology", "Chest pain review"))
        self.assertFalse(is_mental_health_case("Developmental", "Routine review"))

    def test_emotion_strategy_excludes_before_model_call(self):
        model = ExplodingModel()
        strategy = EmotionManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
        )
        case = make_case(category="PSYCHIATRIC")

        modified = strategy.apply(
            case,
            {"ref_answer": "A", "question_idx": 0},
        )

        self.assertEqual(model.calls, 0)
        self.assertTrue(modified.metadata["attack_excluded"])
        self.assertIn("psychiatric/mental-health", modified.metadata["skip_reason"])
        self.assertIsNot(modified, case)

    def test_pipeline_records_emotional_exclusion_as_skip(self):
        model = ExplodingModel()
        strategy = EmotionManipulationStrategy(
            model_id="fake-attacker",
            model_pool=FakeModelPool(model),
        )
        case = make_case(
            question="What is the next mental health step? A: One B: Two C: Three D: Four"
        )
        testee = FakeTestee()
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=1,
            verbose=False,
        )

        results, summary = pipeline.run_attack([make_reference(case)])

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].skipped)
        self.assertEqual(results[0].evaluation_outcome, "attack_excluded")
        self.assertIn("psychiatric/mental-health", results[0].skip_reason)
        self.assertEqual(summary.skipped_samples, 1)
        self.assertEqual(summary.failed_manipulations, 0)
        self.assertEqual(testee.prompts, [])
        self.assertEqual(model.calls, 0)

    def test_quiet_pipeline_disables_progress_bar(self):
        case = make_case()
        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[],
            vote_num_baseline=1,
            verbose=False,
        )

        with patch(
            "med_red_team.bias.pipeline.evaluate_items_fail_fast",
            return_value=[],
        ) as evaluator:
            pipeline.run_baseline([case])

        self.assertFalse(evaluator.call_args.kwargs["use_tqdm"])

    def test_direct_baseline_pipeline_merges_existing_results_with_fixed_population(self):
        case = make_case()
        existing = make_reference(case)
        second = make_case()
        second.case_id = "case-2"
        testee = FakeTestee()
        existing.ref_responses = ["A"]
        existing.ref_configured_total_attempts = 1
        existing.ref_required_votes = 1
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[],
            vote_num_baseline=1,
            verbose=False,
        )

        results, summary = pipeline.run_baseline(
            [case, second],
            existing_results=[existing],
            completed_baseline_keys={(case.case_id, 0)},
            expected_population=2,
            save_cache=False,
        )

        self.assertEqual({result.case_id for result in results}, {"case-1", "case-2"})
        self.assertEqual(summary.baseline_population, 2)
        self.assertEqual(summary.baseline_valid_count, 2)
        self.assertAlmostEqual(summary.baseline_valid_coverage, 1.0)
        self.assertEqual(len(testee.prompts), 1)

    def test_direct_baseline_pipeline_rejects_stale_embedded_source_case(self):
        case = make_case()
        stale = make_reference(case)
        stale.test_case = make_case(clinical_summary="A different patient summary.")
        testee = FakeTestee()
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[],
            vote_num_baseline=1,
            verbose=False,
        )

        results, summary = pipeline.run_baseline(
            [case],
            existing_results=[stale],
            completed_baseline_keys={(case.case_id, 0)},
            expected_population=1,
            save_cache=False,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].case_text, case.format_full_case(0, patient_placeholder="patient"))
        self.assertEqual(summary.baseline_valid_count, 1)
        self.assertEqual(len(testee.prompts), 1)

    def test_direct_baseline_pipeline_recollects_incompatible_cache_row(self):
        case = make_case()
        stale = make_reference(case)
        stale.test_case = make_case(clinical_summary="A different patient summary.")
        testee = FakeTestee()
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[],
            vote_num_baseline=1,
            verbose=False,
        )

        with patch(
            "med_red_team.bias.pipeline.load_baseline_cache",
            return_value={f"{case.case_id}_q0": stale},
        ):
            results, summary = pipeline.run_baseline(
                [case],
                load_from_cache=True,
                cache_file="cache.json",
                save_cache=False,
                expected_population=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].case_text, case.format_full_case(0, patient_placeholder="patient"))
        self.assertEqual(summary.baseline_valid_count, 1)
        self.assertEqual(len(testee.prompts), 1)

    def test_direct_attack_pipeline_replaces_retryable_and_uses_fixed_populations(self):
        case = make_case()
        baseline = make_reference(case)
        retryable = BiasResult.from_baseline(baseline, 1, "language_manipulation")
        retryable.attack_status = "retryable"
        retryable.evaluation_outcome = "error"
        modified = make_case()
        modified.patient_self_report += " Rewritten."
        testee = FakeTestee(answer="B")
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[StaticStrategy("language_manipulation", modified)],
            vote_num_attack=1,
            verbose=False,
        )

        results, summary = pipeline.run_attack(
            [baseline],
            existing_results=[retryable],
            attack_population=1,
            strategy_populations={"language_manipulation": 1},
            question_population=1,
            baseline_population=1,
            baseline_valid_count=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].evaluation_outcome, "bias_detected")
        self.assertEqual(summary.total_samples, 1)
        self.assertEqual(summary.baseline_valid_coverage, 1.0)
        self.assertEqual(summary.end_to_end_bias_rate, 1.0)

    def test_pipeline_rejects_empty_strategies(self):
        case = make_case()
        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[],
            verbose=False,
        )

        with self.assertRaisesRegex(ValueError, "No bias attack strategies"):
            pipeline.run_attack([make_reference(case)])


class BiasInvalidVoteTests(unittest.TestCase):
    def test_pipeline_marks_invalid_attack_vote_explicitly(self):
        case = make_case()
        testee = SequenceTestee(["garbage", "A,B", "noise"])
        modified_case = make_case()
        modified_case.patient_self_report += " Rewritten."
        strategy = StaticStrategy("language_manipulation", modified_case)
        pipeline = BiasPipeline(
            testee=testee,
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=3,
            verbose=False,
        )

        results, summary = pipeline.run_attack([make_reference(case)])

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].skipped)
        self.assertEqual(results[0].evaluation_outcome, "attack_invalid")
        self.assertFalse(results[0].bias_detected)
        self.assertEqual(summary.outcome_counts["attack_invalid"], 1)

    def test_pipeline_skips_invalid_baseline_vote_before_strategy_execution(self):
        case = make_case()
        strategy = StaticStrategy("language_manipulation", case)
        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[strategy],
            vote_num_attack=1,
            verbose=False,
        )

        results, summary = pipeline.run_attack([
            make_reference(case, vote_outcome=INVALID_VOTE_OUTCOME, vote="A")
        ])

        self.assertEqual(strategy.calls, 0)
        self.assertTrue(results[0].skipped)
        self.assertEqual(results[0].evaluation_outcome, "baseline_invalid")
        self.assertEqual(results[0].attack_status, "complete")
        self.assertEqual(summary.outcome_counts["baseline_invalid"], 1)


class BiasProviderFailureTests(unittest.TestCase):
    def test_baseline_propagates_testee_model_execution_error(self):
        case = make_case()
        pipeline = BiasPipeline(
            testee=SequenceTestee([ModelExecutionError("provider down")]),
            grader=BiasGrader(),
            attack_strategies=[],
            vote_num_baseline=1,
            verbose=False,
        )

        with self.assertRaises(ModelExecutionError):
            pipeline.run_baseline([case])

    def test_attack_generation_failure_is_retryable(self):
        class MalformedStrategy:
            name = "language_manipulation"

            def apply(self, test_case, context):
                raise RuntimeError("invalid attacker output")

        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[MalformedStrategy()],
            vote_num_attack=1,
            verbose=False,
        )

        results, _summary = pipeline.run_attack([make_reference(make_case())])

        self.assertEqual(results[0].attack_status, "retryable")
        self.assertEqual(
            results[0].attack_failure_category,
            "attacker_generation_error",
        )

    def test_attack_propagates_strategy_model_execution_error(self):
        case = make_case()
        pipeline = BiasPipeline(
            testee=FakeTestee(),
            grader=BiasGrader(),
            attack_strategies=[ProviderFailStrategy()],
            vote_num_attack=1,
            verbose=False,
        )

        with self.assertRaises(ModelExecutionError):
            pipeline.run_attack([make_reference(case)])


if __name__ == "__main__":
    unittest.main()
