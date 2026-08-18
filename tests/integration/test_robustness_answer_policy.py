import unittest



from med_red_team.models import ModelResponse
from med_red_team.models.utils.response_utils import (
    extract_answer_with_quality_check,
    parse_answer_label_set,
)
from med_red_team.data import TestCase as FrameworkTestCase
from med_red_team.grader import SimpleGrader
from med_red_team.robustness.attacker import (
    AddDistractionSentenceStrategy,
    BiasManipulationStrategy,
    GenerateDistractorOptionsStrategy,
    ReplaceCorrectAnswerStrategy,
)
from med_red_team.attacker_registry import STRATEGY_REGISTRY
from med_red_team.robustness.data import OrchestratorQuestionData
from med_red_team.robustness.tool_policy import (
    check_question_state,
    compute_question_delta,
    normalize_strategy_sequence,
    normalize_tool_sequence,
)


class FakeModel:
    def __init__(self):
        self.last_user_prompt = None

    def generate(self, user_prompt, system_prompt=None, config=None):
        self.last_user_prompt = user_prompt
        return ModelResponse(
            raw_text="",
            model_id="fake",
            final_answer=(
                '{"bias_styles": ["confirmation"], '
                '"modified_question": "Biased question?"}'
            )
        )


class FakeModelPool:
    def __init__(self):
        self.model = FakeModel()

    def get_model(self, model_id, **kwargs):
        return self.model


class BrokenModelPool:
    def get_model(self, model_id, **kwargs):
        class BrokenModel:
            def generate(self, user_prompt, system_prompt=None, config=None):
                return ModelResponse(
                    raw_text="not a numbered distractor list",
                    model_id="fake",
                    final_answer="not a numbered distractor list",
                )

        return BrokenModel()


class AnswerSetTests(unittest.TestCase):
    def test_testcase_and_grader_compare_answer_sets(self):
        case = FrameworkTestCase(
            id="case",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="D, B, C",
        )
        self.assertEqual(case.correct_answer, "B,C,D")
        grade = SimpleGrader().grade(case, "I think D, B and C", {})
        self.assertTrue(grade.is_correct)
        self.assertEqual(grade.metadata["extracted_answer"], "B,C,D")

    def test_extraction_respects_actual_option_labels(self):
        extracted, _, _ = extract_answer_with_quality_check(
            "I think A",
            valid_labels=["A", "B", "C", "D"],
        )
        self.assertEqual(extracted, "A")

    def test_lowercase_prose_is_not_promoted_to_answer_label(self):
        extracted, _, _ = extract_answer_with_quality_check(
            "This is a guess",
            valid_labels=["A", "B", "C", "D"],
        )
        self.assertEqual(extracted, "")

    def test_lowercase_option_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            FrameworkTestCase(
                id="lowercase",
                question="Question?",
                options={"a": "a", "b": "b"},
                correct_answer="a",
            )

    def test_invalid_correct_label_fails_loudly(self):
        with self.assertRaises(ValueError):
            FrameworkTestCase(
                id="invalid",
                question="Question?",
                options={"A": "a", "B": "b"},
                correct_answer="A,Z",
            )

    def test_replace_none_rejects_multi_answer(self):
        case = FrameworkTestCase(
            id="case",
            question="Question?",
            options={"A": "a", "B": "b", "C": "c"},
            correct_answer="B,C",
        )
        modified = ReplaceCorrectAnswerStrategy().apply(case, {})
        self.assertEqual(modified.options, case.options)
        self.assertTrue(modified.metadata["manipulation_failed"])
        self.assertEqual(modified.metadata["failure_category"], "not_applicable")

    def test_bias_targets_only_label_outside_multi_answer_set(self):
        case = FrameworkTestCase(
            id="case",
            question="Inverted question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="B,C,D",
        )
        pool = FakeModelPool()
        strategy = BiasManipulationStrategy(
            model_id="fake",
            model_pool=pool,
            n_bias_styles=1,
        )
        modified = strategy.apply(case, {"incorrect_option_label": "B"})
        self.assertEqual(
            modified.metadata["attack_details"]["bias_manipulation"][
                "incorrect_option_label"
            ],
            "A",
        )
        self.assertEqual(
            parse_answer_label_set(modified.correct_answer, modified.options),
            frozenset({"B", "C", "D"}),
        )

    def test_unparseable_distractors_are_marked_as_failed_attack(self):
        case = FrameworkTestCase(
            id="case",
            question="Question?",
            options={"A": "a", "B": "b"},
            correct_answer="A",
        )
        strategy = GenerateDistractorOptionsStrategy(
            model_id="fake",
            model_pool=BrokenModelPool(),
            num_distractors=2,
        )
        modified = strategy.apply(case, {})
        self.assertEqual(modified.options, case.options)
        self.assertTrue(modified.metadata["manipulation_failed"])
        self.assertEqual(modified.metadata["failure_category"], "parser_error")


class DistractionInsertionTests(unittest.TestCase):
    def setUp(self):
        self.strategy = AddDistractionSentenceStrategy(
            model_id="fake",
            model_pool=FakeModelPool(),
        )

    @staticmethod
    def _remove_insertion(original, modified, distraction):
        marker = f" {distraction}"
        self_removed = modified.replace(marker, "", 1)
        return self_removed

    def test_decimal_measurements_are_preserved(self):
        original = "The patient has a temperature of 38.4°C. What is the diagnosis?"
        distraction = "The patient's dog has diabetes."
        modified = self.strategy._insert_distraction(original, distraction)

        self.assertIn("38.4°C", modified)
        self.assertEqual(self._remove_insertion(original, modified, distraction), original)
        self.assertLess(modified.index(distraction), modified.rindex("What is the diagnosis?"))

    def test_abbreviations_are_preserved(self):
        original = "Dr. Smith reviewed e.g. prior cases. What is the diagnosis?"
        distraction = "A nearby case involved seasonal allergies."
        modified = self.strategy._insert_distraction(original, distraction)

        self.assertEqual(self._remove_insertion(original, modified, distraction), original)
        self.assertIn("Dr. Smith reviewed e.g. prior cases.", modified)

    def test_single_sentence_falls_back_to_prefix_without_rewriting(self):
        original = "Which dose is 2.5 mg?"
        distraction = "The patient also mentioned mild fatigue."
        modified = self.strategy._insert_distraction(original, distraction)

        self.assertTrue(modified.startswith(distraction + " "))
        self.assertEqual(modified[len(distraction) + 1 :], original)


class ToolPolicyTests(unittest.TestCase):
    def test_inversion_bias_and_distraction_are_ordered_not_conflicted(self):
        decision = normalize_tool_sequence([
            "introduce_cognitive_bias_tool",
            "add_distraction_sentence_tool",
            "invert_final_question_and_answer_tool",
            "generate_distractor_options_tool",
        ])
        self.assertTrue(decision.valid)
        self.assertEqual(
            decision.normalized_sequence,
            [
                "generate_distractor_options_tool",
                "invert_final_question_and_answer_tool",
                "introduce_cognitive_bias_tool",
                "add_distraction_sentence_tool",
            ],
        )

    def test_absolute_conflict_is_rejected(self):
        decision = normalize_tool_sequence([
            "invert_final_question_and_answer_tool",
            "replace_correct_answer_to_none_of_the_options_are_correct_tool",
        ])
        self.assertFalse(decision.valid)
        self.assertEqual(decision.issues[0].issue_type, "absolute_conflict")

    def test_unknown_and_duplicate_tools_are_rejected(self):
        unknown = normalize_tool_sequence(["not_a_tool"])
        duplicate = normalize_tool_sequence([
            "introduce_cognitive_bias_tool",
            "introduce_cognitive_bias_tool",
        ])
        self.assertFalse(unknown.valid)
        self.assertFalse(duplicate.valid)

    def test_add_none_and_replace_none_conflict_in_fixed_chain(self):
        decision = normalize_strategy_sequence([
            "add_none_of_the_above",
            "replace_correct_answer_with_none",
        ])
        self.assertFalse(decision.valid)
        self.assertEqual(decision.issues[0].issue_type, "absolute_conflict")

    def test_canonical_strategy_names_are_registry_and_policy_names(self):
        canonical = [
            "add_none_of_the_above",
            "replace_correct_answer_with_none",
            "add_distraction_sentence",
            "generate_distractor_options",
            "bias_manipulation",
            "invert_question_answer",
            "adjust_impossible_measurement",
        ]
        self.assertTrue(set(canonical).issubset(STRATEGY_REGISTRY))
        self.assertTrue(
            all(normalize_strategy_sequence([name]).valid for name in canonical)
        )
        self.assertFalse(normalize_strategy_sequence(["bias_manipulation_question"]).valid)
        self.assertFalse(
            normalize_tool_sequence(["add_none_of_the_options_are_correct_tool"]).valid
        )

    def test_question_state_rejects_empty_options(self):
        question = OrchestratorQuestionData(
            question="Question?",
            options={"A": "answer", "B": ""},
            answer_idx="A",
        )
        state = check_question_state(question)
        self.assertFalse(state.valid)
        self.assertEqual(state.status, "invalid_state")

    def test_question_delta_is_bounded_and_structured(self):
        before = OrchestratorQuestionData(
            question="Original question?",
            options={"A": "a", "B": "b"},
            answer_idx="A",
        )
        after = OrchestratorQuestionData(
            question="X" * 500 + " Modified question?",
            options={"A": "a", "B": "changed", "C": "new"},
            answer_idx="B,C",
        )
        delta = compute_question_delta(before, after)
        self.assertTrue(delta["question"]["changed"])
        self.assertIn("C", delta["options"]["added"])
        self.assertIn("B", delta["options"]["modified"])
        self.assertEqual(delta["answer"]["after"], ["B", "C"])
        for edit in delta["question"]["edits"]:
            self.assertLessEqual(len(edit["before"]), 240)
            self.assertLessEqual(len(edit["after"]), 240)


if __name__ == "__main__":
    unittest.main()
