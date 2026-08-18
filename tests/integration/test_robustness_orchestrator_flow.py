import json
import unittest
from unittest.mock import patch

from tests.fakes.optional_agents import install_fake_agents

install_fake_agents()

from med_red_team.models import ModelResponse
from med_red_team.data import TestCase as FrameworkTestCase
from med_red_team.grader import SimpleGrader
from med_red_team.robustness.data import (
    OrchestratorPlanOutput,
    OrchestratorQuestionData,
    ToolExecutionLog,
)
from med_red_team.robustness.orchestrator_pipeline import OrchestratorPipeline
from med_red_team.robustness.tool_policy import (
    compute_question_delta,
    normalize_tool_sequence,
)


class FakeTestee:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def answer(self, prompt):
        self.calls.append(prompt)
        return ModelResponse(
            raw_text="",
            model_id="fake",
            final_answer=self.answers.pop(0),
        )


class FakeModelPool:
    def get_model(self, model_id, **kwargs):
        return object()


class FakeToolRegistry:
    first_input_by_round = []

    def __init__(self, *args, **kwargs):
        pass

    def validate_and_normalize_sequence(self, sequence):
        return normalize_tool_sequence(sequence)

    def execute_tool(self, tool_name, question_data, context=None):
        before = question_data.model_copy(deep=True)
        if before.question == "Original question?":
            self.first_input_by_round.append(before.question)

        if tool_name == "invert_final_question_and_answer_tool":
            after = OrchestratorQuestionData(
                question=f"Inverted: {before.question}",
                options=before.options,
                answer_idx="B,C,D",
            )
        elif tool_name == "introduce_cognitive_bias_tool":
            after = OrchestratorQuestionData(
                question=f"{before.question} Bias sentence.",
                options=before.options,
                answer_idx=before.answer_idx,
            )
        else:
            after = OrchestratorQuestionData(
                question=f"{before.question} Distraction sentence.",
                options=before.options,
                answer_idx=before.answer_idx,
            )

        log = ToolExecutionLog(
            tool_name=tool_name,
            input_question=before.question,
            output_question=after.question,
            input_options=before.options,
            output_options=after.options,
            input_answer=before.answer_idx,
            output_answer=after.answer_idx,
            tool_metadata={},
            execution_time_ms=1.0,
            success=True,
            delta=compute_question_delta(before, after),
        )
        return after, log


class FakePlannerPipeline(OrchestratorPipeline):
    def __init__(self, plans, *args, **kwargs):
        self._plans = list(plans)
        self.planner_inputs = []
        super().__init__(*args, **kwargs)

    def _run_planner(self, input_items):
        self.planner_inputs.append(input_items)
        plan = self._plans.pop(0)
        if isinstance(plan, Exception):
            raise plan
        return plan


class OrchestratorFlowTests(unittest.TestCase):
    def setUp(self):
        FakeToolRegistry.first_input_by_round = []
        self.case = FrameworkTestCase(
            id="case",
            question="Original question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
        )
        self.baseline = {
            "results": [{
                "test_case_id": "case",
                "original_response": "A",
                "original_correct": True,
            }]
        }

    def _pipeline(self, plans, answers, max_iterations=2, max_planner_attempts=2):
        return FakePlannerPipeline(
            plans=plans,
            testee=FakeTestee(answers),
            grader=SimpleGrader(),
            orchestrator_model_id="fake-planner",
            tools_model_id="fake-tools",
            model_pool=FakeModelPool(),
            max_iterations=max_iterations,
            max_planner_attempts=max_planner_attempts,
            baseline_results=self.baseline,
            verbose=False,
        )

    @patch(
        "med_red_team.robustness.orchestrator_pipeline.ToolRegistry",
        FakeToolRegistry,
    )
    def test_conflict_replans_then_uses_original_seed_each_round(self):
        plans = [
            OrchestratorPlanOutput(
                tool_sequence=[
                    "invert_final_question_and_answer_tool",
                    "replace_correct_answer_to_none_of_the_options_are_correct_tool",
                ],
                reasoning="Conflicting first proposal",
            ),
            OrchestratorPlanOutput(
                tool_sequence=[
                    "introduce_cognitive_bias_tool",
                    "invert_final_question_and_answer_tool",
                ],
                reasoning="Valid first round",
            ),
            OrchestratorPlanOutput(
                tool_sequence=["add_distraction_sentence_tool"],
                reasoning="Second round",
            ),
        ]
        pipeline = self._pipeline(plans, answers=["B,C,D", "B"])
        results, summary = pipeline.run_orchestrator_attack(
            [self.case],
            skip_first_round=True,
        )

        result = results[0]
        self.assertEqual(result.status, "fooled")
        self.assertEqual(result.planner_attempts_used, 3)
        self.assertEqual(result.planner_conflict_rejections, 1)
        self.assertEqual(result.target_tested_iterations, 2)
        self.assertEqual(len(pipeline.testee.calls), 2)
        self.assertEqual(
            FakeToolRegistry.first_input_by_round,
            ["Original question?", "Original question?"],
        )
        self.assertEqual(summary.valid_target_tested, 1)
        self.assertEqual(summary.fooled, 1)

        second_round_payload = json.loads(
            pipeline.planner_inputs[-1][0]["content"]
        )
        self.assertEqual(second_round_payload["attack_iteration"], 2)
        self.assertEqual(
            second_round_payload["original_question_sample"]["question"],
            "Original question?",
        )
        previous = second_round_payload["prior_independent_attempts"][0]
        self.assertTrue(previous["target_outcome"]["remained_correct"])
        self.assertEqual(
            previous["normalized_sequence"][0],
            "invert_final_question_and_answer_tool",
        )

    @patch(
        "med_red_team.robustness.orchestrator_pipeline.ToolRegistry",
        FakeToolRegistry,
    )
    def test_later_planner_error_preserves_prior_target_safe_result(self):
        valid = OrchestratorPlanOutput(
            tool_sequence=["add_distraction_sentence_tool"],
            reasoning="Valid first round",
        )
        pipeline = self._pipeline(
            [valid, RuntimeError("planner down"), RuntimeError("planner down")],
            answers=["A"],
            max_iterations=2,
            max_planner_attempts=2,
        )
        results, summary = pipeline.run_orchestrator_attack(
            [self.case],
            skip_first_round=True,
        )
        result = results[0]
        self.assertEqual(result.status, "target_safe")
        self.assertTrue(result.valid_target_tested)
        self.assertFalse(result.skipped)
        self.assertEqual(result.planner_conflict_rejections, 0)
        self.assertEqual(result.metadata["planner_errors_total"], 2)
        self.assertEqual(summary.target_safe, 1)

    @patch(
        "med_red_team.robustness.orchestrator_pipeline.ToolRegistry",
        FakeToolRegistry,
    )
    def test_max_samples_counts_eligible_attack_targets(self):
        case_two = FrameworkTestCase(
            id="case-two",
            question="Second original question?",
            options={"A": "a", "B": "b", "C": "c", "D": "d"},
            correct_answer="A",
        )
        baseline = {
            "results": [
                {
                    "test_case_id": "case",
                    "original_response": "A",
                    "original_correct": True,
                },
                {
                    "test_case_id": "case-two",
                    "original_response": "A",
                    "original_correct": True,
                },
            ]
        }
        plan = OrchestratorPlanOutput(
            tool_sequence=["add_distraction_sentence_tool"],
            reasoning="One target",
        )
        pipeline = FakePlannerPipeline(
            plans=[plan],
            testee=FakeTestee(["B"]),
            grader=SimpleGrader(),
            orchestrator_model_id="fake-planner",
            tools_model_id="fake-tools",
            model_pool=FakeModelPool(),
            max_iterations=1,
            max_planner_attempts=2,
            baseline_results=baseline,
            verbose=False,
        )
        results, _ = pipeline.run_orchestrator_attack(
            [self.case, case_two],
            max_samples=1,
            skip_first_round=True,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(len(pipeline.testee.calls), 1)

    @patch(
        "med_red_team.robustness.orchestrator_pipeline.ToolRegistry",
        FakeToolRegistry,
    )
    def test_planner_attempt_limit_prevents_target_call(self):
        conflict = OrchestratorPlanOutput(
            tool_sequence=[
                "invert_final_question_and_answer_tool",
                "replace_correct_answer_to_none_of_the_options_are_correct_tool",
            ],
            reasoning="Conflict",
        )
        pipeline = self._pipeline(
            [conflict, conflict],
            answers=[],
            max_iterations=1,
            max_planner_attempts=2,
        )
        results, summary = pipeline.run_orchestrator_attack(
            [self.case],
            skip_first_round=True,
        )

        result = results[0]
        self.assertEqual(result.status, "no_valid_plan")
        self.assertEqual(result.planner_attempts_used, 2)
        self.assertEqual(result.planner_conflict_rejections, 2)
        self.assertFalse(result.valid_target_tested)
        self.assertEqual(len(pipeline.testee.calls), 0)
        self.assertEqual(summary.no_valid_plan, 1)
        self.assertEqual(summary.attack_coverage, 0.0)


if __name__ == "__main__":
    unittest.main()
