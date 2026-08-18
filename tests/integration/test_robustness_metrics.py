import unittest

from med_red_team.robustness.data import (
    RobustnessResult,
    summarize_robustness_results,
)
from med_red_team.shared.checkpoints import merge_unique


class RobustnessMetricTests(unittest.TestCase):
    def _result(self, index, status, manipulated_correct=None, valid=False):
        return RobustnessResult(
            sample_number=index,
            test_case_id=f"case-{index}",
            original_correct=True,
            eligible_baseline_correct=True,
            manipulated_correct=manipulated_correct,
            valid_target_tested=valid,
            manipulation_failed=not valid,
            skipped=not valid,
            status=status,
        )

    def test_dual_metrics_keep_failures_visible(self):
        results = [
            self._result(1, "fooled", False, True),
            self._result(2, "fooled", False, True),
            self._result(3, "target_safe", True, True),
            self._result(4, "no_valid_plan"),
            self._result(5, "invalid_noop"),
        ]
        results[3].planner_conflict_rejections = 2
        summary = summarize_robustness_results(
            results,
            evaluation_type="robustness_attack",
        )
        self.assertEqual(summary.eligible_baseline_correct, 5)
        self.assertEqual(summary.valid_target_tested, 3)
        self.assertEqual(summary.fooled, 2)
        self.assertEqual(summary.target_safe, 1)
        self.assertAlmostEqual(summary.attack_coverage, 3 / 5)
        self.assertAlmostEqual(summary.conditional_attack_success_rate, 2 / 3)
        self.assertAlmostEqual(summary.conditional_robustness, 1 / 3)
        self.assertAlmostEqual(summary.end_to_end_attack_success_rate, 2 / 5)
        self.assertEqual(summary.planner_conflict_rejections, 2)
        self.assertEqual(summary.no_valid_plan, 1)
        self.assertEqual(summary.invalid_noop, 1)
        self.assertEqual(summary.failed_manipulations, 2)

    def test_failed_manipulations_exclude_parser_and_provider_failures(self):
        results = [
            self._result(1, "no_valid_plan"),
            self._result(2, "invalid_noop"),
            self._result(3, "not_applicable"),
            self._result(4, "parser_error"),
            self._result(5, "generation_error"),
            self._result(6, "tool_error"),
            self._result(7, "invalid_state"),
        ]
        summary = summarize_robustness_results(
            results,
            evaluation_type="robustness_attack",
        )

        self.assertEqual(summary.failed_manipulations, 3)
        self.assertEqual(summary.parser_errors, 1)
        self.assertEqual(summary.generation_errors, 1)
        self.assertEqual(summary.tool_errors, 2)

    def test_attack_baseline_accuracy_uses_all_valid_baseline_rows(self):
        results = [
            self._result(index, "target_safe", True, True)
            for index in range(1, 8)
        ]
        results.extend(
            self._result(index, "invalid_noop")
            for index in range(8, 11)
        )
        summary = summarize_robustness_results(
            results,
            evaluation_type="robustness_attack",
        )

        self.assertEqual(summary.baseline_evaluated, 10)
        self.assertEqual(summary.first_round_correct, 10)
        self.assertEqual(summary.first_round_accuracy, 1.0)

    def test_baseline_errors_do_not_enter_accuracy_denominator(self):
        result = RobustnessResult(
            sample_number=1,
            test_case_id="parser-error",
            original_correct=False,
            skipped=True,
            status="parser_error",
        )
        summary = summarize_robustness_results(
            [result],
            evaluation_type="robustness_baseline",
        )
        self.assertEqual(summary.baseline_evaluated, 0)
        self.assertEqual(summary.first_round_correct, 0)
        self.assertEqual(summary.first_round_accuracy, 0.0)

    def test_baseline_summary_has_no_failed_manipulations(self):
        result = RobustnessResult(
            sample_number=1,
            test_case_id="baseline",
            original_correct=True,
            status="baseline_correct",
        )
        summary = summarize_robustness_results(
            [result],
            evaluation_type="robustness_baseline",
        )
        self.assertEqual(summary.failed_manipulations, 0)

    def test_result_status_requires_explicit_v2_fields(self):
        result = RobustnessResult.from_dict({
            "sample_number": 1,
            "test_case_id": "case",
            "original_correct": True,
            "manipulated_correct": False,
            "status": "fooled",
            "eligible_baseline_correct": True,
            "valid_target_tested": True,
        })
        missing_status = RobustnessResult.from_dict({
            "sample_number": 2,
            "test_case_id": "missing-status",
            "original_correct": True,
            "manipulated_correct": False,
        })

        self.assertEqual(result.status, "fooled")
        self.assertTrue(result.eligible_baseline_correct)
        self.assertTrue(result.valid_target_tested)
        self.assertEqual(missing_status.status, "unknown")
        self.assertFalse(missing_status.eligible_baseline_correct)
        self.assertFalse(missing_status.valid_target_tested)

    def test_summary_serializes_metadata_and_new_metrics(self):
        summary = summarize_robustness_results(
            [self._result(1, "target_safe", True, True)],
            evaluation_type="robustness_attack",
            metadata={"policy": "1"},
        )
        payload = summary.to_dict()
        self.assertEqual(payload["metadata"], {"policy": "1"})
        self.assertEqual(payload["valid_target_tested"], 1)
        self.assertEqual(payload["conditional_robustness"], 1.0)

    def test_zero_valid_targets_is_explicit_in_score_note(self):
        summary = summarize_robustness_results(
            [self._result(1, "parser_error")],
            evaluation_type="robustness_attack",
        )
        self.assertEqual(summary.valid_target_tested, 0)
        payload = summary.to_dict()
        self.assertIsNone(payload["robustness_score"])
        self.assertIn("No valid target-tested attacks", payload["score_note"])

    def test_replay_summary_uses_fixed_eligible_population(self):
        summary = summarize_robustness_results(
            [self._result(1, "fooled", False, True)],
            evaluation_type="robustness_attack",
            expected_eligible_population=4,
        )

        self.assertEqual(summary.total_samples, 1)
        self.assertEqual(summary.eligible_baseline_correct, 4)
        self.assertEqual(summary.valid_target_tested, 1)
        self.assertEqual(summary.attack_coverage, 0.25)
        self.assertEqual(summary.conditional_attack_success_rate, 1.0)
        self.assertEqual(summary.end_to_end_attack_success_rate, 0.25)

    def test_result_round_trip_preserves_complete_manipulated_payload(self):
        result = RobustnessResult(
            sample_number=1,
            test_case_id="changed-options",
            original_question="Original?",
            original_options={"A": "alpha", "B": "beta"},
            original_correct_answer="A",
            original_response="A",
            original_correct=True,
            manipulated_question="Changed?",
            manipulated_options={"B": "new beta", "A": "new alpha", "C": "gamma"},
            manipulated_correct_answer="B",
            manipulated_response="B",
            manipulated_correct=True,
            attacks_applied=["generate_distractor_options"],
            status="target_safe",
            eligible_baseline_correct=True,
            valid_target_tested=True,
        )

        restored = RobustnessResult.from_dict(result.to_dict())

        self.assertEqual(restored.manipulated_question, "Changed?")
        self.assertEqual(
            list(restored.manipulated_options.items()),
            [("B", "new beta"), ("A", "new alpha"), ("C", "gamma")],
        )
        self.assertEqual(restored.manipulated_correct_answer, "B")
        self.assertEqual(restored.manipulated_response, "B")
        self.assertTrue(restored.manipulated_correct)


class CheckpointTests(unittest.TestCase):
    def test_axis_identity_merges_existing_without_duplicate_ids(self):
        existing = RobustnessResult(
            sample_number=1,
            test_case_id="existing",
            status="target_safe",
        )
        replacement = RobustnessResult(
            sample_number=2,
            test_case_id="existing",
            status="fooled",
        )
        new = RobustnessResult(
            sample_number=3,
            test_case_id="new",
            status="target_safe",
        )

        merged = merge_unique(
            [existing],
            [replacement, new],
            key=lambda result: result.test_case_id,
        )

        statuses = {row.test_case_id: row.status for row in merged}
        self.assertEqual(statuses, {"existing": "fooled", "new": "target_safe"})


if __name__ == "__main__":
    unittest.main()
