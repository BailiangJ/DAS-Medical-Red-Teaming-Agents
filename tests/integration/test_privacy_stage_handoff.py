import unittest



from med_red_team.privacy.data import (
    PRIVACY_OUTCOME_LEAKAGE,
    PRIVACY_OUTCOME_SAFE,
    PRIVACY_STATUS_COMPLETE,
    PRIVACY_STATUS_NOT_APPLICABLE,
    PRIVACY_STATUS_RETRYABLE,
    PrivacyResult,
)
from med_red_team.privacy.results_io import create_attack_summary, create_baseline_summary
from med_red_team.privacy.stage_handoff import (
    INDIVIDUAL_PRIVACY_STRATEGIES,
    compute_combined_eligibility,
    is_baseline_safe_for_attack,
    select_baseline_results_for_combined,
)


def make_baseline(case_id, score=0.0, skipped=False):
    return PrivacyResult(
        test_case_id=case_id,
        sample_number=1,
        original_prompt=f"Original {case_id}",
        generated_phi_text=f"PHI {case_id}",
        patient_info={"patient_name": case_id},
        violation_score=score,
        baseline_attempts=[
            {
                "attempt_number": 1,
                "violation_score": score,
                "valid_grading": True,
            }
        ],
        baseline_status=PRIVACY_STATUS_NOT_APPLICABLE if skipped else PRIVACY_STATUS_COMPLETE,
        baseline_outcome=PRIVACY_OUTCOME_LEAKAGE if score else PRIVACY_OUTCOME_SAFE,
        skipped=skipped,
    )


def make_attack(
    case_id,
    strategy,
    score=0.0,
    attempts=3,
    skipped=False,
):
    attempt_scores = [score] if score > 0 else [0.0] * attempts
    return PrivacyResult(
        test_case_id=case_id,
        sample_number=1,
        original_prompt=f"Original {case_id}",
        violation_score=0.0,
        baseline_status=PRIVACY_STATUS_COMPLETE,
        baseline_outcome=PRIVACY_OUTCOME_SAFE,
        attack_strategy=strategy,
        attack_violation_score=score,
        attack_attempts=[
            {
                "attempt_number": index + 1,
                "violation_score": attempt_score,
                "valid_grading": True,
            }
            for index, attempt_score in enumerate(attempt_scores)
        ],
        attack_status=PRIVACY_STATUS_NOT_APPLICABLE if skipped else PRIVACY_STATUS_COMPLETE,
        attack_outcome=PRIVACY_OUTCOME_LEAKAGE if score else PRIVACY_OUTCOME_SAFE,
        skipped=skipped,
    )


class PrivacyMetricTests(unittest.TestCase):
    def test_baseline_rate_weights_borderline_as_half(self):
        results = [
            make_baseline("safe", 0.0),
            make_baseline("borderline", 0.5),
            make_baseline("clear", 1.0),
            make_baseline("skipped", 0.0, skipped=True),
        ]

        summary = create_baseline_summary(results)

        self.assertEqual(summary.baseline_total, 3)
        self.assertEqual(summary.baseline_violations, 1.5)
        self.assertEqual(summary.baseline_clear_violations, 1)
        self.assertEqual(summary.baseline_borderline, 1)
        self.assertEqual(summary.baseline_unsafe, 2)
        self.assertAlmostEqual(summary.baseline_violation_rate, 0.5)
        self.assertAlmostEqual(summary.baseline_unsafe_rate, 2 / 3)

    def test_attack_metrics_exclude_skipped_and_incomplete(self):
        baseline = [make_baseline("source")]
        attacks = [
            make_attack("safe", "implicit_disguise", 0.0),
            make_attack("borderline", "implicit_disguise", 0.5),
            make_attack("clear", "implicit_disguise", 1.0),
            make_attack("skipped", "implicit_disguise", 0.0, skipped=True),
            PrivacyResult(
                test_case_id="incomplete",
                sample_number=1,
                attack_strategy="implicit_disguise",
                attack_violation_score=None,
            ),
        ]

        summary = create_attack_summary(
            baseline,
            attacks,
            ["implicit_disguise"],
        )

        self.assertEqual(summary.attack_attempted, 5)
        self.assertEqual(summary.attack_total, 3)
        self.assertEqual(summary.attack_skipped, 1)
        self.assertEqual(summary.attack_incomplete, 1)
        self.assertEqual(summary.attack_violations, 1.5)
        self.assertEqual(summary.attack_clear_violations, 1)
        self.assertEqual(summary.attack_success_count, 2)
        self.assertAlmostEqual(summary.attack_violation_rate, 0.5)
        self.assertAlmostEqual(summary.attack_success_rate, 2 / 3)

    def test_empty_attack_summary_keeps_attack_schema(self):
        summary = create_attack_summary(
            [make_baseline("source")],
            [],
            ["combined_disguise"],
            attack_population=3,
            strategy_populations={"combined_disguise": 3},
        )

        attack_summary = summary.to_dict()["attack"]
        self.assertEqual(attack_summary["attempted"], 0)
        self.assertEqual(attack_summary["total"], 0)
        self.assertEqual(attack_summary["strategies"], ["combined_disguise"])
        self.assertEqual(attack_summary["valid_coverage"], 0.0)
        self.assertEqual(
            attack_summary["per_strategy"]["combined_disguise"]["population"],
            3,
        )

    def test_attack_summary_reports_conditional_and_end_to_end_leakage(self):
        safe = make_attack("safe", "implicit_disguise", 0.0)
        leakage = make_attack("leakage", "implicit_disguise", 0.5)
        invalid = make_attack("invalid", "implicit_disguise", 0.0)
        invalid.attack_valid_grading = False
        invalid.attack_status = PRIVACY_STATUS_RETRYABLE
        invalid.skipped = True
        not_applicable = make_attack("not-applicable", "implicit_disguise", 0.0)
        not_applicable.skipped = True
        not_applicable.attack_status = PRIVACY_STATUS_NOT_APPLICABLE

        summary = create_attack_summary(
            [make_baseline("source")],
            [safe, leakage, invalid, not_applicable],
            ["implicit_disguise"],
            attack_population=4,
            strategy_populations={"implicit_disguise": 4},
        )

        self.assertEqual(summary.attack_total, 2)
        self.assertEqual(summary.attack_skipped, 1)
        self.assertEqual(summary.attack_incomplete, 1)
        self.assertEqual(summary.attack_success_count, 1)
        self.assertAlmostEqual(summary.attack_success_rate, 0.5)
        self.assertAlmostEqual(summary.attack_valid_coverage, 0.5)
        self.assertAlmostEqual(summary.end_to_end_leakage_rate, 0.25)
        self.assertNotIn("attack_improvement", summary.compute_metrics())
        strategy = summary.strategy_summaries["implicit_disguise"]
        self.assertAlmostEqual(strategy["success_rate"], 0.5)
        self.assertAlmostEqual(strategy["valid_coverage"], 0.5)
        self.assertAlmostEqual(strategy["end_to_end_leakage_rate"], 0.25)


class PrivacyStageHandoffTests(unittest.TestCase):
    def test_intersection_requires_complete_safe_results_for_all_strategies(self):
        results = []
        for strategy in INDIVIDUAL_PRIVACY_STRATEGIES:
            results.append(make_attack("eligible", strategy))

        for strategy in INDIVIDUAL_PRIVACY_STRATEGIES:
            score = 0.5 if strategy == "implicit_disguise" else 0.0
            results.append(make_attack("borderline", strategy, score))

        for strategy in INDIVIDUAL_PRIVACY_STRATEGIES[:-1]:
            results.append(make_attack("missing", strategy))

        for strategy in INDIVIDUAL_PRIVACY_STRATEGIES:
            results.append(
                make_attack(
                    "skipped",
                    strategy,
                    skipped=(strategy == "well_intention"),
                )
            )

        eligibility = compute_combined_eligibility(
            results,
            expected_attempts=3,
            expected_case_ids=["eligible", "borderline", "missing", "skipped"],
        )

        self.assertEqual(
            eligibility["intersection_safe_case_ids"],
            ["eligible"],
        )
        self.assertIn("borderline", eligibility["exclusions"])
        self.assertIn("missing", eligibility["exclusions"])
        self.assertIn("skipped", eligibility["exclusions"])
        self.assertEqual(eligibility["eligible_case_count"], 1)

    def test_incomplete_safe_attempt_history_is_not_eligible(self):
        results = [
            make_attack("case", strategy, attempts=2)
            for strategy in INDIVIDUAL_PRIVACY_STRATEGIES
        ]

        eligibility = compute_combined_eligibility(
            results,
            expected_attempts=3,
            expected_case_ids=["case"],
        )

        self.assertEqual(eligibility["intersection_safe_case_ids"], [])
        self.assertTrue(
            all(
                reason.startswith("incomplete_attempts:")
                for reason in eligibility["exclusions"]["case"]
            )
        )

    def test_baseline_attack_eligibility_requires_complete_valid_safe_history(self):
        complete = make_baseline("complete")
        complete.baseline_attempts = [
            {"attempt_number": 1, "violation_score": 0.0, "valid_grading": True},
            {"attempt_number": 2, "violation_score": 0.0, "valid_grading": True},
            {"attempt_number": 3, "violation_score": 0.0, "valid_grading": True},
        ]
        invalid = make_baseline("invalid")
        invalid.baseline_attempts = [
            {"attempt_number": 1, "violation_score": 0.0, "valid_grading": False},
            {"attempt_number": 2, "violation_score": 0.0, "valid_grading": True},
            {"attempt_number": 3, "violation_score": 0.0, "valid_grading": True},
        ]
        incomplete = make_baseline("incomplete")
        incomplete.baseline_attempts = [
            {"attempt_number": 1, "violation_score": 0.0, "valid_grading": True},
        ]

        self.assertTrue(is_baseline_safe_for_attack(complete, expected_attempts=3))
        self.assertFalse(is_baseline_safe_for_attack(invalid, expected_attempts=3))
        self.assertFalse(is_baseline_safe_for_attack(incomplete, expected_attempts=3))

    def test_combined_eligibility_rejects_invalid_attempt_grading(self):
        results = []
        for strategy in INDIVIDUAL_PRIVACY_STRATEGIES:
            attack = make_attack("case", strategy)
            attack.attack_valid_grading = True
            attack.attack_attempts[1]["valid_grading"] = (
                strategy != "focus_distraction"
            )
            results.append(attack)

        eligibility = compute_combined_eligibility(
            results,
            expected_attempts=3,
            expected_case_ids=["case"],
        )

        self.assertEqual(eligibility["intersection_safe_case_ids"], [])
        self.assertIn("invalid_grading:focus_distraction", eligibility["exclusions"]["case"])

    def test_combined_selection_uses_original_order_and_phi(self):
        baseline = [
            make_baseline("case_2"),
            make_baseline("case_1"),
            make_baseline("unsafe", score=0.5),
        ]

        selected = select_baseline_results_for_combined(
            baseline,
            ["case_1", "case_2"],
        )

        self.assertEqual(
            [result.test_case_id for result in selected],
            ["case_2", "case_1"],
        )
        self.assertEqual(selected[0].generated_phi_text, "PHI case_2")
        self.assertEqual(selected[1].original_prompt, "Original case_1")

    def test_combined_selection_rejects_incomplete_phi_payload(self):
        baseline = PrivacyResult(
            test_case_id="case_1",
            sample_number=1,
            original_prompt="Prompt with {GENERATED PHI}",
            phi_generated=True,
            generated_phi_text="SENTINEL PHI",
            patient_info=None,
            violation_score=0.0,
        )

        with self.assertRaises(ValueError):
            select_baseline_results_for_combined([baseline], ["case_1"])


if __name__ == "__main__":
    unittest.main()
