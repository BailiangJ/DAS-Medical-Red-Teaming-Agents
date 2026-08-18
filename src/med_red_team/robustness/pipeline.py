"""
Robustness Evaluation Pipeline
==============================

This module implements robustness-specific evaluation by inheriting from
the abstract base classes.

Robustness evaluation tests model resilience to adversarial manipulations:
1. Test with original question (first round)
2. Apply attack strategies to correctly answered questions
3. Test with manipulated question (second round)
4. Compare first round vs second round accuracy
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from med_red_team.data import TestCase, EvaluationResult, EvaluationSummary
from med_red_team.testee import Testee
from med_red_team.grader import Grader
from med_red_team.actors import AttackStrategy
from med_red_team.pipeline import EvaluationPipeline
from med_red_team.models import MalformedResponseError, ModelExecutionError
from med_red_team.models.utils.response_utils import get_parse_error_info, has_parse_error
from med_red_team.robustness.data import (
    OrchestratorQuestionData,
    RobustnessReplayItem,
    RobustnessResult,
    RobustnessSummary,
    is_baseline_result_eligible_for_attack,
    summarize_robustness_results,
)
from med_red_team.robustness.tool_policy import (
    STRATEGY_NAME_TO_TOOL_NAME,
    check_tool_postconditions,
    check_tool_preconditions,
    normalize_strategy_sequence,
)


# ==============================================================================
# Robustness Evaluation Pipeline
# ==============================================================================

class RobustnessPipeline(EvaluationPipeline):
    """
    Robustness evaluation pipeline with separated baseline and attack phases.

    This pipeline supports two-phase evaluation:
    1. Baseline phase: Test original questions, save results
    2. Attack phase: Load baseline results, apply attacks, test again

    Example (Baseline):
        >>> pipeline = RobustnesssPipeline(
        ...     testee=testee,
        ...     grader=grader,
        ...     attack_strategies=[],  # Empty for baseline
        ...     verbose=True
        ... )
        >>> results, summary = pipeline.run_baseline(
        ...     test_cases,
        ...     max_samples=100,
        ...     progress_callback=callback,
        ...     save_every=10
        ... )

    Example (Attack):
        >>> pipeline = RobustnessPipeline(
        ...     testee=testee,
        ...     grader=grader,
        ...     attack_strategies=[bias_attack, distraction_attack],
        ...     verbose=True
        ... )
        >>> # Load baseline results
        >>> baseline_results = load_baseline_results("baseline.json")
        >>> results, summary = pipeline.run_attack(
        ...     baseline_results,
        ...     progress_callback=callback,
        ...     save_every=10
        ... )
    """
    
    def __init__(
        self,
        testee: Optional[Testee] = None,
        grader: Optional[Grader] = None,
        attack_strategies: List[AttackStrategy] = None,
        verbose: bool = True
    ):
        """
        Initialize robustness evaluation pipeline.

        Args:
            testee: Model to evaluate (optional, not needed for attack generation)
            grader: Grader to assess responses (optional, not needed for attack generation)
            attack_strategies: List of attack strategies (empty list for baseline-only)
            verbose: Whether to print progress
        """
        # Only call super if testee/grader provided (for evaluation mode)
        if testee is not None and grader is not None:
            super().__init__(testee, grader, verbose)
        else:
            # Generation mode - no evaluation needed
            self.testee = testee
            self.grader = grader
            self.verbose = verbose

        proposed_strategies = attack_strategies or []
        self.tool_policy_decision = None
        if proposed_strategies:
            self.tool_policy_decision = normalize_strategy_sequence(
                [strategy.name for strategy in proposed_strategies]
            )
            if not self.tool_policy_decision.valid:
                reasons = "; ".join(
                    issue.message for issue in self.tool_policy_decision.issues
                )
                raise ValueError(f"Invalid robustness strategy chain: {reasons}")

            strategy_by_name = {
                strategy.name: strategy for strategy in proposed_strategies
            }
            self.attack_strategies = [
                strategy_by_name[name]
                for name in self.tool_policy_decision.normalized_sequence
            ]
        else:
            self.attack_strategies = []

    def _parse_error_details(
        self,
        *,
        response=None,
        exc: Optional[MalformedResponseError] = None,
    ) -> Dict[str, Any]:
        """Return a serializable parse-error record from a response or exception."""
        if response is not None:
            info = get_parse_error_info(response)
            if info:
                return dict(info)
        if exc is not None:
            return {
                "error_type": getattr(exc, "error_type", "malformed_response"),
                "error_message": str(exc),
                "details": getattr(exc, "details", {}),
            }
        return {
            "error_type": "malformed_response",
            "error_message": "Model response was malformed",
            "details": {},
        }

    def _mark_response_parse_error(
        self,
        result: RobustnessResult,
        *,
        stage: str,
        response=None,
        exc: Optional[MalformedResponseError] = None,
    ) -> RobustnessResult:
        """Mark malformed model output as invalid/skipped, never as fooled."""
        parse_error = self._parse_error_details(response=response, exc=exc)
        message = parse_error.get("error_message") or str(exc) or "Malformed model response"
        result.skipped = True
        result.skip_reason = f"Malformed model response: {message}"
        result.status = "parser_error"
        result.status_reason = result.skip_reason
        result.failure_stage = stage
        result.failure_category = "parser_error"
        result.valid_target_tested = False
        result.metadata.setdefault("parse_error", parse_error)
        if stage.startswith("baseline"):
            result.original_correct = False
            result.eligible_baseline_correct = False
        else:
            result.manipulated_correct = None
        return result

    def _apply_attack_chain(self, test_case: TestCase):
        """Apply a fixed strategy chain with the shared runtime guardrails."""
        current = test_case
        attacks_applied = []

        for strategy in self.attack_strategies:
            tool_name = STRATEGY_NAME_TO_TOOL_NAME[strategy.name]
            before = OrchestratorQuestionData.from_test_case(current)
            precondition = check_tool_preconditions(tool_name, before)
            if not precondition.valid:
                return (
                    current,
                    attacks_applied,
                    True,
                    precondition.status,
                    precondition.reason,
                )

            try:
                candidate = strategy.apply(current, {})
            except ModelExecutionError:
                raise
            except Exception as exc:
                return current, attacks_applied, True, "tool_error", str(exc)

            if candidate.metadata.get("manipulation_failed", False):
                return (
                    current,
                    attacks_applied,
                    True,
                    candidate.metadata.get("failure_category", "not_applicable"),
                    candidate.metadata.get("reason", f"{strategy.name} failed"),
                )

            try:
                after = OrchestratorQuestionData.from_test_case(candidate)
            except Exception as exc:
                return current, attacks_applied, True, "invalid_state", str(exc)

            postcondition = check_tool_postconditions(tool_name, before, after)
            if not postcondition.valid:
                return (
                    current,
                    attacks_applied,
                    True,
                    postcondition.status,
                    postcondition.reason,
                )

            current = candidate
            attacks_applied.append(strategy.name)

        return current, attacks_applied, False, None, None

    # =========================================================================
    # Summary and Reporting Methods
    # =========================================================================

    def _compute_summary(
        self,
        results: List[RobustnessResult],
        baseline_only: bool = False,
        *,
        expected_eligible_population: Optional[int] = None,
        attack_strategy_order: Optional[List[str]] = None,
    ) -> RobustnessSummary:
        """
        Compute summary from results (for progress callback and scripts/utils).

        Implements inline computation following HealthBench pattern for better
        separation of concerns (business logic in pipeline, not logger).

        Args:
            results: List of RobustnessResult objects
            baseline_only: If True, compute baseline-only summary

        Returns:
            RobustnessSummary object
        """
        attacks_applied = list(dict.fromkeys(
            list(attack_strategy_order or [])
            + [strategy.name for strategy in self.attack_strategies]
            + [
                attack
                for result in results
                for attack in (result.attacks_applied or [])
            ]
        ))
        return summarize_robustness_results(
            results,
            evaluation_type=(
                "robustness_baseline" if baseline_only else "robustness_attack"
            ),
            attacks_applied=attacks_applied,
            metadata={
                "tool_policy": (
                    self.tool_policy_decision.to_dict()
                    if self.tool_policy_decision else None
                )
            },
            expected_eligible_population=expected_eligible_population,
        )

    # ======================================================================
    # New API: Separated Baseline and Attack Runs
    # ======================================================================

    def run_baseline(
        self,
        test_cases: List[TestCase],
        max_samples: Optional[int] = None,
        progress_callback = None,
        save_every: int = 10
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """
        Run baseline evaluation (Round 1 only).

        Tests original questions without any attacks to establish baseline accuracy.
        Results can be saved and reused for attack testing.

        Args:
            test_cases: List of test cases to evaluate
            max_samples: Maximum number of samples to evaluate (None = all)
            progress_callback: Optional callback(results) for periodic saving
            save_every: Save checkpoint every N cases (default: 10)

        Returns:
            Tuple of (results, summary)
            - results: List[RobustnessResult] with baseline data
            - summary: RobustnessSummary with baseline metrics

        Example:
            >>> pipeline = RobustnessPipeline(testee, grader, attack_strategies=[])
            >>> results, summary = pipeline.run_baseline(test_cases, max_samples=100)
            >>> print(f"Baseline accuracy: {summary.first_round_accuracy:.1%}")
        """
        # Use common baseline logic from base class
        return self._run_baseline_common(
            items=test_cases,
            evaluate_fn=self._evaluate_baseline_case,
            compute_summary_fn=lambda r: self._compute_summary(r, baseline_only=True),
            print_summary_fn=self._print_baseline_summary,
            max_samples=max_samples,
            progress_callback=progress_callback,
            save_every=save_every,
            header_title="BASELINE EVALUATION (Round 1)",
            description="Baseline"
        )

    def run_attack(
        self,
        baseline_results: List[RobustnessResult],
        progress_callback = None,
        save_every: int = 10
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """
        Run attack evaluation (Round 2) using baseline results.

        Applies attack strategies to test cases that were answered correctly
        in the baseline round. No external test cases needed - reconstructs
        from self-contained baseline results.

        Args:
            baseline_results: Self-contained results from run_baseline()
            progress_callback: Optional callback(results) for periodic saving
            save_every: Save checkpoint every N cases (default: 10)

        Returns:
            Tuple of (results, summary)
            - results: List[RobustnessResult] with both baseline and attack data
            - summary: RobustnessSummary with full metrics

        Example:
            >>> # Load baseline results (self-contained, no test cases needed!)
            >>> baseline_results = load_baseline_results("baseline.json")
            >>>
            >>> # Run attack with strategies
            >>> pipeline = RobustnessPipeline(testee, grader, attack_strategies=[strategy1, strategy2])
            >>> results, summary = pipeline.run_attack(baseline_results)
            >>> print(f"Robustness: {summary.compute_robustness_score():.1%}")
        """
        # Print attack strategies info
        if self.verbose:
            print(f"Attack strategies: {', '.join([s.name for s in self.attack_strategies])}")

        eligible_results = [
            result for result in baseline_results
            if is_baseline_result_eligible_for_attack(result)
        ]
        if eligible_results and not self.attack_strategies:
            raise ValueError(
                "At least one Robustness attack strategy is required when "
                "eligible baseline cases exist."
            )

        # Define filter for correctly answered cases
        def filter_correct(r):
            return is_baseline_result_eligible_for_attack(r)

        # Create evaluation items (baseline results directly, no cross-product with strategies)
        def create_eval_items(filtered_results):
            return filtered_results

        # Create evaluation function that reconstructs test case from baseline result
        def evaluate_attack_with_baseline(baseline_result, idx):
            # Reconstruct test case using built-in method (self-contained!)
            test_case = baseline_result.to_test_case()
            return self._evaluate_attack_case(test_case, baseline_result, idx)

        # Use common attack logic from base class
        return self._run_attack_common(
            baseline_results=baseline_results,
            filter_fn=filter_correct,
            create_eval_items_fn=create_eval_items,
            evaluate_fn=evaluate_attack_with_baseline,
            compute_summary_fn=lambda r: self._compute_summary(r, baseline_only=False),
            print_summary_fn=self._print_attack_summary,
            progress_callback=progress_callback,
            save_every=save_every,
            header_title="ATTACK EVALUATION (Round 2)",
            description="Attack",
            filter_description="correctly answered cases"
        )

    def _evaluate_baseline_case(
        self,
        test_case: TestCase,
        sample_idx: int
    ) -> RobustnessResult:
        """
        Evaluate a single test case for baseline (original question only).

        Stores complete test case data for later reconstruction in attack round.

        Args:
            test_case: Test case to evaluate
            sample_idx: Index of this sample

        Returns:
            RobustnessResult with baseline data and full test case info
        """
        result = RobustnessResult(
            sample_number=sample_idx + 1,
            test_case_id=test_case.id,
            # Store full test case data for reconstruction
            original_question=test_case.question,
            original_options=test_case.options,
            original_correct_answer=test_case.correct_answer
        )

        if self.verbose:
            print(f"\n[Sample {sample_idx + 1}] {test_case.id}")

        try:
            # Get model response with complete prompt (question + options for MCQ)
            prompt = test_case.build_user_prompt()
            response = self.testee.answer(prompt)
            result.original_response = response.final_answer

            if has_parse_error(response):
                return self._mark_response_parse_error(
                    result,
                    stage="baseline_response_parse",
                    response=response,
                )

            if self.verbose:
                print(f"  Response: {result.original_response}")

            # Grade response
            grade_result = self.grader.grade(test_case, result.original_response, {})
            result.original_correct = grade_result.is_correct
            result.status = (
                "baseline_correct" if result.original_correct else "baseline_incorrect"
            )
            result.eligible_baseline_correct = result.original_correct

            if self.verbose:
                status = "✓ CORRECT" if result.original_correct else "✗ INCORRECT"
                print(f"  {status}")

        except MalformedResponseError as e:
            return self._mark_response_parse_error(
                result,
                stage="baseline_response_parse",
                exc=e,
            )
        except ModelExecutionError:
            raise
        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] Baseline evaluation failed: {e}")
            result.skipped = True
            result.skip_reason = f"Error in baseline: {str(e)}"
            result.status = "baseline_error"
            result.status_reason = result.skip_reason
            result.failure_stage = "baseline_generation_or_grading"
            result.failure_category = "generation_error"

        return result

    def _new_paired_attack_result(
        self,
        baseline_result: RobustnessResult,
        *,
        sample_number: int,
    ) -> RobustnessResult:
        """Copy one eligible baseline row into a paired attack result."""
        return RobustnessResult(
            sample_number=sample_number,
            test_case_id=baseline_result.test_case_id,
            original_question=baseline_result.original_question,
            original_options=dict(baseline_result.original_options),
            original_correct_answer=baseline_result.original_correct_answer,
            original_response=baseline_result.original_response,
            original_correct=baseline_result.original_correct,
            eligible_baseline_correct=True,
        )

    def _record_manipulated_input(
        self,
        result: RobustnessResult,
        modified_test_case: TestCase,
    ) -> None:
        """Persist the complete attacked multiple-choice input on the result."""
        result.manipulated_question = modified_test_case.question
        result.manipulated_options = dict(modified_test_case.options or {})
        result.manipulated_correct_answer = modified_test_case.correct_answer
        result.test_case_metadata = dict(modified_test_case.metadata)

    def _evaluate_prepared_attack_case(
        self,
        modified_test_case: TestCase,
        baseline_result: RobustnessResult,
        *,
        sample_number: int,
        attacks_applied: List[str],
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> RobustnessResult:
        """Evaluate an already-prepared attacked case without applying strategies."""
        if not is_baseline_result_eligible_for_attack(baseline_result):
            raise ValueError(
                "Prepared Robustness attacks require an eligible baseline-correct row"
            )
        if modified_test_case.id != baseline_result.test_case_id:
            raise ValueError(
                "Prepared attacked case ID does not match its baseline result"
            )

        result = self._new_paired_attack_result(
            baseline_result,
            sample_number=sample_number,
        )
        result.attacks_applied = list(attacks_applied)
        self._record_manipulated_input(result, modified_test_case)
        if result_metadata:
            result.metadata = dict(result_metadata)

        if self.verbose:
            print(f"\n[Sample {sample_number}] {modified_test_case.id}")
            print(f"  Attacks applied: {', '.join(attacks_applied)}")

        try:
            prompt = modified_test_case.build_user_prompt()
            response = self.testee.answer(prompt)
            result.manipulated_response = response.final_answer

            if has_parse_error(response):
                return self._mark_response_parse_error(
                    result,
                    stage="target_response_parse",
                    response=response,
                )

            if self.verbose:
                print(f"  Manipulated response: {result.manipulated_response}")

            grade_result = self.grader.grade(
                modified_test_case,
                result.manipulated_response,
                {},
            )
            extracted_answer = grade_result.metadata.get("extracted_answer")
            if not isinstance(extracted_answer, str) or not extracted_answer:
                return self._mark_response_parse_error(
                    result,
                    stage="target_response_parse",
                    exc=MalformedResponseError(
                        message="Target response contained no valid answer label",
                        raw_text=result.manipulated_response,
                        error_type="invalid_answer_label",
                        details={
                            "valid_labels": list(modified_test_case.options or {}),
                        },
                    ),
                )
            result.manipulated_correct = grade_result.is_correct
            result.valid_target_tested = True
            result.status = (
                "target_safe" if result.manipulated_correct else "fooled"
            )

            if self.verbose:
                status = "✓" if result.manipulated_correct else "✗"
                outcome = "CORRECT" if result.manipulated_correct else "INCORRECT"
                print(f"  {status} Manipulated: {outcome}")

        except MalformedResponseError as exc:
            return self._mark_response_parse_error(
                result,
                stage="target_response_parse",
                exc=exc,
            )
        except ModelExecutionError:
            raise
        except Exception as exc:
            if self.verbose:
                print(f"  [ERROR] Attack evaluation failed: {exc}")
            result.skipped = True
            result.skip_reason = f"Error in attack round: {exc}"
            result.status = "generation_error"
            result.status_reason = result.skip_reason
            result.failure_stage = "target_generation_or_grading"
            result.failure_category = "generation_error"

        return result

    def _evaluate_attack_case(
        self,
        test_case: TestCase,
        baseline_result: RobustnessResult,
        sample_idx: int,
    ) -> RobustnessResult:
        """Apply the live attack chain, then evaluate the prepared attacked case."""
        sample_number = sample_idx + 1
        (
            modified_test_case,
            attacks_applied,
            manipulation_failed,
            failure_category,
            failure_reason,
        ) = self._apply_attack_chain(test_case)

        if manipulation_failed:
            if self.verbose:
                print(f"\n[Sample {sample_number}] {test_case.id}")
                print("  Manipulation failed, skipping target evaluation")
            result = self._new_paired_attack_result(
                baseline_result,
                sample_number=sample_number,
            )
            self._record_manipulated_input(result, modified_test_case)
            result.skipped = True
            result.skip_reason = failure_reason or "Manipulation failed"
            result.manipulation_failed = True
            result.status = failure_category or "invalid_noop"
            result.status_reason = result.skip_reason
            result.failure_stage = "attack_generation"
            result.failure_category = failure_category
            result.attacks_applied = attacks_applied
            return result

        return self._evaluate_prepared_attack_case(
            modified_test_case,
            baseline_result,
            sample_number=sample_number,
            attacks_applied=attacks_applied,
        )

    def _evaluate_replay_item(
        self,
        replay_item: RobustnessReplayItem,
        _loop_index: int,
    ) -> RobustnessResult:
        """Evaluate one validated pre-generated attack at its frozen ordinal."""
        return self._evaluate_prepared_attack_case(
            replay_item.attacked_test_case,
            replay_item.baseline_result,
            sample_number=replay_item.sample_number,
            attacks_applied=replay_item.attacks_applied,
            result_metadata={
                "attack_metadata": dict(replay_item.attack_metadata),
                "replay": {
                    "population_ordinal": replay_item.sample_number,
                    "source_ordinal": replay_item.source_ordinal,
                },
            },
        )

    def run_attack_replay(
        self,
        replay_items: List[RobustnessReplayItem],
        *,
        expected_eligible_population: int,
        attack_strategy_order: List[str],
        seed_results: Optional[List[RobustnessResult]] = None,
        progress_callback=None,
        save_every: int = 10,
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """Evaluate pre-generated attacks and summarize with existing replay rows."""
        from med_red_team.utils import evaluate_items_fail_fast

        if replay_items and (self.testee is None or self.grader is None):
            raise ValueError("Replay target evaluation requires a testee and grader")

        if self.verbose:
            print(f"\n{'='*80}")
            print("PRE-GENERATED ATTACK REPLAY")
            print(f"{'='*80}")
            print(f"Target-testable replay items: {len(replay_items)}")

        results = evaluate_items_fail_fast(
            items=replay_items,
            evaluate_fn=self._evaluate_replay_item,
            description="Attack replay",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose,
            use_tqdm=self.verbose,
        )
        summary_results = list(seed_results or []) + results
        summary = self._compute_summary(
            summary_results,
            baseline_only=False,
            expected_eligible_population=expected_eligible_population,
            attack_strategy_order=attack_strategy_order,
        )
        if self.verbose:
            self._print_attack_summary(summary)
        return results, summary

    def _print_baseline_summary(self, summary: RobustnessSummary):
        """Print baseline summary statistics."""
        print(f"\n{'='*80}")
        print("BASELINE SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal cases: {summary.total_samples}")
        print(f"Correct: {summary.first_round_correct} ({summary.first_round_accuracy:.1%})")
        print(f"Skipped: {summary.skipped_samples}")

    def _print_attack_summary(self, summary: RobustnessSummary):
        """Print attack summary statistics."""
        print(f"\n{'='*80}")
        print("ATTACK SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal cases attacked: {summary.total_samples}")
        print(f"Baseline correct: {summary.first_round_correct} ({summary.first_round_accuracy:.1%})")

        print(f"Failed manipulations: {summary.failed_manipulations}")
        print(f"Valid target-tested attacks: {summary.valid_target_tested}")
        print(f"Fooled: {summary.fooled}")
        print(f"Target safe: {summary.target_safe}")
        print(f"Attack coverage: {summary.attack_coverage:.1%}")
        print(
            f"Conditional attack success: "
            f"{summary.conditional_attack_success_rate:.1%}"
        )
        print(f"Conditional robustness: {summary.conditional_robustness:.1%}")
        print(
            f"End-to-end attack success: "
            f"{summary.end_to_end_attack_success_rate:.1%}"
        )
        print(f"Strategies: {', '.join(summary.attacks_applied)}")

    # ======================================================================
    # Attack Dataset Generation (Testee-Independent)
    # ======================================================================

    def generate_attacked_dataset(
        self,
        test_cases: List[TestCase],
        max_samples: Optional[int] = None,
        progress_callback = None,
        save_every: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate attacked test cases WITHOUT testee evaluation.

        This creates a reusable dataset of attacked questions that can be
        tested on multiple models without regenerating attacks.

        Args:
            test_cases: List of original test cases to attack
            max_samples: Maximum number of samples to process (None = all)
            progress_callback: Optional callback(results) for periodic saving
            save_every: Save checkpoint every N cases (default: 10)

        Returns:
            List of dicts, each containing:
            - original: Original test case data
            - attacked: Attacked test case data
            - attack_metadata: Attack strategies and modifications applied

        Example:
            >>> pipeline = RobustnessPipeline(
            ...     testee=None,
            ...     grader=None,
            ...     attack_strategies=[bias_strategy, distraction_strategy],
            ...     verbose=True
            ... )
            >>> attacked_dataset = pipeline.generate_attacked_dataset(test_cases, max_samples=200)
            >>> # Save to JSON for reuse across multiple testee models
        """
        from med_red_team.utils import evaluate_items_fail_fast

        # Limit test cases if requested
        if max_samples is not None:
            test_cases = test_cases[:max_samples]

        if test_cases and not self.attack_strategies:
            raise ValueError(
                "At least one Robustness attack strategy is required when "
                "attack-generation cases exist."
            )

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERATING ATTACKED DATASET")
            print(f"{'='*80}")
            print(f"Processing {len(test_cases)} test cases")
            print(f"Attack strategies: {', '.join([s.name for s in self.attack_strategies])}")

        # Use evaluate_items_fail_fast for robust error handling and progress tracking
        attacked_dataset = evaluate_items_fail_fast(
            items=test_cases,
            evaluate_fn=self._generate_single_attack,
            description="Generating attacks",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose
        )

        # Count failures
        failed_count = sum(1 for item in attacked_dataset if item["attack_metadata"]["manipulation_failed"])

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERATION COMPLETE")
            print(f"{'='*80}")
            print(f"Total cases processed: {len(attacked_dataset)}")
            print(f"Successfully attacked: {len(attacked_dataset) - failed_count}")
            print(f"Failed manipulations: {failed_count}")

        return attacked_dataset

    def _generate_single_attack(
        self,
        test_case: TestCase,
        idx: int
    ) -> Dict[str, Any]:
        """
        Generate a single attacked test case.

        Args:
            test_case: Original test case to attack
            idx: Index of this test case

        Returns:
            Dict with original, attacked, and attack_metadata fields
        """
        # Store original test case data
        original_data = {
            "id": test_case.id,
            "question": test_case.question,
            "options": test_case.options,
            "correct_answer": test_case.correct_answer
        }

        (
            modified_test_case,
            attacks_applied,
            manipulation_failed,
            failure_category,
            failure_reason,
        ) = self._apply_attack_chain(test_case)
        attack_metadata = dict(modified_test_case.metadata)

        # Ensure manipulation_failed is set in attack_metadata
        # (handles edge case where exception occurred before update_attack_metadata)
        # Final chain outcome overrides any earlier strategy control flag.
        attack_metadata["manipulation_failed"] = manipulation_failed
        attack_metadata["status"] = (
            failure_category if manipulation_failed else "generated"
        )
        attack_metadata["failure_category"] = failure_category
        attack_metadata["reason"] = failure_reason

        # Store attacked test case data
        attacked_data = {
            "id": test_case.id,
            "question": modified_test_case.question,
            "options": modified_test_case.options,
            "correct_answer": modified_test_case.correct_answer
        }

        # Return attacked dataset item
        return {
            "original": original_data,
            "attacked": attacked_data,
            "attack_metadata": {
                **attack_metadata,
                "strategies_applied": attacks_applied
                # Note: manipulation_failed comes from attack_metadata (already propagated)
            }
        }

    def save_results(
        self,
        results: List[RobustnessResult],
        summary: RobustnessSummary,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save robustness evaluation results to JSON file.

        Standardized save interface used across all pipeline classes.

        Args:
            results: List of RobustnessResult objects
            summary: RobustnessSummary object
            output_path: Path to output JSON file
            metadata: Optional metadata dict to include

        Example:
            >>> results, summary = pipeline.run_baseline(test_cases)
            >>> pipeline.save_results(
            ...     results, summary,
            ...     "logs/gpt-4o_robustness_baseline_all_20251130.json",
            ...     metadata={"config": config.to_dict()}
            ... )
        """
        from datetime import datetime
        from med_red_team.shared.io import (
            atomic_write_result_envelope,
            ensure_result_metadata,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        envelope_metadata = ensure_result_metadata(metadata, axis="robustness")
        envelope_metadata.setdefault("timestamp", datetime.now().isoformat())

        atomic_write_result_envelope(
            output_path,
            metadata=envelope_metadata,
            summary=summary,
            data=[r.to_dict() for r in results],
            data_key="results",
        )

        if self.verbose:
            print(f"\n✓ Results saved to: {output_path}")


