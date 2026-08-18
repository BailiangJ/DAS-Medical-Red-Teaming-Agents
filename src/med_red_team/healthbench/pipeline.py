"""
HealthBench evaluation pipeline.
"""

import copy
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol
from collections import defaultdict
import numpy as np

from med_red_team.healthbench.data import (
    HealthBenchTestCase,
    RubricItem,
    RubricGradeResult,
    validate_cached_baseline_result,
    validate_healthbench_artifact_metadata,
)
from med_red_team.healthbench.grader import RubricGrader
from med_red_team.shared.config_loading import validate_sample_limit
from med_red_team.healthbench.utils import (
    load_healthbench_jsonl,
    calculate_score,
    calculate_tag_scores,
    compute_bootstrap_std,
    filter_single_turn_cases,
    grades_fully_evaluated,
    validate_attack_result_state,
    validate_grading_results_match_rubrics,
    validate_selected_rubric_index,
)


# ==============================================================================
# Robustness Testing Pipeline
# ==============================================================================

from typing import Tuple
from tqdm import tqdm
from med_red_team.pipeline import EvaluationPipeline
from med_red_team.testee import Testee
from med_red_team.healthbench.data import (
    HealthBenchRobustnessResult,
    HealthBenchRobustnessSummary,
    AttackResult,
)
from med_red_team.healthbench.prompts import IMPOSSIBLE_MEASUREMENT_RUBRICS
from med_red_team.utils import evaluate_items_fail_fast


class HealthBenchConversationAttack(Protocol):
    """Concrete HealthBench attack shape consumed by this pipeline."""

    name: str

    def apply_to_conversation(self, *args, **kwargs) -> AttackResult:
        """Return a HealthBench AttackResult for one conversation."""
        ...


def _actor_generation_config(actor: Any) -> Optional[Dict[str, Any]]:
    """Return an actor's effective generation config when it is inspectable."""
    config = getattr(actor, "config", None)
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if isinstance(config, dict):
        return dict(config)
    return None


class HealthBenchRobustnessPipeline(EvaluationPipeline):
    """
    Concrete HealthBench conversation-and-rubric robustness pipeline.

    This axis reuses shared loop helpers from ``EvaluationPipeline``, but its
    public API is HealthBench-specific: callers pass ``HealthBenchTestCase``
    objects, concrete HealthBench conversation attack implementations, and they
    receive ``HealthBenchRobustnessResult`` / ``HealthBenchRobustnessSummary``
    objects.

    Workflow:
    1. Baseline round: test original conversations.
    2. Attack round: apply HealthBench conversation strategies.
    3. Attacked round: re-test modified conversations with the applicable
       rubric slice.

    Supports:
    - One HealthBench attack strategy per attack run
    - Baseline caching (skip first round if results exist)
    - Extra rubrics for attacked cases (e.g., IMPOSSIBLE_MEASUREMENT_RUBRICS)
    - Single-turn filtering

    Example:
        >>> pipeline = HealthBenchRobustnessPipeline(
        ...     testee=testee,
        ...     grader=grader,
        ...     attack_strategies=[AdjustImpossibleMeasurementStrategy(...)],
        ...     verbose=True
        ... )
        >>> results, summary = pipeline.run_attack(
        ...     test_cases=test_cases,
        ...     max_samples=100,
        ...     baseline_results=None  # Or load from baseline run
        ... )
    """

    def __init__(
        self,
        testee: Testee,
        grader: RubricGrader,
        attack_strategies: List[HealthBenchConversationAttack],
        verbose: bool = True,
        bootstrap_seed: Optional[int] = 0,
    ):
        # Initialize base class
        super().__init__(testee, grader, verbose)

        # HealthBench-specific attributes
        self.attack_strategies = attack_strategies
        self.bootstrap_seed = bootstrap_seed

    @staticmethod
    def _invalid_grading_reason(grades: List[RubricGradeResult]) -> str:
        invalid_explanations = [
            grade.explanation
            for grade in grades
            if grade.criteria_met is None and grade.explanation
        ]
        if invalid_explanations:
            return invalid_explanations[0]
        return "Invalid rubric grading result"

    @classmethod
    def _grading_validation_error(
        cls,
        rubric_items: List[RubricItem],
        grades: List[RubricGradeResult],
        *,
        label: str,
    ) -> Optional[str]:
        mismatch = validate_grading_results_match_rubrics(
            rubric_items,
            grades,
            label=label,
        )
        if mismatch is not None:
            return mismatch
        if not grades_fully_evaluated(grades):
            return cls._invalid_grading_reason(grades)
        return None

    def run_baseline(
        self,
        test_cases: List[HealthBenchTestCase],
        max_samples: Optional[int] = None,
        filter_single_turn: bool = True,
        progress_callback: Optional[callable] = None,
        save_every: int = 10
    ) -> Tuple[List[HealthBenchRobustnessResult], HealthBenchRobustnessSummary]:
        """
        Run baseline evaluation (Round 1 only, no attacks).

        Args:
            test_cases: List of test cases
            max_samples: Maximum number of samples to process
            filter_single_turn: Whether to filter to single-turn conversations
            progress_callback: Optional callback function(results) to save intermediate progress
            save_every: Save progress every N samples (default: 10)

        Returns:
            (results, summary) tuple

        Note:
            If an error occurs during evaluation, this method will return whatever
            results have been collected so far, rather than raising an exception.
            KeyboardInterrupt is re-raised to allow graceful termination.
        """
        max_samples = validate_sample_limit(max_samples)

        # Print additional info before common method
        if self.verbose:
            print(f"Testee: {self.testee.model_id}")
            print(f"Grader: {self.grader.model_id}")

        # Define preprocessing function for single-turn filtering
        preprocess_fn = filter_single_turn_cases if filter_single_turn else None

        # Use common baseline logic from base class
        return self._run_baseline_common(
            items=test_cases,
            evaluate_fn=self._evaluate_baseline,
            compute_summary_fn=lambda r: self._compute_summary(r, baseline_only=True),
            print_summary_fn=lambda s: self._print_summary(s, baseline_only=True),
            max_samples=max_samples,
            preprocess_fn=preprocess_fn,
            progress_callback=progress_callback,
            save_every=save_every,
            header_title="HEALTHBENCH BASELINE EVALUATION",
            description="Baseline"
        )

    def run_attack(
        self,
        test_cases: List[HealthBenchTestCase],
        max_samples: Optional[int] = None,
        filter_single_turn: bool = True,
        baseline_results: Optional[List[HealthBenchRobustnessResult]] = None,
        progress_callback: Optional[callable] = None,
        save_every: int = 10,
        *,
        baseline_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[HealthBenchRobustnessResult], HealthBenchRobustnessSummary]:
        """
        Run attack evaluation (Baseline + Attack + Attacked rounds).

        Args:
            test_cases: List of test cases
            max_samples: Maximum samples to attack
            filter_single_turn: Filter to single-turn
            baseline_results: Pre-computed baseline results (skip Round 1 if provided)
            baseline_metadata: Complete baseline artifact metadata required when cached
                results are reused by a baseline-dependent attack

        Returns:
            (results, summary) tuple
        """
        max_samples = validate_sample_limit(max_samples)

        if filter_single_turn:
            test_cases = filter_single_turn_cases(test_cases)

        if max_samples is not None:
            test_cases = test_cases[:max_samples]

        if test_cases and len(self.attack_strategies) != 1:
            raise ValueError(
                "Exactly one HealthBench attack strategy must be configured per run. "
                "Run each strategy separately."
            )

        provided_baseline_results = baseline_results is not None
        need_baseline = any(
            strategy.name in {"distraction", "cognitive_bias"}
            for strategy in self.attack_strategies
        )
        if test_cases and need_baseline and provided_baseline_results:
            if baseline_metadata is None:
                raise ValueError(
                    "Cached HealthBench baseline results require metadata from the "
                    "same complete baseline artifact"
                )
            validate_healthbench_artifact_metadata(
                baseline_metadata,
                phase="baseline",
                label="Cached HealthBench baseline",
                expected_testee_model=getattr(self.testee, "model_id", None),
                expected_testee_config=_actor_generation_config(self.testee),
                expected_testee_system_prompt=getattr(
                    self.testee,
                    "system_prompt",
                    None,
                ),
                expected_grader_model=getattr(self.grader, "model_id", None),
                expected_grader_config=_actor_generation_config(self.grader),
            )

        # Build a lookup only when the configured strategy consumes baselines.
        baseline_lookup = {}
        if need_baseline and baseline_results is not None:
            duplicates = set()
            for result in baseline_results:
                if result.test_case_id in baseline_lookup:
                    duplicates.add(result.test_case_id)
                baseline_lookup[result.test_case_id] = result
            if duplicates:
                formatted = ", ".join(sorted(duplicates))
                raise ValueError(
                    f"Cached baseline results contain duplicate test_case_id values: {formatted}"
                )
            if self.verbose:
                print(f"Loaded {len(baseline_lookup)} baseline results for caching")

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"HEALTHBENCH ATTACK EVALUATION")
            print(f"{'='*80}")
            print(f"Test cases: {len(test_cases)}")
            print(f"Max samples: {'All' if max_samples is None else max_samples}")
            print(f"Testee: {self.testee.model_id}")
            print(f"Grader: {self.grader.model_id}")
            print(f"Attack strategies: {[s.name for s in self.attack_strategies]}")
            print(f"Using cached baseline: {len(baseline_lookup) > 0}")

        # Check whether an explicitly supplied cache covers every requested case.
        if need_baseline and provided_baseline_results:
            missing_baselines = [
                test_case.prompt_id
                for test_case in test_cases
                if test_case.prompt_id not in baseline_lookup
            ]
            if missing_baselines:
                preview = ", ".join(missing_baselines[:5])
                raise ValueError(
                    "Cached baseline results do not cover all requested HealthBench cases "
                    f"(examples: {preview})"
                )
            for test_case in test_cases:
                validate_cached_baseline_result(
                    test_case,
                    baseline_lookup[test_case.prompt_id],
                )

        # Create evaluation function for safe loop
        def evaluate_attack_case(test_case: HealthBenchTestCase, idx: int) -> HealthBenchRobustnessResult:
            baseline_result = None
            if need_baseline:
                if test_case.prompt_id in baseline_lookup:
                    baseline_result = baseline_lookup[test_case.prompt_id]
                else:
                    baseline_result = self._evaluate_baseline(test_case, idx)

            attack_result = (
                self._apply_attack_no_baseline(test_case)
                if not need_baseline
                else self._apply_attack(test_case, baseline_result)
            )

            selected_rubric_index = None
            if attack_result.attack_strategy in {"distraction", "cognitive_bias"}:
                selected_rubric_index = attack_result.attack_metadata.get(
                    "selected_rubric_index",
                    -1,
                )
            attack_state_error = validate_attack_result_state(
                applicable=attack_result.applicable,
                modified_conversation=attack_result.modified_conversation,
                original_conversation=test_case.conversation,
                selected_rubric_index=selected_rubric_index,
            )
            if attack_state_error:
                skip_reason = attack_state_error
                if attack_result.error:
                    skip_reason = f"{attack_result.error}; {attack_state_error}"
                if need_baseline:
                    attacked_result = copy.deepcopy(baseline_result)
                    attacked_result.attack_result = attack_result
                    attacked_result.skipped = True
                    attacked_result.skip_reason = skip_reason
                    return attacked_result
                return HealthBenchRobustnessResult(
                    test_case_id=test_case.prompt_id,
                    sample_number=idx + 1,
                    original_conversation=test_case.conversation,
                    original_rubrics=test_case.rubrics,
                    baseline_completion=None,
                    baseline_grades=[],
                    baseline_score=None,
                    attack_result=attack_result,
                    attacked_conversation=None,
                    attacked_rubrics=None,
                    attacked_completion=None,
                    attacked_grades=None,
                    attacked_score=None,
                    skipped=True,
                    skip_reason=skip_reason,
                )

            if attack_result.applicable is True:
                if attack_result.attack_strategy == "impossible_measurement":
                    return self._evaluate_attacked_standalone(
                        test_case,
                        attack_result,
                        idx,
                    )
                if attack_result.attack_strategy in ["distraction", "cognitive_bias"]:
                    return self._evaluate_attacked(
                        test_case,
                        attack_result,
                        baseline_result,
                    )
                return self._evaluate_attacked_standalone(
                    test_case,
                    attack_result,
                    idx,
                )

            if need_baseline:
                attacked_result = copy.deepcopy(baseline_result)
                attacked_result.attack_result = attack_result
                attacked_result.skipped = True
                attacked_result.skip_reason = attack_result.error or "Attack not applicable"
                return attacked_result

            return HealthBenchRobustnessResult(
                test_case_id=test_case.prompt_id,
                sample_number=idx + 1,
                original_conversation=test_case.conversation,
                original_rubrics=test_case.rubrics,
                baseline_completion=None,
                baseline_grades=[],
                baseline_score=None,
                attack_result=attack_result,
                attacked_conversation=None,
                attacked_rubrics=None,
                attacked_completion=None,
                attacked_grades=None,
                attacked_score=None,
                skipped=True,
                skip_reason=attack_result.error or "Attack not applicable",
            )

        # Use shared safe evaluation loop
        results = evaluate_items_fail_fast(
            items=test_cases,
            evaluate_fn=evaluate_attack_case,
            description="Attack",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose,
            use_tqdm=True
        )

        summary = self._compute_summary(results, baseline_only=False)

        if self.verbose:
            self._print_summary(summary, baseline_only=False)

        return results, summary

    def _evaluate_baseline(
        self,
        test_case: HealthBenchTestCase,
        sample_idx: int
    ) -> HealthBenchRobustnessResult:
        """Evaluate baseline (original conversation)."""
        # Format conversation
        conv_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in test_case.conversation
        ])

        # Get completion
        response = self.testee.answer(conv_text)
        completion = response.final_answer

        # Check if response has parsing error (e.g., unclosed thinking tag)
        # Skip expensive LLM grading to avoid wasting tokens
        from med_red_team.models.utils.response_utils import has_parse_error, get_parse_error_info

        if has_parse_error(response):
            error_info = get_parse_error_info(response)
            warning_msg = error_info.get("error_message", "Unknown parse error")

            if self.verbose:
                print(f"⚠ Skipping grading due to malformed testee response (test case {test_case.prompt_id}): {warning_msg}")

            # Create result with empty grades and no scientific score.
            return HealthBenchRobustnessResult(
                test_case_id=test_case.prompt_id,
                sample_number=sample_idx + 1,
                original_conversation=test_case.conversation,
                original_rubrics=test_case.rubrics,
                baseline_completion=completion,
                baseline_grades=[],
                baseline_score=None,
                attack_result=None,
                attacked_conversation=None,
                attacked_rubrics=None,
                attacked_completion=None,
                attacked_grades=None,
                attacked_score=None,
                skipped=True,
                skip_reason=warning_msg
            )

        # Grade
        grades = self.grader.grade(
            conversation=test_case.conversation,
            completion=completion,
            rubric_items=test_case.rubrics
        )

        score = calculate_score(test_case.rubrics, grades)
        validation_error = self._grading_validation_error(
            test_case.rubrics,
            grades,
            label="HealthBench baseline grades",
        )
        if validation_error is not None:
            return HealthBenchRobustnessResult(
                test_case_id=test_case.prompt_id,
                sample_number=sample_idx + 1,
                original_conversation=test_case.conversation,
                original_rubrics=test_case.rubrics,
                baseline_completion=completion,
                baseline_grades=grades,
                baseline_score=None,
                attack_result=None,
                attacked_conversation=None,
                attacked_rubrics=None,
                attacked_completion=None,
                attacked_grades=None,
                attacked_score=None,
                skipped=True,
                skip_reason=validation_error,
            )

        return HealthBenchRobustnessResult(
            test_case_id=test_case.prompt_id,
            sample_number=sample_idx + 1,
            original_conversation=test_case.conversation,
            original_rubrics=test_case.rubrics,
            baseline_completion=completion,
            baseline_grades=grades,
            baseline_score=score,
            attack_result=None,
            attacked_conversation=None,
            attacked_rubrics=None,
            attacked_completion=None,
            attacked_grades=None,
            attacked_score=None
        )

    def _apply_attack_no_baseline(
        self,
        test_case: HealthBenchTestCase
    ) -> AttackResult:
        """Apply attack strategy that doesn't need baseline (e.g., impossible_measurement)."""
        # Try each strategy and return its result (attacker handles applicable/error logic)
        for strategy in self.attack_strategies:
            if strategy.name == "impossible_measurement":
                return strategy.apply_to_conversation(
                    conversation=test_case.conversation,
                    test_case_id=test_case.prompt_id
                )

        # No recognized strategy found
        return AttackResult(
            test_case_id=test_case.prompt_id,
            attack_strategy="none",
            applicable=False,
            original_conversation=test_case.conversation,
            attack_metadata={},
            modified_conversation=None
        )

    def _filter_attackable_rubrics(
        self,
        baseline_grades: List[RubricGradeResult]
    ) -> Tuple[List[Dict[str, Any]], List[RubricItem]]:
        """
        Filter to attackable rubrics:
        - Positive rubrics that were MET (points > 0, criteria_met = True)
        - Negative rubrics that were NOT MET (points < 0, criteria_met = False)

        Args:
            baseline_grades: Grading results from baseline evaluation

        Returns:
            Tuple of:
            - List of dicts with rubric, grader_explanation, and rubric_index (for attacker)
            - List of RubricItem objects (for Round 2 grading)
        """
        attackable_rubrics_with_context = []
        attackable_rubric_items = []

        for i, grade in enumerate(baseline_grades):
            points = grade.rubric_item.points
            criteria_met = grade.criteria_met

            # Positive rubric that was met (positive rubric currently met; unsafe-direction target is unmet)
            if points > 0 and criteria_met is True:
                attackable_rubrics_with_context.append({
                    'rubric': grade.rubric_item.to_dict(),
                    'grader_explanation': grade.explanation,
                    'rubric_index': i  # Original index in full rubrics array
                })
                attackable_rubric_items.append(grade.rubric_item)
            # Negative rubric that was NOT met (negative rubric currently unmet; unsafe-direction target is met)
            elif points < 0 and criteria_met is False:
                attackable_rubrics_with_context.append({
                    'rubric': grade.rubric_item.to_dict(),
                    'grader_explanation': grade.explanation,
                    'rubric_index': i  # Original index in full rubrics array
                })
                attackable_rubric_items.append(grade.rubric_item)

        return attackable_rubrics_with_context, attackable_rubric_items

    def _apply_attack(
        self,
        test_case: HealthBenchTestCase,
        baseline_result: HealthBenchRobustnessResult
    ) -> AttackResult:
        """Apply attack strategy to test case using baseline grading results."""
        # Try each strategy and return its result (attacker handles applicable/error logic)
        for strategy in self.attack_strategies:
            if strategy.name == "impossible_measurement":
                return strategy.apply_to_conversation(
                    conversation=test_case.conversation,
                    test_case_id=test_case.prompt_id
                )
            elif strategy.name in ["distraction", "cognitive_bias"]:
                # Filter to attackable rubrics (positive MET + negative NOT MET)
                attackable_rubrics_with_context, attackable_rubric_items = self._filter_attackable_rubrics(
                    baseline_result.baseline_grades
                )

                attack_result = strategy.apply_to_conversation(
                    conversation=test_case.conversation,
                    attackable_rubrics_with_context=attackable_rubrics_with_context,
                    test_case_id=test_case.prompt_id
                )

                # Store attackable_rubric_items for Round 2 use (need to serialize to dict)
                attack_result.attack_metadata['attackable_rubric_items'] = [
                    rubric.to_dict() for rubric in attackable_rubric_items
                ]

                return attack_result
            else:
                continue

        # No recognized strategy found
        return AttackResult(
            test_case_id=test_case.prompt_id,
            attack_strategy="none",
            applicable=False,
            original_conversation=test_case.conversation,
            attack_metadata={},
            modified_conversation=None
        )

    def _evaluate_attacked_standalone(
        self,
        test_case: HealthBenchTestCase,
        attack_result: AttackResult,
        sample_idx: int
    ) -> HealthBenchRobustnessResult:
        """Evaluate attacked conversation without baseline (for impossible_measurement)."""
        if attack_result.attack_strategy == "impossible_measurement":
            attacked_rubrics = [
                RubricItem.from_dict(IMPOSSIBLE_MEASUREMENT_RUBRICS)
            ]
        else:
            attacked_rubrics = test_case.rubrics.copy()

        conv_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in attack_result.modified_conversation
        ])

        response = self.testee.answer(conv_text)
        completion = response.final_answer

        from med_red_team.models.utils.response_utils import has_parse_error, get_parse_error_info

        if has_parse_error(response):
            error_info = get_parse_error_info(response)
            warning_msg = error_info.get("error_message", "Unknown parse error")

            if self.verbose:
                print(f"⚠ Skipping grading due to malformed testee response (test case {test_case.prompt_id}): {warning_msg}")

            return HealthBenchRobustnessResult(
                test_case_id=test_case.prompt_id,
                sample_number=sample_idx + 1,
                original_conversation=test_case.conversation,
                original_rubrics=test_case.rubrics,
                baseline_completion=None,
                baseline_grades=[],
                baseline_score=None,
                attack_result=attack_result,
                attacked_conversation=attack_result.modified_conversation,
                attacked_rubrics=attacked_rubrics,
                attacked_completion=completion,
                attacked_grades=[],
                attacked_score=None,
                skipped=True,
                skip_reason=warning_msg,
            )

        grades = self.grader.grade(
            conversation=attack_result.modified_conversation,
            completion=completion,
            rubric_items=attacked_rubrics
        )
        score = calculate_score(attacked_rubrics, grades)

        result = HealthBenchRobustnessResult(
            test_case_id=test_case.prompt_id,
            sample_number=sample_idx + 1,
            original_conversation=test_case.conversation,
            original_rubrics=test_case.rubrics,
            baseline_completion=None,
            baseline_grades=[],
            baseline_score=None,
            attack_result=attack_result,
            attacked_conversation=attack_result.modified_conversation,
            attacked_rubrics=attacked_rubrics,
            attacked_completion=completion,
            attacked_grades=grades,
            attacked_score=score,
        )
        validation_error = self._grading_validation_error(
            attacked_rubrics,
            grades,
            label="HealthBench attacked grades",
        )
        if validation_error is not None:
            result.skipped = True
            result.skip_reason = validation_error
        return result

    def _evaluate_attacked(
        self,
        test_case: HealthBenchTestCase,
        attack_result: AttackResult,
        baseline_result: HealthBenchRobustnessResult
    ) -> HealthBenchRobustnessResult:
        """Evaluate attacked conversation with baseline (for distraction/cognitive_bias)."""
        attackable_rubric_dicts = attack_result.attack_metadata.get('attackable_rubric_items', [])
        raw_selected_idx = attack_result.attack_metadata.get('selected_rubric_index', -1)
        raw_original_idx = attack_result.attack_metadata.get('original_rubric_index', -1)
        comparable_baseline_score = baseline_result.baseline_score

        try:
            selected_idx = validate_selected_rubric_index(
                raw_selected_idx,
                len(attackable_rubric_dicts),
            )
        except ValueError as e:
            result = copy.deepcopy(baseline_result)
            result.attack_result = attack_result
            result.skipped = True
            result.skip_reason = str(e)
            return result

        if attack_result.attack_strategy == "impossible_measurement":
            attacked_rubrics = [
                RubricItem.from_dict(IMPOSSIBLE_MEASUREMENT_RUBRICS)
            ]
        elif attack_result.attack_strategy in ["distraction", "cognitive_bias"]:
            if 0 <= selected_idx < len(attackable_rubric_dicts):
                attacked_rubrics = [
                    RubricItem.from_dict(attackable_rubric_dicts[selected_idx])
                ]
            else:
                result = copy.deepcopy(baseline_result)
                result.attack_result = attack_result
                result.skipped = True
                result.skip_reason = f"Invalid selected_rubric_index {selected_idx}"
                return result

            try:
                original_idx = validate_selected_rubric_index(
                    raw_original_idx,
                    len(baseline_result.original_rubrics),
                )
            except ValueError as e:
                result = copy.deepcopy(baseline_result)
                result.attack_result = attack_result
                result.skipped = True
                result.skip_reason = str(e)
                return result

            if original_idx >= len(baseline_result.baseline_grades):
                result = copy.deepcopy(baseline_result)
                result.attack_result = attack_result
                result.skipped = True
                result.skip_reason = (
                    "Selected baseline rubric index is missing from baseline grades"
                )
                return result

            selected_baseline_rubric = baseline_result.original_rubrics[original_idx]
            selected_baseline_grade = baseline_result.baseline_grades[original_idx]
            if attacked_rubrics[0] != selected_baseline_rubric:
                result = copy.deepcopy(baseline_result)
                result.attack_result = attack_result
                result.skipped = True
                result.skip_reason = (
                    "Selected attacked rubric does not match the referenced baseline rubric"
                )
                return result

            baseline_validation_error = self._grading_validation_error(
                [selected_baseline_rubric],
                [selected_baseline_grade],
                label="HealthBench selected baseline grade",
            )
            if baseline_validation_error is not None:
                result = copy.deepcopy(baseline_result)
                result.attack_result = attack_result
                result.skipped = True
                result.skip_reason = baseline_validation_error
                return result

            comparable_baseline_score = calculate_score(
                [selected_baseline_rubric],
                [selected_baseline_grade],
            )
        else:
            attacked_rubrics = test_case.rubrics.copy()

        conv_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in attack_result.modified_conversation
        ])

        response = self.testee.answer(conv_text)
        completion = response.final_answer

        from med_red_team.models.utils.response_utils import has_parse_error, get_parse_error_info

        if has_parse_error(response):
            error_info = get_parse_error_info(response)
            warning_msg = error_info.get("error_message", "Unknown parse error")

            if self.verbose:
                print(f"⚠ Skipping grading due to malformed testee response (test case {test_case.prompt_id}): {warning_msg}")

            result = copy.deepcopy(baseline_result)
            result.attack_result = attack_result
            result.baseline_score = comparable_baseline_score
            result.attacked_conversation = attack_result.modified_conversation
            result.attacked_rubrics = attacked_rubrics
            result.attacked_completion = completion
            result.attacked_grades = []
            result.attacked_score = None
            result.skipped = True
            result.skip_reason = warning_msg
            return result

        grades = self.grader.grade(
            conversation=attack_result.modified_conversation,
            completion=completion,
            rubric_items=attacked_rubrics
        )
        score = calculate_score(attacked_rubrics, grades)

        result = copy.deepcopy(baseline_result)
        result.attack_result = attack_result
        result.baseline_score = comparable_baseline_score
        result.attacked_conversation = attack_result.modified_conversation
        result.attacked_rubrics = attacked_rubrics
        result.attacked_completion = completion
        result.attacked_grades = grades
        result.attacked_score = score
        validation_error = self._grading_validation_error(
            attacked_rubrics,
            grades,
            label="HealthBench attacked grades",
        )
        if validation_error is not None:
            result.skipped = True
            result.skip_reason = validation_error
        return result

    def _compute_summary(
        self,
        results: List[HealthBenchRobustnessResult],
        baseline_only: bool
    ) -> HealthBenchRobustnessSummary:
        """Compute summary statistics."""
        baseline_scores = [r.baseline_score for r in results if r.baseline_score is not None]

        baseline_avg = float(np.mean(baseline_scores)) if baseline_scores else 0.0
        baseline_std = (
            compute_bootstrap_std(
                baseline_scores,
                seed=self.bootstrap_seed,
            )
            if baseline_scores else 0.0
        )

        # Count rubrics met at baseline
        total_rubrics_met = sum(
            sum(1 for g in r.baseline_grades if g.criteria_met is True)
            for r in results
        )

        if baseline_only:
            return HealthBenchRobustnessSummary(
                total_cases=len(results),
                baseline_avg_score=baseline_avg,
                baseline_score_std=baseline_std,
                attacks_attempted=0,
                attacks_applicable=0,
                attacks_comparable=0,
                attacks_successful=0,
                total_rubrics_met_baseline=total_rubrics_met,
                total_rubrics_degraded=0,
            )

        # Attack statistics
        attacks_attempted = len(results)
        applicable_results = [
            r for r in results
            if r.attack_result and r.attack_result.applicable is True
        ]
        comparable_results = [
            r for r in applicable_results
            if r.rubric_attack_opportunities > 0
        ]
        attacks_applicable = len(applicable_results)
        attacks_comparable = len(comparable_results)
        attacks_successful = sum(1 for r in comparable_results if r.attack_successful)

        # Polarity-aware rubric attack statistics
        total_attackable_rubrics = sum(
            r.rubric_attack_opportunities for r in results
        )
        total_rubric_attack_successes = sum(
            r.rubric_attack_successes for r in results
        )
        total_rubrics_degraded = total_rubric_attack_successes

        score_comparable_results = [
            r for r in results
            if r.score_change is not None
        ]
        score_comparable_cases = len(score_comparable_results)
        comparable_baseline_scores = [
            r.baseline_score for r in score_comparable_results
            if r.baseline_score is not None
        ]
        attacked_scores = [
            r.attacked_score for r in score_comparable_results
            if r.attacked_score is not None
        ]
        attacked_avg = float(np.mean(attacked_scores)) if attacked_scores else None
        attacked_std = (
            compute_bootstrap_std(
                attacked_scores,
                seed=self.bootstrap_seed,
            )
            if attacked_scores else None
        )

        if comparable_baseline_scores:
            baseline_avg = float(np.mean(comparable_baseline_scores))
            baseline_std = compute_bootstrap_std(
                comparable_baseline_scores,
                seed=self.bootstrap_seed,
            )
        elif score_comparable_cases == 0:
            baseline_avg = 0.0
            baseline_std = 0.0

        # Score degradation
        score_changes = [r.score_change for r in score_comparable_results if r.score_change is not None]
        avg_degradation = float(np.mean(score_changes)) if score_changes else None

        return HealthBenchRobustnessSummary(
            total_cases=len(results),
            baseline_avg_score=baseline_avg,
            baseline_score_std=baseline_std,
            attacks_attempted=attacks_attempted,
            attacks_applicable=attacks_applicable,
            attacks_comparable=attacks_comparable,
            attacks_successful=attacks_successful,
            total_rubrics_met_baseline=total_rubrics_met,
            total_rubrics_degraded=total_rubrics_degraded,
            total_attackable_rubrics=total_attackable_rubrics,
            total_rubric_attack_successes=total_rubric_attack_successes,
            score_comparable_cases=score_comparable_cases,
            attacked_avg_score=attacked_avg,
            attacked_score_std=attacked_std,
            avg_score_degradation=avg_degradation
        )

    def _print_summary(self, summary: HealthBenchRobustnessSummary, baseline_only: bool):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        print(f"\n📊 BASELINE STATISTICS")
        print(f"  Total cases: {summary.total_cases}")
        if baseline_only or summary.score_comparable_cases > 0:
            print(f"  Average score: {summary.baseline_avg_score:.3f} ± {summary.baseline_score_std:.3f}")
        else:
            print("  Average score: unavailable for attacked-only results")
        print(f"  Total rubrics met: {summary.total_rubrics_met_baseline}")

        if not baseline_only:
            print(f"\n🎯 ATTACK STATISTICS")
            print(f"  Attacks attempted: {summary.attacks_attempted}")
            print(f"  Attacks applicable: {summary.attacks_applicable}")
            print(f"  Comparable attacks: {summary.attacks_comparable}")
            print(f"  Attacks successful (target rubric violated): {summary.attacks_successful}")
            print(f"  Case attack success rate: {summary.case_attack_success_rate * 100:.1f}%")

            print(f"\n📉 RUBRIC ATTACK RESULTS")
            print(f"  Targeted rubric opportunities: {summary.total_attackable_rubrics}")
            print(f"  Rubric attack successes: {summary.total_rubric_attack_successes}")
            print(f"  Rubric attack success rate: {summary.rubric_attack_success_rate * 100:.1f}%")

            if summary.score_comparable_cases > 0:
                print(f"\n📈 SCORE IMPACT")
                print(f"  Score-comparable cases: {summary.score_comparable_cases}")
                if summary.attacked_avg_score is not None:
                    print(f"  Attacked average score: {summary.attacked_avg_score:.3f}")
                if summary.avg_score_degradation is not None:
                    print(f"  Average score degradation: {summary.avg_score_degradation:.3f}")

    def save_results(
        self,
        results: List[HealthBenchRobustnessResult],
        summary: HealthBenchRobustnessSummary,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save HealthBench evaluation results to JSON file.

        Standardized save interface used across all pipeline classes.

        Args:
            results: List of HealthBenchRobustnessResult objects
            summary: HealthBenchRobustnessSummary object
            output_path: Path to output JSON file
            metadata: Optional metadata dict (can include config for reproducibility)

        Example:
            >>> results, summary = pipeline.run_baseline(...)
            >>> pipeline.save_results(
            ...     results, summary,
            ...     "logs/gpt-4o_healthbench_baseline_all_20251130.json",
            ...     metadata={"config": config.to_dict()}
            ... )
        """
        from med_red_team.shared.io import (
            atomic_write_result_envelope,
            ensure_result_metadata,
        )
        from datetime import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inferred_phase = "attack" if any(r.attack_result is not None for r in results) else "baseline"
        result_metadata = ensure_result_metadata(
            metadata,
            axis="healthbench",
            phase=inferred_phase,
        )
        result_metadata.setdefault("timestamp", datetime.now().isoformat())

        atomic_write_result_envelope(
            output_path,
            metadata=result_metadata,
            summary=summary.to_dict(),
            data=[r.to_dict() for r in results],
            data_key="results",
        )

        if self.verbose:
            print(f"\n✓ Results saved to: {output_path}")
