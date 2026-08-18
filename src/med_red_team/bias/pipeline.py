"""Bias evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from med_red_team.bias.data import (
    BiasReferenceResult,
    BiasResult,
    BIAS_STATUS_COMPLETE,
    BIAS_STATUS_MANIPULATION_FAILED,
    BIAS_STATUS_NOT_APPLICABLE,
    BIAS_STATUS_RETRYABLE,
    BiasSummary,
    BiasTestCase,
    bias_reference_matches_current_source,
    create_attack_summary,
    create_baseline_summary,
    load_baseline_cache,
    reclassify_bias_attack_result,
    reclassify_bias_reference_result,
    save_baseline_cache,
    validate_bias_attack_reference_binding,
)
from med_red_team.bias.grader import BiasGrader
from med_red_team.bias.prompts import BIAS_TESTING_SYSTEM_PROMPT
from med_red_team.bias.utils import (
    INVALID_VOTE_OUTCOME,
    NO_WINNER_VOTE_OUTCOME,
    VALID_VOTE_OUTCOME,
    calculate_vote_entropy,
    clean_answer_response,
    get_full_choice_text,
    summarize_votes,
)
from med_red_team.models import ModelExecutionError
from med_red_team.pipeline import EvaluationPipeline
from med_red_team.shared.checkpoints import merge_unique
from med_red_team.testee import Testee
from med_red_team.utils import evaluate_items_fail_fast


class BiasPipeline(EvaluationPipeline):
    """Bias pipeline with explicit vote normalization and comparable-only grading."""

    def __init__(
        self,
        testee: Testee,
        grader: BiasGrader,
        attack_strategies: List,
        vote_num_baseline: int = 5,
        vote_num_attack: int = 5,
        system_prompt: str = BIAS_TESTING_SYSTEM_PROMPT,
        verbose: bool = True,
    ):
        super().__init__(testee, grader, verbose)
        self.attack_strategies = attack_strategies or []
        self.vote_num_baseline = vote_num_baseline
        self.vote_num_attack = vote_num_attack
        self.system_prompt = system_prompt

        if self.verbose:
            print("[BiasPipeline] Initialized")
            print(f"  Testee: {testee.model_id}")
            print(f"  Attack strategies: {len(self.attack_strategies)}")
            print(f"  Baseline votes: {vote_num_baseline}")
            print(f"  Attack votes: {vote_num_attack}")

    def _baseline_cache_metadata(
        self,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Describe every execution input that can change cached baseline votes."""
        testee_config = getattr(self.testee, "config", None)
        if is_dataclass(testee_config):
            serialized_config: Any = asdict(testee_config)
        elif hasattr(testee_config, "to_dict"):
            serialized_config = testee_config.to_dict()
        elif isinstance(testee_config, dict) or testee_config is None:
            serialized_config = testee_config
        else:
            serialized_config = repr(testee_config)
        return {
            "testee_model": getattr(self.testee, "model_id", None),
            "testee_config": serialized_config,
            "testee_system_prompt": self.system_prompt,
            "vote_num_baseline": self.vote_num_baseline,
            "source": dict(source_metadata or {}),
        }

    def rebuild_baseline_cache(
        self,
        results: List[BiasReferenceResult],
        cache_file: str,
        cache_source_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write a complete validated baseline population to the cache."""
        save_baseline_cache(
            results,
            cache_file,
            metadata=self._baseline_cache_metadata(cache_source_metadata),
        )

    def _response_vote_text(self, response: Any) -> str:
        metadata = getattr(response, "metadata", {}) or {}
        if isinstance(metadata, dict) and "parse_error" in metadata:
            return ""
        return clean_answer_response(getattr(response, "final_answer", ""))

    def _collect_votes(
        self,
        prompt: str,
        num_votes: int,
        indent: str = "  ",
    ) -> List[str]:
        responses: List[str] = []
        for vote_idx in range(num_votes):
            response = self.testee.answer(prompt, system_prompt=self.system_prompt)
            cleaned = self._response_vote_text(response)
            responses.append(cleaned)
            if self.verbose:
                printable = cleaned or "<invalid>"
                print(f"{indent}Vote {vote_idx + 1}: {printable}")
        return responses

    def run_baseline(
        self,
        test_cases: List[BiasTestCase],
        load_from_cache: bool = False,
        cache_file: Optional[str] = None,
        save_cache: bool = True,
        cache_source_metadata: Optional[Dict[str, Any]] = None,
        progress_callback=None,
        save_every: int = 10,
        completed_baseline_keys: Optional[set[tuple[str, int]]] = None,
        existing_results: Optional[List[BiasReferenceResult]] = None,
        expected_population: Optional[int] = None,
    ) -> Tuple[List[BiasReferenceResult], BiasSummary]:
        if self.verbose:
            print(f"\n{'=' * 80}")
            print("BASELINE TESTING (Round 1)")
            print(f"{'=' * 80}")
            print(f"Testing {len(test_cases)} cases")
            print(f"Votes per question: {self.vote_num_baseline}")

        existing_results = list(existing_results or [])
        current_cases_by_id = {
            test_case.case_id: test_case
            for test_case in test_cases
        }
        validated_existing_results: List[BiasReferenceResult] = []
        validated_existing_keys: set[tuple[str, int]] = set()
        for result in existing_results:
            current_case = current_cases_by_id.get(result.case_id)
            if not bias_reference_matches_current_source(result, current_case):
                continue
            if reclassify_bias_reference_result(
                result,
                configured_total_attempts=self.vote_num_baseline,
            ):
                validated_existing_results.append(result)
                validated_existing_keys.add((result.case_id, result.question_idx))
        existing_results = validated_existing_results
        if completed_baseline_keys is None:
            completed_baseline_keys = validated_existing_keys
        else:
            completed_baseline_keys = set(completed_baseline_keys) & validated_existing_keys
        cache_metadata = self._baseline_cache_metadata(cache_source_metadata)
        cache: Dict[str, BiasReferenceResult] = {}
        if load_from_cache and cache_file:
            cache = load_baseline_cache(
                cache_file,
                expected_metadata=cache_metadata,
            )

        eval_items = [
            (test_case, q_idx)
            for test_case in test_cases
            for q_idx, _question_text in enumerate(test_case.questions)
            if test_case.has_valid_question(q_idx)
            and (test_case.case_id, q_idx) not in completed_baseline_keys
        ]

        def _evaluate_baseline_question(item, _idx):
            test_case, q_idx = item
            cache_key = f"{test_case.case_id}_q{q_idx}"
            if cache_key in cache:
                cached_result = cache[cache_key]
                if (
                    bias_reference_matches_current_source(cached_result, test_case)
                    and reclassify_bias_reference_result(
                        cached_result,
                        configured_total_attempts=self.vote_num_baseline,
                    )
                ):
                    if self.verbose:
                        print("  (from strict-majority cache)")
                    return cached_result
                if self.verbose:
                    print("  (cache entry incompatible; recollecting votes)")

            case_text = test_case.format_full_case(
                question_idx=q_idx,
                patient_placeholder="patient",
            )
            question_text = test_case.questions[q_idx]
            ref_responses = self._collect_votes(case_text, self.vote_num_baseline, indent="  ")
            vote_summary = summarize_votes(
                ref_responses,
                question_text=question_text,
                configured_total_attempts=self.vote_num_baseline,
            )
            ref_vote_entropy = calculate_vote_entropy(vote_summary.valid_votes)

            if vote_summary.outcome == VALID_VOTE_OUTCOME:
                unbiased_model_choice = get_full_choice_text(
                    vote_summary.majority_vote,
                    question_text,
                )
            elif vote_summary.outcome == NO_WINNER_VOTE_OUTCOME:
                unbiased_model_choice = "No strict majority winner"
            else:
                unbiased_model_choice = "Insufficient valid votes"

            if self.verbose:
                printable = vote_summary.majority_vote or vote_summary.outcome
                print(f"  Majority vote: {printable}")
                print(f"  Vote entropy: {ref_vote_entropy:.3f}")
                print(f"  Invalid votes excluded: {len(vote_summary.invalid_votes)}")

            return BiasReferenceResult(
                case_id=test_case.case_id,
                question_idx=q_idx,
                category=test_case.category,
                case_text=case_text,
                ref_responses=vote_summary.raw_responses,
                ref_majority_vote=vote_summary.majority_vote,
                ref_vote_entropy=ref_vote_entropy,
                unbiased_model_choice=unbiased_model_choice,
                test_case=test_case,
                ref_vote_outcome=vote_summary.outcome,
                ref_valid_vote_count=len(vote_summary.valid_votes),
                ref_invalid_vote_count=len(vote_summary.invalid_votes),
                ref_vote_counts=dict(vote_summary.counts),
                baseline_status=BIAS_STATUS_COMPLETE,
                ref_configured_total_attempts=vote_summary.configured_total_attempts,
                ref_required_votes=vote_summary.required_votes,
            )

        results = evaluate_items_fail_fast(
            items=eval_items,
            evaluate_fn=_evaluate_baseline_question,
            description="Baseline testing",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose,
            use_tqdm=self.verbose,
        )

        merged_results = merge_unique(
            existing_results,
            results,
            key=lambda result: (result.case_id, result.question_idx),
        )
        resolved_population = (
            sum(
                test_case.num_valid_questions()
                for test_case in test_cases
            )
            if expected_population is None
            else expected_population
        )
        summary = create_baseline_summary(
            merged_results,
            expected_population=resolved_population,
        )

        if save_cache and cache_file:
            self.rebuild_baseline_cache(
                merged_results,
                cache_file,
                cache_source_metadata,
            )

        if self.verbose:
            print(f"\n[INFO] Baseline testing complete: {len(results)} question results")

        return merged_results, summary

    def run_attack(
        self,
        baseline_results: List[BiasReferenceResult],
        strategies: Optional[List] = None,
        progress_callback=None,
        save_every: int = 10,
        completed_attack_keys: Optional[set[tuple[str, int, str]]] = None,
        existing_results: Optional[List[BiasResult]] = None,
        attack_population: Optional[int] = None,
        strategy_populations: Optional[Dict[str, int]] = None,
        question_population: Optional[int] = None,
        baseline_population: Optional[int] = None,
        baseline_valid_count: Optional[int] = None,
    ) -> Tuple[List[BiasResult], BiasSummary]:
        strategies = strategies or self.attack_strategies
        existing_results = list(existing_results or [])
        baseline_by_key = {
            (result.case_id, result.question_idx): result
            for result in baseline_results
        }
        validated_existing_results: List[BiasResult] = []
        validated_existing_keys: set[tuple[str, int, str]] = set()
        for result in existing_results:
            baseline = baseline_by_key.get((result.case_id, result.question_idx))
            if baseline is None or baseline.test_case is None:
                continue
            if validate_bias_attack_reference_binding(result, baseline) is not None:
                continue
            if (
                result.ref_vote_outcome != baseline.ref_vote_outcome
                or result.ref_majority_vote != baseline.ref_majority_vote
            ):
                continue
            has_votes = bool(result.manipulated_responses)
            vote_outcome = result.evaluation_outcome in {
                "bias_detected",
                "no_bias",
                "attack_invalid",
                "attack_no_winner",
            }
            accepted = False
            if has_votes or vote_outcome:
                accepted = reclassify_bias_attack_result(
                    result,
                    configured_total_attempts=self.vote_num_attack,
                    question_text=baseline.test_case.questions[baseline.question_idx],
                )
            elif (
                not result.manipulated_responses
                and not result.manipulated_majority_vote
                and result.manipulated_valid_vote_count == 0
                and result.manipulated_invalid_vote_count == 0
                and not result.manipulated_vote_counts
                and not result.bias_detected
                and result.skipped
            ):
                expected_baseline_outcome = f"baseline_{baseline.ref_vote_outcome}"
                accepted = (
                    baseline.ref_vote_outcome != VALID_VOTE_OUTCOME
                    and result.evaluation_outcome == expected_baseline_outcome
                    and result.attack_status == BIAS_STATUS_COMPLETE
                ) or (
                    result.evaluation_outcome == "attack_excluded"
                    and result.attack_status == BIAS_STATUS_NOT_APPLICABLE
                ) or (
                    result.evaluation_outcome == "manipulation_failed"
                    and result.attack_status == BIAS_STATUS_MANIPULATION_FAILED
                )
            if accepted:
                validated_existing_results.append(result)
                validated_existing_keys.add(
                    (result.case_id, result.question_idx, result.attack_strategy)
                )
        existing_results = validated_existing_results
        if completed_attack_keys is None:
            completed_attack_keys = validated_existing_keys
        else:
            completed_attack_keys = set(completed_attack_keys) & validated_existing_keys

        if not strategies:
            raise ValueError("No bias attack strategies configured")

        if self.verbose:
            print(f"\n{'=' * 80}")
            print("ATTACK TESTING (Round 2)")
            print(f"{'=' * 80}")
            print(f"Baseline results: {len(baseline_results)} questions")
            print(f"Attacking with: {', '.join([s.name for s in strategies])}")

        if not baseline_results:
            if self.verbose:
                print("[WARNING] No baseline results, nothing to attack")
            return existing_results, create_attack_summary(
                existing_results,
                [s.name for s in strategies],
                attack_population=attack_population or 0,
                strategy_populations=strategy_populations,
                question_population=question_population or 0,
                baseline_population=baseline_population or 0,
                baseline_valid_count=baseline_valid_count or 0,
            )

        eval_items = [
            (baseline_result, strategy, baseline_idx * len(strategies) + strat_idx + 1)
            for baseline_idx, baseline_result in enumerate(baseline_results)
            for strat_idx, strategy in enumerate(strategies)
            if (baseline_result.case_id, baseline_result.question_idx, strategy.name)
            not in completed_attack_keys
        ]

        resolved_strategy_names = [s.name for s in strategies]
        resolved_attack_population = (
            len(baseline_results) * len(strategies)
            if attack_population is None
            else attack_population
        )
        resolved_strategy_populations = strategy_populations or {
            strategy_name: len(baseline_results)
            for strategy_name in resolved_strategy_names
        }
        resolved_question_population = (
            len({
                (result.case_id, result.question_idx)
                for result in baseline_results
                if result.ref_vote_outcome == VALID_VOTE_OUTCOME
            })
            if question_population is None
            else question_population
        )
        resolved_baseline_population = (
            len({(result.case_id, result.question_idx) for result in baseline_results})
            if baseline_population is None
            else baseline_population
        )
        resolved_baseline_valid_count = (
            resolved_question_population
            if baseline_valid_count is None
            else baseline_valid_count
        )

        if not eval_items:
            return existing_results, create_attack_summary(
                existing_results,
                resolved_strategy_names,
                attack_population=resolved_attack_population,
                strategy_populations=resolved_strategy_populations,
                question_population=resolved_question_population,
                baseline_population=resolved_baseline_population,
                baseline_valid_count=resolved_baseline_valid_count,
            )

        def _evaluate_attack_item(item, _idx):
            baseline_result, strategy, sample_number = item
            return self._evaluate_attack_case(baseline_result, strategy, sample_number)

        attack_results = evaluate_items_fail_fast(
            items=eval_items,
            evaluate_fn=_evaluate_attack_item,
            description="Attack testing",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose,
            use_tqdm=self.verbose,
        )

        merged_results = merge_unique(
            existing_results,
            attack_results,
            key=lambda result: (
                result.case_id,
                result.question_idx,
                result.attack_strategy,
            ),
        )
        summary = create_attack_summary(
            merged_results,
            resolved_strategy_names,
            attack_population=resolved_attack_population,
            strategy_populations=resolved_strategy_populations,
            question_population=resolved_question_population,
            baseline_population=resolved_baseline_population,
            baseline_valid_count=resolved_baseline_valid_count,
        )
        return merged_results, summary

    def _evaluate_attack_case(
        self,
        baseline_result: BiasReferenceResult,
        strategy,
        sample_number: int,
    ) -> BiasResult:
        result = BiasResult.from_baseline(baseline_result, sample_number, strategy.name)

        if baseline_result.ref_vote_outcome != VALID_VOTE_OUTCOME:
            result.skipped = True
            result.evaluation_outcome = f"baseline_{baseline_result.ref_vote_outcome}"
            result.attack_status = BIAS_STATUS_COMPLETE
            result.attack_failure_category = result.evaluation_outcome
            if baseline_result.ref_vote_outcome == NO_WINNER_VOTE_OUTCOME:
                result.skip_reason = "Baseline vote had no strict majority winner"
            else:
                result.skip_reason = (
                    "Baseline vote had insufficient valid answers for strict majority"
                )
            result.comparison_reason = result.skip_reason
            return result

        try:
            if baseline_result.test_case is None:
                raise ValueError("Original test case not available in baseline result")

            context = baseline_result.to_attack_context()
            modified_test_case = strategy.apply(baseline_result.test_case, context)

            if modified_test_case.metadata.get("attack_excluded", False):
                result.skipped = True
                result.skip_reason = modified_test_case.metadata.get(
                    "skip_reason",
                    "Attack excluded",
                )
                result.evaluation_outcome = "attack_excluded"
                result.attack_status = BIAS_STATUS_NOT_APPLICABLE
                result.attack_failure_category = modified_test_case.metadata.get(
                    "failure_category",
                    "attack_excluded",
                )
                result.comparison_reason = result.skip_reason
                if self.verbose:
                    print(f"    {result.skip_reason}")
                return result

            if modified_test_case.metadata.get("manipulation_failed", False):
                result.skipped = True
                result.skip_reason = modified_test_case.metadata.get(
                    "skip_reason",
                    "Manipulation failed",
                )
                result.evaluation_outcome = "manipulation_failed"
                result.attack_status = BIAS_STATUS_MANIPULATION_FAILED
                result.attack_failure_category = modified_test_case.metadata.get(
                    "failure_category",
                    "manipulation_failed",
                )
                result.comparison_reason = result.skip_reason
                if self.verbose:
                    print("    Manipulation failed")
                return result

            result.agent_advice = modified_test_case.get_attack_advice()
            manipulated_case_text = modified_test_case.format_full_case(
                question_idx=baseline_result.question_idx,
                patient_placeholder="patient",
            )
            result.manipulated_case = manipulated_case_text
            if manipulated_case_text == baseline_result.case_text:
                result.skipped = True
                result.skip_reason = "Attack manipulation produced no final prompt change"
                result.evaluation_outcome = "manipulation_failed"
                result.attack_status = BIAS_STATUS_MANIPULATION_FAILED
                result.attack_failure_category = "unchanged_final_prompt"
                result.comparison_reason = result.skip_reason
                return result

            if self.verbose:
                print(f"    Agent advice: {result.agent_advice}")
                print(
                    f"    Manipulated case length: {len(manipulated_case_text.split())} words"
                )

            manipulated_responses = self._collect_votes(
                manipulated_case_text,
                self.vote_num_attack,
                indent="    ",
            )
            question_text = baseline_result.test_case.questions[baseline_result.question_idx]
            vote_summary = summarize_votes(
                manipulated_responses,
                question_text=question_text,
                configured_total_attempts=self.vote_num_attack,
            )
            manipulated_vote_entropy = calculate_vote_entropy(vote_summary.valid_votes)

            result.manipulated_responses = vote_summary.raw_responses
            result.manipulated_majority_vote = vote_summary.majority_vote
            result.manipulated_vote_entropy = manipulated_vote_entropy
            result.manipulated_vote_outcome = vote_summary.outcome
            result.manipulated_valid_vote_count = len(vote_summary.valid_votes)
            result.manipulated_invalid_vote_count = len(vote_summary.invalid_votes)
            result.manipulated_vote_counts = dict(vote_summary.counts)
            result.manipulated_configured_total_attempts = (
                vote_summary.configured_total_attempts
            )
            result.manipulated_required_votes = vote_summary.required_votes

            if self.verbose:
                printable = vote_summary.majority_vote or vote_summary.outcome
                print(f"    Majority vote: {printable}")
                print(f"    Vote entropy: {manipulated_vote_entropy:.3f}")
                print(f"    Invalid votes excluded: {len(vote_summary.invalid_votes)}")

            if vote_summary.outcome != VALID_VOTE_OUTCOME:
                result.skipped = True
                result.evaluation_outcome = f"attack_{vote_summary.outcome}"
                if vote_summary.outcome == NO_WINNER_VOTE_OUTCOME:
                    result.skip_reason = "Attack vote had no strict majority winner"
                else:
                    result.skip_reason = (
                        "Attack vote had insufficient valid answers for "
                        "strict majority"
                    )
                result.comparison_reason = result.skip_reason
                result.attack_status = BIAS_STATUS_COMPLETE
                result.attack_failure_category = vote_summary.outcome
                return result

            grading_context = baseline_result.to_grading_context(
                attack_vote_entropy=manipulated_vote_entropy,
                attack_vote_outcome=vote_summary.outcome,
            )
            grading_result = self.grader.grade(
                modified_test_case,
                vote_summary.majority_vote,
                grading_context,
            )
            outcome = grading_result.metadata.get("evaluation_outcome", "no_bias")
            result.evaluation_outcome = str(outcome)
            result.comparison_reason = grading_result.reasoning
            result.bias_detected = bool(
                grading_result.metadata.get("bias_detected", False)
            )
            result.skipped = not bool(grading_result.metadata.get("comparable", True))
            result.attack_status = BIAS_STATUS_COMPLETE
            result.attack_failure_category = None
            if result.skipped and not result.skip_reason:
                result.skip_reason = grading_result.reasoning

            if self.verbose:
                if result.evaluation_outcome == "bias_detected":
                    print(
                        "    Bias detected "
                        f"({baseline_result.ref_majority_vote} → {vote_summary.majority_vote})"
                    )
                elif result.evaluation_outcome == "no_bias":
                    print(f"    No bias (both {baseline_result.ref_majority_vote})")
                else:
                    print(f"    {grading_result.reasoning}")

        except ModelExecutionError:
            raise
        except Exception as exc:
            if self.verbose:
                print(f"    [ERROR] Attack failed: {exc}")
            result.skipped = True
            result.evaluation_outcome = "error"
            result.attack_status = BIAS_STATUS_RETRYABLE
            result.attack_failure_category = (
                "attacker_generation_error"
                if isinstance(exc, RuntimeError)
                else "attacker_error"
            )
            result.skip_reason = f"Error in attack: {str(exc)[:100]}"
            result.comparison_reason = result.skip_reason

        return result

    def _compute_summary(
        self,
        results: Sequence[Any],
        baseline_only: bool = False,
        attack_strategies: Optional[List[str]] = None,
    ) -> BiasSummary:
        if baseline_only:
            return create_baseline_summary(list(results))
        return create_attack_summary(list(results), attack_strategies)

    def save_results(
        self,
        results: List[Any],
        summary: BiasSummary,
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        announce: Optional[bool] = None,
    ):
        from datetime import datetime

        from med_red_team.shared.io import (
            atomic_write_result_envelope,
            ensure_result_metadata,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inferred_phase = None
        if metadata:
            inferred_phase = metadata.get("phase") or metadata.get("mode")
        if inferred_phase is None and results:
            inferred_phase = "attack" if isinstance(results[0], BiasResult) else "baseline"
        inferred_phase = inferred_phase or "baseline"

        if summary.phase != inferred_phase:
            raise ValueError(
                "Bias summary phase does not match result metadata "
                f"({summary.phase!r} != {inferred_phase!r})"
            )
        if summary.total_samples != len(results):
            raise ValueError(
                "Bias summary total_samples does not match the persisted result count "
                f"({summary.total_samples!r} != {len(results)!r})"
            )
        result_metadata = ensure_result_metadata(
            metadata,
            axis="bias",
            phase=inferred_phase,
        )
        result_metadata.setdefault("timestamp", datetime.now().isoformat())

        atomic_write_result_envelope(
            output_path,
            metadata=result_metadata,
            summary=summary.to_dict(),
            data=[result.to_dict() for result in results],
            data_key="results",
        )

        if self.verbose if announce is None else announce:
            print(f"\nResults saved to: {output_path}")
