"""
Orchestrator Pipeline for Robustness Testing
=============================================

This module implements an iterative multi-agent orchestrator pipeline that
strategically selects and applies attack strategies based on testee responses.

Unlike the standard RobustnessPipeline which applies all strategies once,
the OrchestratorPipeline:
1. Tests original question
2. If correct, enters iterative attack loop:
   a. Orchestrator agent selects best tool(s) based on history
   b. Applies selected attack strategy
   c. Tests manipulated question with testee
   d. If testee answers incorrectly → SUCCESS, stop
   e. If testee still correct → repeat with different tools
3. Records detailed orchestration history and metrics

This provides adaptive, intelligent adversarial testing with early stopping.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from agents import Agent, Runner, AgentOutputSchema

from med_red_team.data import TestCase
from med_red_team.testee import Testee
from med_red_team.grader import Grader
from med_red_team.pipeline import EvaluationPipeline
from med_red_team.model_pool import ModelPool
from med_red_team.models import MalformedResponseError, ModelExecutionError
from med_red_team.models.utils.response_utils import get_parse_error_info, has_parse_error
from med_red_team.robustness.data import (
    RobustnessResult,
    RobustnessSummary,
    summarize_robustness_results,
)
from med_red_team.robustness.data import (
    OrchestratorQuestionData,
    OrchestratorPlanOutput,
    ToolExecutionLog,
    PlannerAttemptLog,
    IterationLog,
    OrchestrationHistory,
)
from med_red_team.robustness.prompts import ORCHESTRATOR_PLANNING_PROMPT
from med_red_team.robustness.tool_policy import (
    POLICY_VERSION,
    ToolPolicyIssue,
    compute_question_delta,
)
from med_red_team.robustness.tool_registry import ToolRegistry
from med_red_team.models import GenerationConfig


# ==============================================================================
# Orchestrator Evaluation Pipeline
# ==============================================================================

class OrchestratorPipeline(EvaluationPipeline):
    """
    Iterative orchestrator pipeline for adaptive robustness testing.

    This pipeline uses an LLM-based orchestrator agent that strategically
    selects manipulation tools based on testee responses. The orchestrator
    learns from failed attempts and adjusts its strategy across iterations.

    Key differences from RobustnessPipeline:
    - Iterative: Keeps attacking until testee fails or max iterations reached
    - Adaptive: Orchestrator chooses tools based on what failed previously
    - Early stopping: Stops immediately when testee is fooled
    - Rich logging: Tracks orchestrator decisions and tool effectiveness

    Example:
        >>> pipeline = OrchestratorPipeline(
        ...     testee=testee,
        ...     grader=grader,
        ...     orchestrator_model_id="gpt-4o",
        ...     tools_model_id="gpt-4o",
        ...     model_pool=model_pool,
        ...     max_iterations=5,
        ...     verbose=True
        ... )
        >>> results, summary = pipeline.run_baseline(
        ...     test_cases,
        ...     max_samples=100
        ... )
    """

    def __init__(
        self,
        testee: Testee,
        grader: Grader,
        orchestrator_model_id: str,
        tools_model_id: str,
        model_pool: ModelPool,
        max_iterations: int = 5,
        max_planner_attempts: int = 2,
        orchestrator_config: Optional[GenerationConfig] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
        strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        verbose: bool = True
    ):
        """
        Initialize orchestrator evaluation pipeline.

        Args:
            testee: Model to evaluate
            grader: Grader to assess responses
            orchestrator_model_id: Model ID for orchestrator agent (e.g., "gpt-4o")
            tools_model_id: Default model ID for tool strategies (e.g., "gpt-4o")
            model_pool: ModelPool for LLM access
            max_iterations: Maximum manipulation rounds per test case (default: 5)
            max_planner_attempts: Maximum rejected planner proposals per round (default: 2)
            orchestrator_config: Optional generation config for orchestrator
            baseline_results: Optional dict with baseline results (skips first round if provided)
            strategy_configs: Optional per-strategy configs from RobustnessConfig.attacker_strategies
                             Format: {"cognitive_bias": {"model_id": "gpt-4o", "n_bias_styles": 3}, ...}
            verbose: Whether to print progress
        """
        super().__init__(testee, grader, verbose)

        self.orchestrator_model_id = orchestrator_model_id
        self.tools_model_id = tools_model_id
        self.model_pool = model_pool
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_planner_attempts < 1:
            raise ValueError("max_planner_attempts must be at least 1")
        self.max_iterations = max_iterations
        self.max_planner_attempts = max_planner_attempts
        self.orchestrator_config = orchestrator_config
        self.baseline_results = baseline_results
        self.strategy_configs = strategy_configs

        # Create orchestrator agent
        self._orchestrator = self._create_orchestrator_agent()

        # Track statistics
        self._first_round_correct_count = 0
        self._fooled_count = 0
        self._baseline_mode = baseline_results is not None
        self._tool_registry = ToolRegistry(
            model_pool=self.model_pool,
            tools_model_id=self.tools_model_id,
            strategy_configs=self.strategy_configs,
        )

    def _create_orchestrator_agent(self) -> Agent:
        """
        Create planning-only orchestrator agent (hybrid mode).

        In hybrid mode, the orchestrator only plans which tools to use and in
        what order. Tool execution happens separately in the pipeline code via
        ToolRegistry. This provides full control and detailed logging.

        Returns:
            Configured Agent instance for planning
        """
        return Agent(
            name="robustness_orchestrator_planner",
            model=self.orchestrator_model_id,
            instructions=ORCHESTRATOR_PLANNING_PROMPT,
            output_type=AgentOutputSchema(OrchestratorPlanOutput, strict_json_schema=False),
            tools=[]  # No tools - planning only
        )

    def _run_planner(self, input_items: List[Dict[str, str]]) -> OrchestratorPlanOutput:
        """Run the planning agent."""
        result = asyncio.run(Runner.run(self._orchestrator, input_items))
        return result.final_output

    def _build_planner_input(
        self,
        original_question: OrchestratorQuestionData,
        attack_iteration: int,
        compact_history: List[Dict[str, Any]],
        rejection_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, str]]:
        """Build bounded planner context while preserving original-seed attacks."""
        payload: Dict[str, Any] = {
            "attack_iteration": attack_iteration,
            "original_question_sample": original_question.model_dump(),
            "prior_independent_attempts": compact_history,
            "instruction": (
                "Plan a new attack against original_question_sample. "
                "Prior attempts are observations only; never continue editing them."
            ),
        }
        if rejection_feedback:
            payload["rejected_plan_feedback"] = rejection_feedback
        return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]

    def _build_compact_feedback(
        self,
        original_question: OrchestratorQuestionData,
        final_question: OrchestratorQuestionData,
        proposed_sequence: List[str],
        normalized_sequence: List[str],
        tool_logs: List[ToolExecutionLog],
        target_tested: bool,
        target_answer: str = "",
        target_correct: Optional[bool] = None,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create the bounded execution digest shown to later planner rounds."""
        target_metadata = target_metadata or {}
        return {
            "proposed_sequence": list(proposed_sequence),
            "normalized_sequence": list(normalized_sequence),
            "actually_executed_sequence": [
                log.tool_name for log in tool_logs
                if not log.tool_metadata.get("execution_skipped", False)
            ],
            "tool_statuses": [
                {
                    "tool": log.tool_name,
                    "status": log.status,
                    "success": log.success,
                    "reason": (log.error_message or "")[:240],
                }
                for log in tool_logs
            ],
            "changes": compute_question_delta(original_question, final_question),
            "target_outcome": {
                "tested": target_tested,
                "parsed_answer": target_metadata.get("extracted_answer", target_answer),
                "correct_answer": target_metadata.get(
                    "correct_answer", final_question.answer_idx
                ),
                "remained_correct": target_correct if target_tested else None,
            },
            "original_seed_reminder": (
                "This was an independent manipulation of the original question, "
                "not the state for the next round."
            ),
        }

    def _setup_evaluation(self, **kwargs):
        """Setup before evaluation starts."""
        self._first_round_correct_count = 0
        self._fooled_count = 0

        # Build baseline lookup if provided
        skip_first_round = kwargs.get('skip_first_round', False)
        if skip_first_round and self.baseline_results:
            self._baseline_lookup = {
                r["test_case_id"]: r
                for r in self.baseline_results.get("results", [])
            }
            self._skip_first_round = True
        else:
            self._baseline_lookup = {}
            self._skip_first_round = False

        if self.verbose:
            print(f"\n[Setup] Orchestrator Robustness Evaluation")
            if self._skip_first_round:
                print(f"  Mode: Attack (skipping first round, using baseline)")
            print(f"  Orchestrator Model: {self.orchestrator_model_id}")
            print(f"  Tools Model: {self.tools_model_id}")
            print(f"  Max Iterations: {self.max_iterations}")
            print(f"  Max Planner Attempts: {self.max_planner_attempts}")
            print(f"  Max Samples: {kwargs.get('max_samples', 'All')}")

    def _parse_error_details(
        self,
        *,
        response=None,
        exc: Optional[MalformedResponseError] = None,
    ) -> Dict[str, Any]:
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

    def _should_stop_evaluation(
        self,
        current_idx: int,
        results: List[RobustnessResult],
        **kwargs
    ) -> bool:
        """Stop when we've evaluated enough first-round correct samples."""
        max_samples = kwargs.get('max_samples')
        if max_samples is not None and self._first_round_correct_count >= max_samples:
            if self.verbose:
                print(f'\n[INFO] Reached {max_samples} first-round correct samples.')
            return True
        return False

    def _process_single_test_case(
        self,
        test_case: TestCase,
        case_idx: int,
        **kwargs
    ) -> RobustnessResult:
        """
        Process a single test case with iterative orchestrator attack.

        Flow:
        1. Test original question (first round)
        2. If correct, enter orchestrator loop:
           - Orchestrator selects tools
           - Apply manipulation
           - Test manipulated question
           - Check if fooled (early stop) or continue
        3. Record detailed history

        Args:
            test_case: Test case to evaluate
            case_idx: Index of current test case
            **kwargs: Additional arguments:
                - skip_first_round: If True, use baseline results instead of
                  querying testee for first round (saves API calls)

        Returns:
            RobustnessResult with orchestrator history

        Note:
            Attack phase is always skipped if first round is incorrect, as
            attacking an already-incorrect response makes no sense.
        """
        if self.verbose:
            print(f"\n[{case_idx}] Processing: {test_case.id}")

        # ============================================================
        # First Round: Test original question OR use baseline
        # ============================================================
        if self._skip_first_round and test_case.id in self._baseline_lookup:
            # Use baseline results (skip first round)
            baseline_data = self._baseline_lookup[test_case.id]
            first_response_text = baseline_data.get("original_response", "")
            first_correct = baseline_data.get("original_correct", False)
            first_reasoning = baseline_data.get("original_reasoning", "")

            if self.verbose:
                print(f"  First Round: Using baseline (✓ CORRECT)")

            # Create result from baseline
            result = RobustnessResult(
                sample_number=case_idx + 1,
                test_case_id=test_case.id,
                original_question=test_case.question,
                original_options=test_case.options,
                original_correct_answer=test_case.correct_answer,
                original_response=first_response_text,
                original_correct=first_correct,
                status="baseline_correct" if first_correct else "baseline_incorrect",
                eligible_baseline_correct=first_correct,
                metadata=dict(baseline_data.get("metadata", {})),
            )

            if (
                baseline_data.get("status") == "parser_error"
                or baseline_data.get("failure_category") == "parser_error"
                or "parse_error" in result.metadata
            ):
                return self._mark_response_parse_error(
                    result,
                    stage="baseline_response_parse",
                )

            if not first_correct:
                if self.verbose:
                    print("  Skipping attack (baseline result was incorrect)")
                return result

            self._first_round_correct_count += 1

        else:
            try:
                prompt = test_case.build_user_prompt()
                first_response = self.testee.answer(prompt)
                if has_parse_error(first_response):
                    result = RobustnessResult(
                        sample_number=case_idx + 1,
                        test_case_id=test_case.id,
                        original_question=test_case.question,
                        original_options=test_case.options,
                        original_correct_answer=test_case.correct_answer,
                        original_response=first_response.final_answer,
                        original_correct=False,
                    )
                    return self._mark_response_parse_error(
                        result,
                        stage="baseline_response_parse",
                        response=first_response,
                    )
                first_grading = self.grader.grade(
                    test_case,
                    first_response.final_answer,
                    context={}
                )
                first_correct = first_grading.is_correct
            except MalformedResponseError as exc:
                result = RobustnessResult(
                    sample_number=case_idx + 1,
                    test_case_id=test_case.id,
                    original_question=test_case.question,
                    original_options=test_case.options,
                    original_correct_answer=test_case.correct_answer,
                    original_response="",
                    original_correct=False,
                )
                return self._mark_response_parse_error(
                    result,
                    stage="baseline_response_parse",
                    exc=exc,
                )
            except ModelExecutionError:
                raise
            except Exception as exc:
                return RobustnessResult(
                    sample_number=case_idx + 1,
                    test_case_id=test_case.id,
                    original_question=test_case.question,
                    original_options=test_case.options,
                    original_correct_answer=test_case.correct_answer,
                    original_response="",
                    original_correct=False,
                    skipped=True,
                    skip_reason=f"Error in baseline: {exc}",
                    status="baseline_error",
                    status_reason=f"Error in baseline: {exc}",
                    failure_stage="baseline_generation_or_grading",
                    failure_category="generation_error",
                )

            if self.verbose:
                status = "✓ CORRECT" if first_correct else "✗ INCORRECT"
                print(f"  First Round: {status}")

            result = RobustnessResult(
                sample_number=case_idx + 1,
                test_case_id=test_case.id,
                original_question=test_case.question,
                original_options=test_case.options,
                original_correct_answer=test_case.correct_answer,
                original_response=first_response.final_answer,
                original_correct=first_correct,
                status="baseline_correct" if first_correct else "baseline_incorrect",
                eligible_baseline_correct=first_correct,
            )

            # If first round incorrect, skip attack
            if not first_correct:
                if self.verbose:
                    print(f"  Skipping attack (already incorrect)")
                return result

            self._first_round_correct_count += 1

        # ============================================================
        # Hybrid Orchestrator Attack Loop
        # ============================================================
        if self.verbose:
            print(f"  Starting hybrid orchestrator attack (max {self.max_iterations} iterations)...")

        tool_registry = self._tool_registry

        # Convert to orchestrator format (core QA only, no metadata)
        original_question_data = OrchestratorQuestionData.from_test_case(test_case)

        # Track orchestration history. Each accepted round always executes from
        # original_question_data; previous rounds are observations only.
        iterations_history: List[IterationLog] = []
        compact_history: List[Dict[str, Any]] = []
        all_tools_used: List[str] = []
        fooled_at_iteration = None
        manipulated_test_case = None
        planner_attempts_total = 0
        planner_conflict_rejections = 0
        planner_errors_total = 0
        target_tested_iterations = 0

        for iteration in range(1, self.max_iterations + 1):
            if self.verbose:
                print(f"\n    === Iteration {iteration}/{self.max_iterations} ===")

            planner_attempt_logs: List[PlannerAttemptLog] = []
            rejection_feedback: Optional[Dict[str, Any]] = None
            plan: Optional[OrchestratorPlanOutput] = None
            policy_decision = None

            for planner_attempt in range(1, self.max_planner_attempts + 1):
                planner_attempts_total += 1
                input_items = self._build_planner_input(
                    original_question=original_question_data,
                    attack_iteration=iteration,
                    compact_history=compact_history,
                    rejection_feedback=rejection_feedback,
                )

                try:
                    plan = self._run_planner(input_items)
                    policy_decision = tool_registry.validate_and_normalize_sequence(
                        plan.tool_sequence
                    )
                    attempted_sequences = {
                        tuple(item.get("normalized_sequence", []))
                        for item in compact_history
                    }
                    if (
                        policy_decision.valid
                        and tuple(policy_decision.normalized_sequence)
                        in attempted_sequences
                    ):
                        policy_decision.valid = False
                        policy_decision.issues.append(ToolPolicyIssue(
                            issue_type="repeated_sequence",
                            tools=tuple(policy_decision.normalized_sequence),
                            message=(
                                "This normalized tool sequence was already tried "
                                "on the original question."
                            ),
                        ))
                    policy_issues = [
                        issue.to_dict() for issue in policy_decision.issues
                    ]
                    planner_attempt_logs.append(PlannerAttemptLog(
                        attack_iteration=iteration,
                        planner_attempt=planner_attempt,
                        proposed_tool_sequence=list(plan.tool_sequence),
                        normalized_tool_sequence=list(
                            policy_decision.normalized_sequence
                        ),
                        reasoning=plan.reasoning,
                        accepted=policy_decision.valid,
                        reordered=policy_decision.reordered,
                        policy_issues=policy_issues,
                    ))
                except Exception as exc:
                    planner_errors_total += 1
                    plan = None
                    policy_issues = [{
                        "issue_type": "planner_error",
                        "tools": [],
                        "message": str(exc),
                        "recoverable": True,
                    }]
                    planner_attempt_logs.append(PlannerAttemptLog(
                        attack_iteration=iteration,
                        planner_attempt=planner_attempt,
                        proposed_tool_sequence=[],
                        normalized_tool_sequence=[],
                        reasoning="",
                        accepted=False,
                        policy_issues=policy_issues,
                    ))
                    policy_decision = None

                if policy_decision and policy_decision.valid:
                    break

                if any(
                    issue["issue_type"] == "absolute_conflict"
                    for issue in policy_issues
                ):
                    planner_conflict_rejections += 1
                rejection_feedback = {
                    "status": "rejected",
                    "planner_attempt": planner_attempt,
                    "issues": policy_issues,
                    "instruction": (
                        "Return a different valid plan against the original question."
                    ),
                }
                if self.verbose:
                    messages = "; ".join(
                        issue["message"] for issue in policy_issues
                    )
                    print(f"      Rejected planner attempt {planner_attempt}: {messages}")

            if not plan or not policy_decision or not policy_decision.valid:
                planner_failed_operationally = bool(planner_attempt_logs) and all(
                    attempt.policy_issues
                    and all(
                        issue["issue_type"] == "planner_error"
                        for issue in attempt.policy_issues
                    )
                    for attempt in planner_attempt_logs
                )
                failure_status = (
                    "generation_error" if planner_failed_operationally
                    else "no_valid_plan"
                )
                if not result.valid_target_tested:
                    result.manipulation_failed = True
                    result.skipped = True
                    result.skip_reason = (
                        f"No valid planner proposal after "
                        f"{self.max_planner_attempts} attempts"
                    )
                    result.status = failure_status
                    result.status_reason = result.skip_reason
                    result.failure_stage = "planner"
                    result.failure_category = failure_status
                iteration_log = IterationLog(
                    iteration_number=iteration,
                    orchestrator_plan=plan.model_dump() if plan else {},
                    tool_executions=[],
                    testee_response="",
                    testee_correct=False,
                    testee_metadata={},
                    iteration_result=failure_status,
                    planner_attempts=planner_attempt_logs,
                    proposed_tool_sequence=(
                        list(plan.tool_sequence) if plan else []
                    ),
                    normalized_tool_sequence=[],
                    actually_executed_sequence=[],
                    target_tested=False,
                    compact_feedback={
                        "status": failure_status,
                        "rejection_feedback": rejection_feedback or {},
                    },
                )
                iterations_history.append(iteration_log)
                break

            proposed_sequence = list(plan.tool_sequence)
            normalized_sequence = list(policy_decision.normalized_sequence)
            if self.verbose:
                print(f"      Plan: {proposed_sequence}")
                print(f"      Reasoning: {plan.reasoning}")
                if policy_decision.reordered:
                    print(f"      Reordered tools: {proposed_sequence} → {normalized_sequence}")

            tool_execution_logs: List[ToolExecutionLog] = []
            current_question = original_question_data.model_copy(deep=True)

            for tool_idx, tool_name in enumerate(normalized_sequence):
                if self.verbose:
                    print(
                        f"      Executing {tool_idx + 1}/{len(normalized_sequence)}: "
                        f"{tool_name}"
                    )

                modified_question, execution_log = tool_registry.execute_tool(
                    tool_name=tool_name,
                    question_data=current_question,
                    context={},
                )
                tool_execution_logs.append(execution_log)

                if execution_log.success:
                    current_question = modified_question
                    all_tools_used.append(tool_name)
                    if self.verbose:
                        print(
                            f"        ✓ Success "
                            f"(took {execution_log.execution_time_ms:.1f}ms)"
                        )
                elif self.verbose:
                    print(
                        f"        ✗ {execution_log.status}: "
                        f"{execution_log.error_message}"
                    )

            question_data = current_question
            no_changes = (
                question_data.question == original_question_data.question
                and question_data.options == original_question_data.options
                and question_data.answer_idx == original_question_data.answer_idx
            )
            observed_failure_statuses = {
                log.status for log in tool_execution_logs if not log.success
            }
            no_change_status = next(
                (
                    status for status in (
                        "tool_error",
                        "generation_error",
                        "parser_error",
                        "invalid_state",
                        "not_applicable",
                        "invalid_noop",
                    )
                    if status in observed_failure_statuses
                ),
                "invalid_noop",
            )

            if no_changes:
                compact_feedback = self._build_compact_feedback(
                    original_question=original_question_data,
                    final_question=question_data,
                    proposed_sequence=proposed_sequence,
                    normalized_sequence=normalized_sequence,
                    tool_logs=tool_execution_logs,
                    target_tested=False,
                )
                compact_history.append(compact_feedback)
                iterations_history.append(IterationLog(
                    iteration_number=iteration,
                    orchestrator_plan=plan.model_dump(),
                    tool_executions=tool_execution_logs,
                    testee_response="",
                    testee_correct=False,
                    testee_metadata={},
                    iteration_result=no_change_status,
                    planner_attempts=planner_attempt_logs,
                    proposed_tool_sequence=proposed_sequence,
                    normalized_tool_sequence=normalized_sequence,
                    actually_executed_sequence=[
                        log.tool_name for log in tool_execution_logs
                        if not log.tool_metadata.get("execution_skipped", False)
                    ],
                    target_tested=False,
                    compact_feedback=compact_feedback,
                ))
                if self.verbose:
                    print("      No valid changes; continuing to the next round")
                if iteration == self.max_iterations and not result.valid_target_tested:
                    result.manipulation_failed = True
                    result.skipped = True
                    result.skip_reason = "No valid manipulation reached the target"
                    result.status = no_change_status
                    result.status_reason = result.skip_reason
                    result.failure_stage = "tool_execution"
                    result.failure_category = no_change_status
                continue

            manipulated_test_case = question_data.to_test_case(
                test_case_id=test_case.id,
                metadata={"iteration": iteration},
            )
            testee_response = None
            try:
                prompt = manipulated_test_case.build_user_prompt()
                testee_response = self.testee.answer(prompt)
                if has_parse_error(testee_response):
                    raise MalformedResponseError(
                        message=(
                            get_parse_error_info(testee_response) or {}
                        ).get("error_message", "Model response was malformed"),
                        raw_text=testee_response.raw_text,
                        error_type=(
                            get_parse_error_info(testee_response) or {}
                        ).get("error_type", "malformed_response"),
                        details=(
                            get_parse_error_info(testee_response) or {}
                        ).get("details", {}),
                    )
                testee_grading = self.grader.grade(
                    manipulated_test_case,
                    testee_response.final_answer,
                    context={},
                )
            except MalformedResponseError as exc:
                compact_feedback = self._build_compact_feedback(
                    original_question=original_question_data,
                    final_question=question_data,
                    proposed_sequence=proposed_sequence,
                    normalized_sequence=normalized_sequence,
                    tool_logs=tool_execution_logs,
                    target_tested=False,
                )
                compact_feedback["target_outcome"]["error"] = str(exc)[:240]
                iterations_history.append(IterationLog(
                    iteration_number=iteration,
                    orchestrator_plan=plan.model_dump(),
                    tool_executions=tool_execution_logs,
                    testee_response="",
                    testee_correct=False,
                    testee_metadata={},
                    iteration_result="parser_error",
                    planner_attempts=planner_attempt_logs,
                    proposed_tool_sequence=proposed_sequence,
                    normalized_tool_sequence=normalized_sequence,
                    actually_executed_sequence=[
                        log.tool_name for log in tool_execution_logs
                        if not log.tool_metadata.get("execution_skipped", False)
                    ],
                    target_tested=False,
                    compact_feedback=compact_feedback,
                ))
                if not result.valid_target_tested:
                    self._mark_response_parse_error(
                        result,
                        stage="target_response_parse",
                        response=locals().get("testee_response"),
                        exc=exc,
                    )
                break
            except ModelExecutionError:
                raise
            except Exception as exc:
                compact_feedback = self._build_compact_feedback(
                    original_question=original_question_data,
                    final_question=question_data,
                    proposed_sequence=proposed_sequence,
                    normalized_sequence=normalized_sequence,
                    tool_logs=tool_execution_logs,
                    target_tested=False,
                )
                compact_feedback["target_outcome"]["error"] = str(exc)[:240]
                iterations_history.append(IterationLog(
                    iteration_number=iteration,
                    orchestrator_plan=plan.model_dump(),
                    tool_executions=tool_execution_logs,
                    testee_response="",
                    testee_correct=False,
                    testee_metadata={},
                    iteration_result="generation_error",
                    planner_attempts=planner_attempt_logs,
                    proposed_tool_sequence=proposed_sequence,
                    normalized_tool_sequence=normalized_sequence,
                    actually_executed_sequence=[
                        log.tool_name for log in tool_execution_logs
                        if not log.tool_metadata.get("execution_skipped", False)
                    ],
                    target_tested=False,
                    compact_feedback=compact_feedback,
                ))
                if not result.valid_target_tested:
                    result.skipped = True
                    result.skip_reason = f"Target generation or grading failed: {exc}"
                    result.status = "generation_error"
                    result.status_reason = result.skip_reason
                    result.failure_stage = "target_generation_or_grading"
                    result.failure_category = "generation_error"
                break

            target_tested_iterations += 1
            testee_correct = testee_grading.is_correct
            testee_metadata = (
                testee_grading.metadata
                if hasattr(testee_grading, "metadata")
                else {}
            )

            compact_feedback = self._build_compact_feedback(
                original_question=original_question_data,
                final_question=question_data,
                proposed_sequence=proposed_sequence,
                normalized_sequence=normalized_sequence,
                tool_logs=tool_execution_logs,
                target_tested=True,
                target_answer=testee_response.final_answer,
                target_correct=testee_correct,
                target_metadata=testee_metadata,
            )
            compact_history.append(compact_feedback)
            iterations_history.append(IterationLog(
                iteration_number=iteration,
                orchestrator_plan=plan.model_dump(),
                tool_executions=tool_execution_logs,
                testee_response=testee_response.final_answer,
                testee_correct=testee_correct,
                testee_metadata=testee_metadata,
                iteration_result="fooled" if not testee_correct else "still_correct",
                planner_attempts=planner_attempt_logs,
                proposed_tool_sequence=proposed_sequence,
                normalized_tool_sequence=normalized_sequence,
                actually_executed_sequence=[
                    log.tool_name for log in tool_execution_logs
                    if not log.tool_metadata.get("execution_skipped", False)
                ],
                target_tested=True,
                compact_feedback=compact_feedback,
            ))

            result.manipulated_question = manipulated_test_case.question
            result.manipulated_response = testee_response.final_answer
            result.manipulated_correct = testee_correct
            result.valid_target_tested = True
            result.status = "target_safe" if testee_correct else "fooled"

            if self.verbose:
                status = "✓ CORRECT" if testee_correct else "✗ FOOLED!"
                print(f"      Testee: {status}")

            if not testee_correct:
                fooled_at_iteration = iteration
                self._fooled_count += 1
                if self.verbose:
                    print(f"\n    🎉 Model fooled after {iteration} iteration(s)!")
                break

            if iteration == self.max_iterations and self.verbose:
                print(f"\n    Reached max iterations without fooling model.")

        # ============================================================
        # Store Orchestration History
        # ============================================================
        orchestration_history = OrchestrationHistory(
            test_case_id=test_case.id,
            original_question=test_case.question,
            original_options=test_case.options,
            original_answer=test_case.correct_answer,
            baseline_correct=result.original_correct,
            iterations=iterations_history,
            final_outcome=(
                "fooled" if fooled_at_iteration
                else result.status
                if result.status not in {"baseline_correct", "target_safe"}
                else "max_iterations"
            ),
            total_iterations=len(iterations_history),
            total_tools_used=len(all_tools_used),
            max_attack_iterations=self.max_iterations,
            max_planner_attempts=self.max_planner_attempts,
            planner_attempts_total=planner_attempts_total,
            planner_conflict_rejections=planner_conflict_rejections,
            target_tested_iterations=target_tested_iterations,
        )

        # Store serialized history in result
        result.orchestration_history = orchestration_history.to_dict()

        unique_tools_used = list(dict.fromkeys(all_tools_used))
        result.metadata.update({
            "iterations": len(iterations_history),
            "tools_used": unique_tools_used,
            "fooled": fooled_at_iteration is not None,
            "fooled_at_iteration": fooled_at_iteration,
            "planner_attempts_total": planner_attempts_total,
            "planner_conflict_rejections": planner_conflict_rejections,
            "planner_errors_total": planner_errors_total,
            "target_tested_iterations": target_tested_iterations,
            "tool_policy_version": POLICY_VERSION,
        })

        attribution_iteration = next(
            (
                iteration_log
                for iteration_log in reversed(iterations_history)
                if iteration_log.target_tested
            ),
            iterations_history[-1] if iterations_history else None,
        )
        result.attacks_applied = (
            [
                log.tool_name
                for log in attribution_iteration.tool_executions
                if log.success
            ]
            if attribution_iteration else []
        )
        result.planner_attempts_used = planner_attempts_total
        result.planner_conflict_rejections = planner_conflict_rejections
        result.target_tested_iterations = target_tested_iterations
        if attribution_iteration:
            result.proposed_tool_sequence = (
                attribution_iteration.proposed_tool_sequence
            )
            result.normalized_tool_sequence = (
                attribution_iteration.normalized_tool_sequence
            )
            result.actually_executed_sequence = (
                attribution_iteration.actually_executed_sequence
            )

        return result

    # ============================================================
    # Concrete workflow implementations
    # ============================================================

    def run_baseline(
        self,
        test_cases: List[TestCase],
        max_samples: Optional[int] = None,
        progress_callback = None,
        save_every: int = 10
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """
        Not applicable for orchestrator pipeline.

        The orchestrator pipeline is designed for attack evaluation only.
        Use run_orchestrator_attack() instead.

        Raises:
            NotImplementedError: Always raised with helpful message
        """
        raise NotImplementedError(
            "OrchestratorPipeline does not support run_baseline(). "
            "Use run_orchestrator_attack() with skip_first_round=False instead."
        )

    def run_attack(
        self,
        baseline_results: List[RobustnessResult],
        progress_callback = None,
        save_every: int = 10
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """
        Not applicable for orchestrator pipeline.

        The orchestrator pipeline uses an iterative orchestrator approach.
        Use run_orchestrator_attack() instead.

        Raises:
            NotImplementedError: Always raised with helpful message
        """
        raise NotImplementedError(
            "OrchestratorPipeline does not support run_attack(). "
            "Use run_orchestrator_attack() instead."
        )

    # ============================================================
    # Orchestrator Attack Evaluation
    # ============================================================

    def run_orchestrator_attack(
        self,
        test_cases: List[TestCase],
        progress_callback = None,
        save_every: int = 10,
        max_samples: Optional[int] = None,
        skip_first_round: bool = False
    ) -> Tuple[List[RobustnessResult], RobustnessSummary]:
        """
        Run orchestrator attack evaluation on test cases.

        This is the main entry point for orchestrator-based robustness testing.
        Similar to run_attack() but uses iterative orchestrator strategy.

        Args:
            test_cases: List of test cases to evaluate
            progress_callback: Optional callback(results) for periodic saving
            save_every: Save checkpoint every N cases (default: 10)
            max_samples: Maximum number of first-round correct samples to attack
            skip_first_round: If True, use baseline results instead of testing first round

        Returns:
            Tuple of (results, summary)

        Example:
            >>> results, summary = pipeline.run_orchestrator_attack(
            ...     test_cases,
            ...     max_samples=100,
            ...     skip_first_round=True
            ... )
        """
        from med_red_team.utils import evaluate_items_fail_fast

        # Setup evaluation
        self._setup_evaluation(skip_first_round=skip_first_round, max_samples=max_samples)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ORCHESTRATOR ATTACK EVALUATION")
            print(f"{'='*80}")
            print(f"Testing {len(test_cases)} cases")
            if skip_first_round:
                print(f"Mode: Attack (skipping first round, using baseline)")
            if max_samples is not None:
                print(f"Max samples: {max_samples}")

        # Do not pre-slice raw cases: max_samples counts baseline-correct attack targets.
        # _should_stop_evaluation enforces the eligible-case limit during iteration.

        # Use evaluate_items_fail_fast
        results = evaluate_items_fail_fast(
            items=test_cases,
            evaluate_fn=lambda tc, idx: self._process_single_test_case(
                tc, idx, skip_first_round=skip_first_round
            ),
            description="Orchestrator Attack",
            progress_callback=progress_callback,
            save_every=save_every,
            verbose=self.verbose,
            stop_fn=lambda idx, current_results: self._should_stop_evaluation(
                idx,
                current_results,
                max_samples=max_samples,
            ),
        )

        # Create summary
        summary = self._create_summary(results)

        # Print summary
        if self.verbose:
            self._print_orchestrator_summary(summary)

        return results, summary

    def _create_summary(
        self,
        results: List[RobustnessResult]
    ) -> RobustnessSummary:
        """
        Create evaluation summary with orchestrator-specific metrics.

        Args:
            results: List of evaluation results

        Returns:
            RobustnessSummary with orchestrator metrics
        """
        total = len(results)
        fooled = sum(1 for result in results if result.status == "fooled")
        avg_iterations = (
            sum(result.metadata.get("iterations", 0) for result in results) / total
            if total else 0.0
        )
        avg_fooled_iterations = (
            sum(
                result.metadata.get("fooled_at_iteration", 0)
                for result in results
                if result.status == "fooled"
            ) / fooled
            if fooled else 0.0
        )
        return summarize_robustness_results(
            results,
            evaluation_type="robustness_orchestrator",
            attacks_applied=list(dict.fromkeys(
                attack
                for result in results
                for attack in (result.attacks_applied or [])
            )),
            metadata={
                "orchestrator_model": self.orchestrator_model_id,
                "tools_model": self.tools_model_id,
                "max_iterations": self.max_iterations,
                "max_planner_attempts": self.max_planner_attempts,
                "tool_policy_version": POLICY_VERSION,
                "avg_iterations_per_sample": avg_iterations,
                "avg_iterations_to_fool": avg_fooled_iterations,
            },
        )

    def _print_orchestrator_summary(self, summary: RobustnessSummary):
        """Print orchestrator attack summary statistics."""
        print(f"\n{'='*80}")
        print("ORCHESTRATOR ATTACK SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal cases attacked: {summary.total_samples}")
        print(f"Baseline correct: {summary.first_round_correct} ({summary.first_round_accuracy:.1%})")

        fooled = summary.fooled
        print(f"Failed manipulations: {summary.failed_manipulations}")
        print(f"Valid target-tested attacks: {summary.valid_target_tested}")
        print(f"Fooled: {fooled}")
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

        avg_iterations = summary.metadata.get("avg_iterations_per_sample", 0)
        avg_fooled_iterations = summary.metadata.get("avg_iterations_to_fool", 0)
        print(f"\nAvg iterations per sample: {avg_iterations:.2f}")
        if fooled > 0:
            print(f"Avg iterations to fool: {avg_fooled_iterations:.2f}")

        print(f"\nOrchestrator model: {summary.metadata.get('orchestrator_model', 'unknown')}")
        print(f"Tools model: {summary.metadata.get('tools_model', 'unknown')}")

    def save_results(
        self,
        results: List[RobustnessResult],
        summary: RobustnessSummary,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save evaluation results to JSON file.

        Standardized save interface matching pipeline.py.

        Args:
            results: List of evaluation results
            summary: Evaluation summary
            output_path: Path to save results
            metadata: Optional metadata dict to include
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
