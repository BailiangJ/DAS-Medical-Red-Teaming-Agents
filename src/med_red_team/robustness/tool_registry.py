"""
Tool Registry for Orchestrator Pipeline
========================================

Maps tool names to strategy instances and handles manual execution.

In hybrid mode, the orchestrator only plans which tools to use, and this
registry executes them manually in sequence, capturing detailed logs at each step.
"""

import time
from typing import Dict, Any, Tuple, Optional

from med_red_team.robustness.data import OrchestratorQuestionData, ToolExecutionLog
from med_red_team.robustness.tool_policy import (
    ORCHESTRATOR_TOOL_NAMES,
    TOOL_NAME_TO_STRATEGY_NAME,
    ToolPolicyDecision,
    check_tool_postconditions,
    check_tool_preconditions,
    compute_question_delta,
    normalize_tool_sequence,
)
from med_red_team.robustness.attacker import (
    GenerateDistractorOptionsStrategy,
    ReplaceCorrectAnswerStrategy,
    InvertQuestionAnswerStrategy,
    AddDistractionSentenceStrategy,
    AdjustImpossibleMeasurementStrategy,
    BiasManipulationStrategy,
)
from med_red_team.model_pool import ModelPool
from med_red_team.models import ModelExecutionError


class ToolRegistry:
    """
    Registry that maps tool names to attack strategy instances.

    Handles:
    - Strategy initialization with configs
    - Tool name to strategy name mapping
    - Manual tool execution with metadata separation
    - Detailed logging of each execution
    - Complement answer updates for tool chaining

    The registry maintains strategy instances across tool calls for efficiency.
    """

    # Backward-compatible mapping sourced from the shared runtime policy.
    TOOL_NAME_MAP = {
        tool_name: TOOL_NAME_TO_STRATEGY_NAME[tool_name]
        for tool_name in ORCHESTRATOR_TOOL_NAMES
    }

    def __init__(
        self,
        model_pool: ModelPool,
        tools_model_id: str,
        strategy_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize tool registry with strategy instances.

        Args:
            model_pool: ModelPool for LLM access
            tools_model_id: Default model ID for strategies
            strategy_configs: Per-strategy configuration overrides
        """
        self.model_pool = model_pool
        self.tools_model_id = tools_model_id
        self.strategy_configs = strategy_configs or {}

        # Initialize all strategies
        self.strategies = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize all attack strategy instances."""

        # Rule-based strategies (no model needed)
        self.strategies["replace_correct_answer_with_none"] = ReplaceCorrectAnswerStrategy()

        # LLM-based strategies
        self.strategies["add_distraction_sentence"] = self._create_distraction_strategy()
        self.strategies["generate_distractor_options"] = self._create_distractor_options_strategy()
        self.strategies["bias_manipulation"] = self._create_bias_strategy()
        self.strategies["invert_question_answer"] = self._create_invert_strategy()
        self.strategies["adjust_impossible_measurement"] = self._create_measurement_strategy()

    def _get_strategy_config(self, strategy_name: str, **defaults) -> Dict[str, Any]:
        """
        Get merged config for a strategy (defaults + overrides).

        Args:
            strategy_name: Name of the strategy
            **defaults: Default kwargs for the strategy

        Returns:
            Merged config dict
        """
        merged = dict(defaults)
        if strategy_name in self.strategy_configs:
            merged.update(self.strategy_configs[strategy_name])
        return merged

    def _create_distraction_strategy(self):
        """Create AddDistractionSentenceStrategy with config."""
        config = self._get_strategy_config(
            "add_distraction_sentence",
            model_id=self.tools_model_id,
            model_pool=self.model_pool
        )
        return AddDistractionSentenceStrategy(**config)

    def _create_distractor_options_strategy(self):
        """Create GenerateDistractorOptionsStrategy with config."""
        config = self._get_strategy_config(
            "generate_distractor_options",
            model_id=self.tools_model_id,
            model_pool=self.model_pool,
            num_distractors=4
        )
        return GenerateDistractorOptionsStrategy(**config)

    def _create_bias_strategy(self):
        """Create BiasManipulationStrategy with config."""
        config = self._get_strategy_config(
            "bias_manipulation",
            model_id=self.tools_model_id,
            model_pool=self.model_pool,
            n_bias_styles=2
        )
        return BiasManipulationStrategy(**config)

    def _create_invert_strategy(self):
        """Create InvertQuestionAnswerStrategy with config."""
        config = self._get_strategy_config(
            "invert_question_answer",
            model_id=self.tools_model_id,
            model_pool=self.model_pool
        )
        return InvertQuestionAnswerStrategy(**config)

    def _create_measurement_strategy(self):
        """Create AdjustImpossibleMeasurementStrategy with config."""
        config = self._get_strategy_config(
            "adjust_impossible_measurement",
            model_id=self.tools_model_id,
            model_pool=self.model_pool
        )
        return AdjustImpossibleMeasurementStrategy(**config)

    def validate_and_normalize_sequence(
        self,
        tool_sequence: list,
    ) -> ToolPolicyDecision:
        """Validate conflicts and normalize ordering through the shared policy."""
        return normalize_tool_sequence(tool_sequence)

    def reorder_tool_sequence(self, tool_sequence: list) -> list:
        """Compatibility wrapper returning a normalized sequence or raising."""
        decision = self.validate_and_normalize_sequence(tool_sequence)
        if not decision.valid:
            reasons = "; ".join(issue.message for issue in decision.issues)
            raise ValueError(reasons)
        return decision.normalized_sequence

    def execute_tool(
        self,
        tool_name: str,
        question_data: OrchestratorQuestionData,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[OrchestratorQuestionData, ToolExecutionLog]:
        """
        Execute a single tool and return modified question + execution log.

        This is the core method for hybrid mode execution. It:
        1. Converts to TestCase (core QA only, no metadata)
        2. Executes the strategy
        3. Extracts metadata separately
        4. Validates the resulting question state
        5. Returns modified question and detailed log

        Args:
            tool_name: Name of tool to execute (e.g., "add_distraction_sentence_tool")
            question_data: Current question state (clean, no metadata)
            context: Optional execution context

        Returns:
            (modified_question, execution_log)
        """
        start_time = time.time()
        context = context or {}

        # Store input state for logging
        input_state = {
            "question": question_data.question,
            "options": question_data.options.copy(),
            "answer": question_data.answer_idx
        }

        try:
            if tool_name not in self.TOOL_NAME_MAP:
                raise ValueError(f"Unknown tool: {tool_name}")

            precondition = check_tool_preconditions(tool_name, question_data)
            if not precondition.valid:
                execution_time = (time.time() - start_time) * 1000
                log = ToolExecutionLog(
                    tool_name=tool_name,
                    input_question=input_state["question"],
                    output_question=input_state["question"],
                    input_options=input_state["options"],
                    output_options=input_state["options"],
                    input_answer=input_state["answer"],
                    output_answer=input_state["answer"],
                    tool_metadata={
                        "reason": precondition.reason,
                        "execution_skipped": True,
                    },
                    execution_time_ms=execution_time,
                    success=False,
                    error_message=precondition.reason,
                    status=precondition.status,
                    failure_category=precondition.status,
                    delta=compute_question_delta(question_data, question_data),
                )
                return question_data, log

            strategy_name = self.TOOL_NAME_MAP[tool_name]
            strategy = self.strategies[strategy_name]
            test_case = question_data.to_test_case(test_case_id="temp_orchestrator")
            modified = strategy.apply(test_case, context)
            tool_metadata = modified.metadata.copy() if modified.metadata else {}
            modified_question = OrchestratorQuestionData.from_test_case(modified)

            manipulation_failed = tool_metadata.get("manipulation_failed", False)
            failure_category = tool_metadata.get("failure_category")
            error_message = tool_metadata.get("reason") if manipulation_failed else None
            status = failure_category or ("invalid_noop" if manipulation_failed else "success")

            if not manipulation_failed:
                postcondition = check_tool_postconditions(
                    tool_name,
                    question_data,
                    modified_question,
                )
                if not postcondition.valid:
                    manipulation_failed = True
                    status = postcondition.status
                    failure_category = postcondition.status
                    error_message = postcondition.reason
                    tool_metadata = {
                        **tool_metadata,
                        "manipulation_failed": True,
                        "failure_category": postcondition.status,
                        "reason": postcondition.reason,
                    }

            execution_time = (time.time() - start_time) * 1000
            log = ToolExecutionLog(
                tool_name=tool_name,
                input_question=input_state["question"],
                output_question=modified_question.question,
                input_options=input_state["options"],
                output_options=modified_question.options,
                input_answer=input_state["answer"],
                output_answer=modified_question.answer_idx,
                tool_metadata=tool_metadata,
                execution_time_ms=execution_time,
                success=not manipulation_failed,
                error_message=error_message,
                status=status,
                failure_category=failure_category,
                delta=compute_question_delta(question_data, modified_question),
            )

            return (
                modified_question if not manipulation_failed else question_data,
                log,
            )

        except ModelExecutionError:
            raise
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            log = ToolExecutionLog(
                tool_name=tool_name,
                input_question=input_state["question"],
                output_question=input_state["question"],
                input_options=input_state["options"],
                output_options=input_state["options"],
                input_answer=input_state["answer"],
                output_answer=input_state["answer"],
                tool_metadata={},
                execution_time_ms=execution_time,
                success=False,
                error_message=str(e),
                status="tool_error",
                failure_category="tool_error",
                delta=compute_question_delta(question_data, question_data),
            )
            return question_data, log
