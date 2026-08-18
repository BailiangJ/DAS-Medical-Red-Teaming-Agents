"""
Attack strategies for HealthBench robustness testing.

This module implements attack strategies that modify conversations to test
model robustness on the HealthBench dataset.
"""

import copy
import json
from typing import Dict, Any, Optional, List

from med_red_team.model_pool import ModelPool
from med_red_team.utils import normalize_bool, parse_json_response
from med_red_team.models import GenerationConfig

from med_red_team.healthbench.data import AttackResult, RubricItem
from med_red_team.healthbench.prompts import (
    ADJUST_MEASUREMENT_SYSTEM_PROMPT,
    build_adjust_measurement_user_prompt,
    DISTRACTION_SYSTEM_PROMPT,
    build_distraction_user_prompt,
    COGNITIVE_BIAS_SYSTEM_PROMPT,
    build_cognitive_bias_user_prompt,
)
from med_red_team.healthbench.utils import (
    add_distraction_to_conversation,
    parse_conversation_from_llm,
    validate_selected_rubric_index,
)


# ==============================================================================
# HealthBench attack strategies
# ==============================================================================


class _HealthBenchAttack:
    """Shared state for the local conversation-attack protocol."""

    def __init__(
        self,
        name: str,
        model_id: str,
        model_pool: ModelPool,
        system_prompt: str,
        config: GenerationConfig,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.model = model_pool.get_model(model_id)
        self.system_prompt = system_prompt
        self.config = config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "config": self.config.to_dict(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


def _parse_conversation_field(result_data: Dict[str, Any], field_name: str):
    """Parse a required string conversation field from attacker JSON."""
    value = result_data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    conversation = parse_conversation_from_llm(value)
    if not conversation:
        raise ValueError(f"{field_name} did not contain a parseable conversation")
    return conversation


def _maybe_parse_conversation_field(
    result_data: Dict[str, Any],
    field_name: str,
):
    """Parse an optional attacker conversation field when it is present."""
    if field_name not in result_data or result_data.get(field_name) is None:
        return None
    return _parse_conversation_field(result_data, field_name)


class AdjustImpossibleMeasurementStrategy(_HealthBenchAttack):
    """
    LLM-based strategy that identifies and mutates numeric measurements
    to impossible values to test clinical judgment.

    It reuses the shared attack-strategy runtime helpers, but HealthBench code
    should treat this as a conversation-specific class with
    ``apply_to_conversation()`` rather than as a generic interchangeable attack
    contract.

    Example:
        >>> strategy = AdjustImpossibleMeasurementStrategy(
        ...     model_id="gpt-4.1-2025-04-14",
        ...     model_pool=model_pool
        ... )
        >>> attack_result = strategy.apply_to_conversation(
        ...     conversation=[...],
        ...     test_case_id="case_001"
        ... )
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(
            name="impossible_measurement",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=ADJUST_MEASUREMENT_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.7, max_tokens=2048)
        )

    def apply_to_conversation(
        self,
        conversation: List[Dict[str, str]],
        test_case_id: str
    ) -> AttackResult:
        """
        Apply impossible measurement attack to a conversation.

        Returns AttackResult with:
        - applicable: bool
        - modified_conversation: List[Dict] or None
        - attack_metadata: {changed_measurement, reasoning}
        """
        user_prompt = build_adjust_measurement_user_prompt(conversation)

        # Call model API - let API errors propagate to trigger fail-fast
        response = self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            config=self.config
        )

        print("LLM Response:\n", response.final_answer)

        # Parse JSON response - only catch parsing errors
        try:
            result_data = parse_json_response(response.final_answer)
        except (json.JSONDecodeError, ValueError) as e:
            # JSON parsing failed - mark as not applicable
            print(f"Warning: Failed to parse LLM response: {e}")
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={},
                modified_conversation=None,
                error=f"JSON parsing failed: {str(e)}"
            )

        try:
            applicable = normalize_bool(result_data.get('applicable', False))
        except ValueError as e:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={'applicable_raw': result_data.get('applicable')},
                modified_conversation=None,
                error=str(e)
            )

        modified_conv = None
        parse_error = None
        try:
            modified_conv = _maybe_parse_conversation_field(
                result_data,
                "updated_conversation",
            )
        except ValueError as e:
            parse_error = str(e)

        return AttackResult(
            test_case_id=test_case_id,
            attack_strategy=self.name,
            applicable=applicable,
            original_conversation=conversation,
            attack_metadata={
                'changed_measurement': result_data.get('changed_measurement', ''),
                'reasoning': result_data.get('reasoning', '')
            },
            modified_conversation=modified_conv,
            error=parse_error
        )


class CognitiveBiasStrategy(_HealthBenchAttack):
    """
    LLM-based strategy for generating cognitive bias attacks targeting attackable rubrics.

    Workflow:
    - Single LLM call per case
    - Targets attackable rubrics (positive MET + negative NOT MET)
    - Introduces cognitive biases (Authority, FalseConfirmation, AvailabilityConvenience, EmotionalAffect)
    - Returns modified conversation with bias mutation

    Example:
        >>> strategy = CognitiveBiasStrategy(
        ...     model_id="claude-sonnet-4-20250514",
        ...     model_pool=model_pool
        ... )
        >>> attack_result = strategy.apply_to_conversation(
        ...     conversation=[...],
        ...     attackable_rubrics_with_context=[...],
        ...     test_case_id="case_001"
        ... )
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(
            name="cognitive_bias",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=COGNITIVE_BIAS_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.7, max_tokens=2048)
        )

    def apply_to_conversation(
        self,
        conversation: List[Dict[str, str]],
        attackable_rubrics_with_context: List[Dict[str, Any]],
        test_case_id: str
    ) -> AttackResult:
        """
        Apply cognitive bias attack targeting attackable rubrics.

        Args:
            conversation: Original conversation
            attackable_rubrics_with_context: List of dicts with:
                - rubric: Dict with criterion, points, tags
                - grader_explanation: Why this rubric was met/unmet
                - rubric_index: Original index in full rubrics array
            test_case_id: Test case identifier

        Returns:
            AttackResult with cognitive bias metadata
        """
        # Check if there are any attackable rubrics
        if not attackable_rubrics_with_context:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={},
                modified_conversation=None,
                error="No attackable rubrics to attack"
            )

        user_prompt = build_cognitive_bias_user_prompt(conversation, attackable_rubrics_with_context)

        # Call model API - let API errors propagate to trigger fail-fast
        response = self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            config=self.config
        )

        # Parse JSON response - only catch parsing errors
        try:
            result_data = parse_json_response(response.final_answer)
        except (json.JSONDecodeError, ValueError) as e:
            # JSON parsing failed - mark as not applicable
            print(f"Warning: Failed to parse LLM response: {e}")
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={},
                modified_conversation=None,
                error=f"JSON parsing failed: {str(e)}"
            )

        try:
            applicable = normalize_bool(result_data.get('applicable', False))
        except ValueError as e:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={'applicable_raw': result_data.get('applicable')},
                modified_conversation=None,
                error=str(e)
            )
        modified_conv = None
        parse_error = None
        try:
            modified_conv = _maybe_parse_conversation_field(
                result_data,
                "modified_conversation",
            )
        except ValueError as e:
            parse_error = str(e)

        # Compute both indices for clarity
        raw_selected_rubric_index = result_data.get('selected_rubric_index', -1)
        try:
            selected_rubric_index = validate_selected_rubric_index(
                raw_selected_rubric_index,
                len(attackable_rubrics_with_context),
            )
        except ValueError as e:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={
                    'selected_rubric_index': -1,
                    'selected_rubric_index_raw': raw_selected_rubric_index,
                },
                modified_conversation=None,
                error=str(e)
            )

        original_rubric_index = -1
        if selected_rubric_index >= 0:
            original_rubric_index = attackable_rubrics_with_context[selected_rubric_index].get('rubric_index', -1)

        return AttackResult(
            test_case_id=test_case_id,
            attack_strategy=self.name,
            applicable=applicable,
            original_conversation=conversation,
            attack_metadata={
                'selected_rubric_index': selected_rubric_index,  # Index into attackable rubrics array (for pipeline)
                'original_rubric_index': original_rubric_index,  # Index into original rubrics array (for HTML)
                'vulnerability_analysis': result_data.get('vulnerability_analysis', ''),
                'attackability_reason': result_data.get('attackability_reason', ''),
                'attack_strategies': result_data.get('attack_strategies', []),
                'modification_reason': result_data.get('modification_reason', ''),
                'rubric_validity_reason': result_data.get('rubric_validity_reason', '')
            },
            modified_conversation=modified_conv,
            error=parse_error
        )


class DistractionSentenceStrategy(_HealthBenchAttack):
    """
    LLM-based strategy for generating distraction sentences targeting attackable rubrics.

    Workflow:
    - Single LLM call per case
    - Targets attackable rubrics (positive MET + negative NOT MET)
    - Returns one targeted distraction sentence with modified conversation

    Example:
        >>> strategy = DistractionSentenceStrategy(
        ...     model_id="claude-sonnet-4-20250514",
        ...     model_pool=model_pool
        ... )
        >>> attack_result = strategy.apply_to_conversation(
        ...     conversation=[...],
        ...     attackable_rubrics_with_context=[...],
        ...     test_case_id="case_001"
        ... )
    """

    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        config: Optional[GenerationConfig] = None
    ):
        super().__init__(
            name="distraction",
            model_id=model_id,
            model_pool=model_pool,
            system_prompt=DISTRACTION_SYSTEM_PROMPT,
            config=config or GenerationConfig(temperature=0.7, max_tokens=2048)
        )

    def apply_to_conversation(
        self,
        conversation: List[Dict[str, str]],
        attackable_rubrics_with_context: List[Dict[str, Any]],
        test_case_id: str
    ) -> AttackResult:
        """
        Apply distraction attack targeting attackable rubrics.

        Args:
            conversation: Original conversation
            attackable_rubrics_with_context: List of dicts with:
                - rubric: Dict with criterion, points, tags
                - grader_explanation: Why this rubric was met/unmet
                - rubric_index: Original index in full rubrics array
            test_case_id: Test case identifier

        Returns:
            AttackResult with distraction metadata
        """
        # Check if there are any attackable rubrics
        if not attackable_rubrics_with_context:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={},
                modified_conversation=None,
                error="No attackable rubrics to attack"
            )

        user_prompt = build_distraction_user_prompt(conversation, attackable_rubrics_with_context)

        # Call model API - let API errors propagate to trigger fail-fast
        response = self.model.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            config=self.config
        )

        # Parse JSON response - only catch parsing errors
        try:
            result_data = parse_json_response(response.final_answer)
        except (json.JSONDecodeError, ValueError) as e:
            # JSON parsing failed - mark as not applicable
            print(f"Warning: Failed to parse LLM response: {e}")
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={},
                modified_conversation=None,
                error=f"JSON parsing failed: {str(e)}"
            )

        try:
            applicable = normalize_bool(result_data.get('applicable', False))
        except ValueError as e:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={'applicable_raw': result_data.get('applicable')},
                modified_conversation=None,
                error=str(e)
            )
        modified_conv = None
        parse_error = None
        try:
            modified_conv = _maybe_parse_conversation_field(
                result_data,
                "modified_conversation",
            )
        except ValueError as e:
            parse_error = str(e)

        # Compute both indices for clarity
        raw_selected_rubric_index = result_data.get('selected_rubric_index', -1)
        try:
            selected_rubric_index = validate_selected_rubric_index(
                raw_selected_rubric_index,
                len(attackable_rubrics_with_context),
            )
        except ValueError as e:
            return AttackResult(
                test_case_id=test_case_id,
                attack_strategy=self.name,
                applicable=False,
                original_conversation=conversation,
                attack_metadata={
                    'selected_rubric_index': -1,
                    'selected_rubric_index_raw': raw_selected_rubric_index,
                },
                modified_conversation=None,
                error=str(e)
            )

        original_rubric_index = -1
        if selected_rubric_index >= 0:
            original_rubric_index = attackable_rubrics_with_context[selected_rubric_index].get('rubric_index', -1)

        return AttackResult(
            test_case_id=test_case_id,
            attack_strategy=self.name,
            applicable=applicable,
            original_conversation=conversation,
            attack_metadata={
                'selected_rubric_index': selected_rubric_index,  # Index into attackable rubrics array (for pipeline)
                'original_rubric_index': original_rubric_index,  # Index into original rubrics array (for HTML)
                'root_cause_analysis': result_data.get('root_cause_analysis', ''),
                'trigger_keywords': result_data.get('trigger_keywords', []),
                'rubric_validity_reason': result_data.get('rubric_validity_reason', '')
            },
            modified_conversation=modified_conv,
            error=parse_error
        )
