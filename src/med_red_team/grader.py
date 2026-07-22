"""
Grader concrete implementations.

This module contains SimpleGrader and RubricGrader.
"""

from typing import Dict, Any, Optional

from med_red_team.models import GenerationConfig
from med_red_team.actors import Grader
from med_red_team.data import TestCase, GradingResult
from med_red_team.model_pool import ModelPool
from med_red_team.models.utils.response_utils import (
    extract_multiple_choice_letters,
    extract_answer_with_quality_check,
    format_answer_label_set,
    parse_answer_label_set,
)


# ==============================================================================
# Simple Grader (for MedQA)
# ==============================================================================

class SimpleGrader(Grader):
    """
    Simple grader for multiple-choice questions.

    This is a rule-based grader that does not require an LLM.

    Grading logic:
    - Extracts letters from testee's answer (normalized and sorted)
    - Compares with correct answer (normalized in TestCase.__post_init__)
    - Returns binary correct/incorrect

    Supports both single and multiple correct answers:
    - Single: "A"
    - Multiple: "A,B,C" (order-independent, normalized)

    The comparison is robust to:
    - Different orderings: "B,A,C" vs "A,B,C" → both treated as "A,B,C"
    - Extra spaces: "A, B, C" vs "A,B,C" → both treated as "A,B,C"
    - Duplicates: "A,A,B" → treated as "A,B"

    Usage:
        grader = SimpleGrader()
        result = grader.grade(test_case, answer="The answer is B", context={})
    """
    def __init__(self):
        """Initialize simple grader (rule-based, no LLM required)."""
        super().__init__(
            name="choice_grader",
            model_id=None,
            model_pool=None,
            system_prompt=None,
            config=None
        )

    def grade(
        self,
        test_case: TestCase,
        answer: str,
        context: Dict[str, Any]
    ) -> GradingResult:
        """
        Grade a multiple-choice answer.

        Comparison is order-independent and robust:
        - TestCase.correct_answer is normalized in TestCase.__post_init__
        - Extracted answer is normalized by extract_answer_with_quality_check
        - Both are in format: "A" or "A,B,C" (sorted, no spaces)

        Also performs quality checking to detect responses with excessive extra text
        that may indicate the model didn't follow instructions properly.

        Args:
            test_case: Test case with correct_answer (already normalized)
            answer: Testee's answer (may contain reasoning, will be normalized)
            context: Additional context. Can contain:
                - max_extra_words: Override default threshold for extra words (default: 5)

        Returns:
            GradingResult with is_correct, reasoning, and metadata including:
            - extracted_answer: The normalized answer
            - answer_format_clean: Whether the response format is clean
            - format_warning: Warning message if format is not clean
        """
        # Get max_extra_words threshold from context or use default
        max_extra_words = context.get('max_extra_words', 5)

        # Extract and normalize letters from answer with quality checking
        valid_labels = test_case.options.keys()
        extracted, is_clean, warning_msg = extract_answer_with_quality_check(
            answer,
            max_extra_words=max_extra_words,
            valid_labels=valid_labels
        )

        extracted_labels = parse_answer_label_set(extracted, valid_labels)
        correct_labels = parse_answer_label_set(test_case.correct_answer, valid_labels)
        correct_answer = format_answer_label_set(correct_labels, valid_labels)
        is_correct = extracted_labels == correct_labels

        # Build reasoning
        reasoning = f"Expected '{correct_answer}', got '{extracted}'"
        if not is_clean:
            reasoning += f" [FORMAT WARNING: {warning_msg}]"

        # Build result with quality metadata
        return GradingResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            reasoning=reasoning,
            metadata={
                "extracted_answer": extracted,
                "correct_answer": correct_answer,
                "answer_format_clean": is_clean,
                "format_warning": warning_msg if not is_clean else None
            }
        )
    
    def __repr__(self) -> str:
        return "SimpleGrader(type='multiple_choice')"


