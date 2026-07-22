"""
Data structures for the red-teaming framework.

This module contains ONLY dataclasses.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

from med_red_team.models import ModelResponse


# ==============================================================================
# Core Data Structures
# ==============================================================================

@dataclass
class TestCase:
    """
    A single test case for red-teaming evaluation.

    Attributes:
        id: Unique identifier
        question: Original question text
        options: Multiple choice options (for MedQA) or None (for HealthBench)
        correct_answer: Ground truth answer (letter for MedQA, None for HealthBench)
        ideal_answer: Reference answer (None for MedQA, str for HealthBench)
        task_type: "multiple_choice" or "open_ended"
        metadata: Additional context (rubric, etc.)
    """
    id: str
    question: str
    options: Optional[Dict[str, str]] = None
    correct_answer: Optional[str] = None
    ideal_answer: Optional[str] = None
    task_type: str = "multiple_choice"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate test case consistency after initialization.

        Raises:
            ValueError: If validation fails
        """
        # Validate task_type
        valid_task_types = ["multiple_choice", "open_ended"]
        if self.task_type not in valid_task_types:
            raise ValueError(
                f"Invalid task_type: '{self.task_type}'. "
                f"Must be one of {valid_task_types}"
            )

        # Validate multiple choice requirements
        if self.task_type == "multiple_choice":
            if not self.options:
                raise ValueError(
                    "Multiple choice tasks must have options. "
                    f"TestCase {self.id} has options=None"
                )

            if not self.correct_answer:
                raise ValueError(
                    "Multiple choice tasks must have correct_answer. "
                    f"TestCase {self.id} has correct_answer=None"
                )

            invalid_option_labels = [
                label for label in self.options
                if str(label) != str(label).strip().upper()
            ]
            if invalid_option_labels:
                raise ValueError(
                    f"Option labels must be normalized uppercase labels: "
                    f"{invalid_option_labels}"
                )

            # Normalize correct_answer for robust comparison
            # This handles: "A", "A,B,C", "C,B,A", "A, B, C", etc.
            from med_red_team.models.utils.response_utils import normalize_answer, parse_answer_label_set
            raw_answer_labels = parse_answer_label_set(
                self.correct_answer,
                valid_labels="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            )
            invalid_labels = raw_answer_labels - {
                str(label).strip().upper() for label in self.options
            }
            if invalid_labels:
                raise ValueError(
                    f"Correct answer contains invalid labels: {sorted(invalid_labels)}. "
                    f"TestCase {self.id} has options: {list(self.options.keys())}"
                )

            self.correct_answer = normalize_answer(
                self.correct_answer,
                valid_labels=self.options.keys()
            )
            if not self.correct_answer:
                raise ValueError(
                    f"Correct answer contains no valid option labels. "
                    f"TestCase {self.id} has options: {list(self.options.keys())}"
                )

        # Validate open-ended requirements
        if self.task_type == "open_ended":
            if self.options is not None:
                raise ValueError(
                    "Open-ended tasks should not have options. "
                    f"TestCase {self.id} has options={self.options}"
                )

            if not self.ideal_answer:
                raise ValueError(
                    "Open-ended tasks should have ideal_answer. "
                    f"TestCase {self.id} has ideal_answer=None"
                )

        # Validate question is not empty
        if not self.question or not self.question.strip():
            raise ValueError(
                f"TestCase {self.id} has empty or whitespace-only question"
            )

    def build_user_prompt(self) -> str:
        """
        Build complete user prompt for the testee.

        For multiple choice questions, includes both question and options.
        For open-ended questions, returns just the question.

        Returns:
            Formatted prompt string ready to send to model

        Example:
            >>> # Multiple choice
            >>> test_case = TestCase(
            ...     id="medqa_001",
            ...     question="What is the mechanism of action?",
            ...     options={'A': 'Inhibition', 'B': 'Activation', 'C': 'Blockade'},
            ...     correct_answer='A',
            ...     task_type="multiple_choice"
            ... )
            >>> prompt = test_case.build_user_prompt()
            >>> print(prompt)
            What is the mechanism of action? Options: (A) Inhibition (B) Activation (C) Blockade

            >>> # Open-ended
            >>> test_case = TestCase(
            ...     id="hb_001",
            ...     question="Explain the diagnosis.",
            ...     task_type="open_ended",
            ...     ideal_answer="..."
            ... )
            >>> prompt = test_case.build_user_prompt()
            >>> print(prompt)
            Explain the diagnosis.
        """
        if self.task_type == "multiple_choice":
            # Format MCQ with options (similar to create_question in utils_general.py)
            prompt = self.question + " Options: "
            options = []
            for key, value in self.options.items():
                options.append(f"({key}) {value}")
            prompt += " ".join(options)
            return prompt
        else:
            # For open-ended, just return the question
            return self.question


@dataclass
class GradingResult:
    """
    Result of grading a response.
    
    Attributes:
        is_correct: Binary correctness (for multiple choice)
        score: Numeric score (0-10 for rubric-based grading)
        reasoning: Explanation of the grade
        metadata: Additional grading info
    """
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackAttempt:
    """
    A single attack attempt (baseline or adversarial).
    
    Attributes:
        attempt_type: "baseline" or "adversarial"
        question: The question asked (original or manipulated)
        response: Testee's full response
        extracted_answer: Parsed answer (letter or text)
        grading: Result of grading
        attack_strategy: Name of attack used (None for baseline)
    """
    attempt_type: str  # "baseline" or "adversarial"
    question: str
    response: ModelResponse
    extracted_answer: str
    grading: GradingResult
    attack_strategy: Optional[str] = None


@dataclass
class TestResult:
    """
    Complete result for a single test case.
    
    Attributes:
        test_case: Original test case
        baseline_attempt: Testee's response to original question
        adversarial_attempts: List of attack attempts
        attack_succeeded: Whether any attack succeeded
        final_accuracy: Final correctness/score after attacks
    """
    test_case: TestCase
    baseline_attempt: AttackAttempt
    adversarial_attempts: List[AttackAttempt]
    attack_succeeded: bool
    final_accuracy: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class EvaluationResult:
    """
    Abstract base class for evaluation results.
    
    All evaluation types must include these core fields.
    Subclasses add task-specific fields.
    """
    sample_number: int
    test_case_id: str
    skipped: bool = False
    skip_reason: Optional[str] = None
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        pass

@dataclass
class EvaluationSummary:
    """
    Abstract base class for evaluation summary.
    
    All evaluation types must provide these core metrics.
    Subclasses add task-specific metrics.
    """
    total_samples: int
    skipped_samples: int
    evaluation_type: str  # "bias", "robustness", "privacy", "hallucination"
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary for JSON serialization."""
        pass
    
    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        """Compute task-specific metrics."""
        pass