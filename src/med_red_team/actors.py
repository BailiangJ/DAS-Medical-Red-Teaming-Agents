"""
Actor base classes (abstractions only).

This module defines the interfaces that all actors must implement:
- Testee: The model being tested
- Grader: Evaluates responses
- AttackStrategy: Modifies test cases adversarially
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from med_red_team.models import ModelResponse, GenerationConfig
from med_red_team.data import TestCase, GradingResult
from med_red_team.model_pool import ModelPool


# ==============================================================================
# Testee Base Class
# ==============================================================================

class Testee(ABC):
    """
    Abstract base class for testees.
    
    A Testee is the model being evaluated/attacked.
    """
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        system_prompt: str,
        config: Optional[GenerationConfig],
    ):
        self.model_id = model_id
        self.model = model_pool.get_model(model_id)

        self.system_prompt = system_prompt
        self.config = config or GenerationConfig(
            temperature=0.0,  # Deterministic for consistency
            max_tokens=2048
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert testee configuration to dictionary for logging/serialization.

        Returns:
            Dictionary with testee configuration
        """
        testee_dict = {
            "type": self.__class__.__name__,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
        }

        # Add config if available
        if self.config:
            if hasattr(self.config, '__dataclass_fields__'):
                from dataclasses import asdict
                testee_dict["config"] = asdict(self.config)
            elif hasattr(self.config, 'to_dict'):
                testee_dict["config"] = self.config.to_dict()

        return testee_dict

    @abstractmethod
    def answer(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> ModelResponse:
        """
        Answer a question.
        
        Args:
            question: The question to answer
            system_prompt: Optional system instructions
            config: Optional generation configuration
            
        Returns:
            ModelResponse with reasoning and final_answer
        """
        pass


# ==============================================================================
# Grader Base Class
# ==============================================================================

class Grader(ABC):
    """
    Abstract base class for graders.

    Graders evaluate testee responses and return scores/correctness.
    They can be:
    - Rule-based (deterministic, no LLM)
    - LLM-based (evaluate using an LLM with rubrics or criteria)

    Attributes:
        name: Human-readable name for this grader
        model_id: Optional model ID for LLM-based graders
        model: Optional model instance (None for rule-based graders)
        system_prompt: Optional system instructions for LLM-based graders
        config: Generation configuration (defaults to temperature=0.0 for consistency)
    """
    def __init__(
        self,
        name: str,
        model_id: Optional[str] = None,
        model_pool: Optional[ModelPool] = None,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize grader.

        Args:
            name: Human-readable name for this grader
            model_id: Optional model ID for LLM-based graders (required for LLM graders)
            model_pool: Optional model pool (required for LLM graders)
            system_prompt: Optional system instructions for LLM-based graders
            config: Optional generation configuration
        """
        self.name = name
        self.model_id = model_id
        if model_pool is None or model_id is None:
            self.model = None
        else:
            self.model = model_pool.get_model(model_id)

        self.system_prompt = system_prompt
        self.config = config or GenerationConfig(
            temperature=0.0,  # Deterministic for consistency
            max_tokens=2048
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert grader configuration to dictionary for logging/serialization.

        Returns:
            Dictionary with grader configuration
        """
        grader_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
        }

        # Add config if available
        if self.config:
            if hasattr(self.config, '__dataclass_fields__'):
                from dataclasses import asdict
                grader_dict["config"] = asdict(self.config)
            elif hasattr(self.config, 'to_dict'):
                grader_dict["config"] = self.config.to_dict()

        return grader_dict

    @abstractmethod
    def grade(
        self,
        test_case: TestCase,
        answer: str,
        context: Dict[str, Any]
    ) -> GradingResult:
        """
        Grade a testee's answer.
        
        Args:
            test_case: Original test case (has correct answer)
            answer: Testee's answer (extracted string)
            context: Additional context
            
        Returns:
            GradingResult with score and reasoning
        """
        pass


# ==============================================================================
# AttackStrategy Base Class
# ==============================================================================

class AttackStrategy(ABC):
    """
    Abstract base class for attack strategies.

    Attack strategies modify questions to try to make the testee fail.
    They can be:
    - Rule-based (deterministic, no LLM)
    - LLM-based (generate modifications using an LLM)

    Attributes:
        name: Human-readable name for this strategy
        model_id: Optional model ID for LLM-based strategies
        model: Optional model instance (None for rule-based strategies)
        system_prompt: Optional system instructions for LLM-based strategies
        config: Generation configuration (defaults to temperature=0.0 for consistency)
    """

    def __init__(
        self,
        name: str,
        model_id: Optional[str] = None,
        model_pool: Optional[ModelPool] = None,
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize attack strategy.

        Args:
            name: Human-readable name for this strategy
            model_id: Optional model ID for LLM-based strategies (required for LLM strategies)
            model_pool: Optional model pool (required for LLM strategies)
            system_prompt: Optional system instructions for LLM-based strategies
            config: Optional generation configuration
        """
        self.name = name
        self.model_id = model_id
        if model_pool is None or model_id is None:
            self.model = None
        else:
            self.model = model_pool.get_model(model_id)

        self.system_prompt = system_prompt
        self.config = config or GenerationConfig(
            temperature=0.0,  # Deterministic for consistency
            max_tokens=2048
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert attack strategy configuration to dictionary for logging/serialization.

        Returns:
            Dictionary with strategy configuration
        """
        strategy_dict = {
            "name": self.name,
            "type": self.__class__.__name__,
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
        }

        # Add config if available
        if self.config:
            if hasattr(self.config, '__dataclass_fields__'):
                from dataclasses import asdict
                strategy_dict["config"] = asdict(self.config)
            elif hasattr(self.config, 'to_dict'):
                strategy_dict["config"] = self.config.to_dict()

        return strategy_dict

    @abstractmethod
    def apply(self, test_case: TestCase, context: Dict[str, Any]) -> TestCase:
        """
        Apply the attack to a test case.
        
        Args:
            test_case: Original test case
            context: Additional context (previous attempts, feedback, etc.)
            
        Returns:
            Modified test case (adversarial version)
            
        Note:
            - Should NOT modify the original test_case
            - Should return a new TestCase with manipulated content
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"