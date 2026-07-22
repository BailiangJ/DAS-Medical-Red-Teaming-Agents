"""
Testee concrete implementation.

This module contains the concrete Testee class that wraps an LLM.
"""

from typing import Optional

from med_red_team.models import GenerationConfig, ModelResponse
from med_red_team.model_pool import ModelPool
from med_red_team.actors import Testee as TesteeBase


class Testee(TesteeBase):
    """
    Concrete implementation of Testee.
    
    Wraps an LLM to act as the model being tested.
    
    The Testee:
    - Receives questions (original or manipulated)
    - Generates answers
    - Does NOT know it's being attacked
    
    Usage:
        pool = ModelPool()
        testee = Testee(model_id="o3-mini", model_pool=pool)
        response = testee.answer("What is hypertension?")
    """
    
    def __init__(
        self,
        model_id: str,
        model_pool: ModelPool,
        system_prompt: str,
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize Testee.
        
        Args:
            model_id: Model to use (e.g., "gpt-4o", "qwq")
            model_pool: Shared model pool for efficiency
            system_prompt: Default instructions for testee
            config: Default generation config
        """
        super().__init__(model_id, model_pool, system_prompt, config)
    
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
            system_prompt: Override default system prompt
            config: Override default generation config
            
        Returns:
            ModelResponse with raw_text, reasoning, final_answer
        """
        return self.model.generate(
            user_prompt=question,
            system_prompt=system_prompt or self.system_prompt,
            config=config or self.config
        )
    
    def __repr__(self) -> str:
        return f"Testee(model_id='{self.model_id}')"