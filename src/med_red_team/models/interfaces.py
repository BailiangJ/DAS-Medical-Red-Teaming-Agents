"""
Core abstractions for the model_execution module.

This module defines the contracts that all LLM implementations must follow.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime


# ==============================================================================
# Custom Exceptions
# ==============================================================================

class ModelExecutionError(Exception):
    """Base exception for model execution errors."""
    pass


class ModelGenerationError(ModelExecutionError):
    """Raised when the model API or vLLM fails during generation."""
    pass


class APIKeyMissingError(ModelExecutionError):
    """Raised when a required API key is not found."""
    pass


class ModelNotFoundError(ModelExecutionError):
    """Raised when a model_id is not in the registry."""
    pass


class MalformedResponseError(ModelExecutionError):
    """
    Raised when model output has malformed structure.

    Examples:
    - Unclosed thinking tags: "<think>..." without "</think>"
    - Invalid format that prevents parsing

    Attributes:
        message: Human-readable error description
        raw_text: The malformed raw output text
        error_type: Category of error (e.g., "unclosed_thinking_tag")
        details: Additional context about the error
    """
    def __init__(
        self,
        message: str,
        raw_text: str,
        error_type: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MalformedResponseError.

        Args:
            message: Human-readable error description
            raw_text: The malformed raw output
            error_type: Error category (e.g., "unclosed_thinking_tag")
            details: Optional dict with additional context
        """
        self.raw_text = raw_text
        self.error_type = error_type
        self.details = details or {}
        super().__init__(message)


# ==============================================================================
# Configuration Data Structures
# ==============================================================================

@dataclass
class GenerationConfig:
    """
    Unified hyperparameters for text generation.
    These can be overridden per-request.
    
    Usage:
        config = GenerationConfig(temperature=0.3, max_tokens=500)
        response = model.generate(..., config=config)
    """
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 2048
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    use_thinking: bool = False
    max_thinking_tokens: int = 2048
    reasoning_effort: str = "medium"

    # Future extensions:
    # output_grammar: Optional[str] = None
    # seed: Optional[int] = None
    # presence_penalty: float = 0.0
    # frequency_penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class InfrastructureConfig:
    """
    Infrastructure configuration for open-source models (vLLM).
    Set once during model initialization.

    Usage:
        infra = InfrastructureConfig(gpu_memory_utilization=0.95)
        model = ModelFactory.create("llama-3-70b", infra_config=infra)
    """
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 2
    max_model_len: int = 16384
    max_num_seqs: int = 2
    dtype: str = "bfloat16"
    enable_tf32: bool = True
    quantization: Optional[str] = None
    seed: int | None = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ==============================================================================
# Response Data Structure (Minimal + Extensible)
# ==============================================================================

@dataclass
class ModelResponse:
    """
    Structured response from any LLM.
    
    Design:
    - Preserves full raw output for debugging
    - Separates reasoning (CoT) from final answer
    - Extensible metadata for logging
    
    Usage:
        response = model.generate(...)
        print(response.raw_text)       # Full original output
        print(response.reasoning)      # CoT/thinking (if present)
        print(response.final_answer)   # Answer after reasoning
        print(response.metadata)       # Token usage, latency, etc.
    """
    # Core fields (always populated)
    raw_text: str
    model_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Parsed components (automatically extracted from raw_text)
    reasoning: str = ""          # Chain-of-thought/thinking part
    final_answer: str = ""       # Answer after reasoning
    
    # Extensible metadata (optional, for logging/debugging)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional: Store the config used for this generation
    generation_config: Optional[GenerationConfig] = None
    
    # Examples of what metadata could contain:
    # - "token_usage": {"prompt_tokens": 120, "completion_tokens": 80}
    # - "finish_reason": "stop" | "length" | "content_filter"
    # - "latency_ms": 1234.5
    # - "provider_request_id": "req_abc123"
    # - "model_version": "gpt-4o-2024-05-13"


# ==============================================================================
# Core Interface
# ==============================================================================

class LLMInterface(ABC):
    """
    The unified contract for all LLM interactions.
    
    Any implementation (API or local) must adhere to this interface.
    This allows Testee/Attacker/Grader to work with any model transparently.
    
    Design Pattern: Strategy Pattern
    - The interface defines the "what" (generate text)
    - Concrete classes define the "how" (API call vs vLLM)
    
    Usage:
        class MyModel(LLMInterface):
            def generate(self, user_prompt, system_prompt, config):
                # Implementation here
                return ModelResponse(...)
    """
    
    def __init__(self, model_id: str):
        """
        Initialize the model.
        
        Args:
            model_id: Unique identifier (e.g., "gpt-4o", "llama-3-70b")
        """
        self.model_id = model_id
    
    @abstractmethod
    def generate(
        self, 
        user_prompt: str, 
        system_prompt: str,
        config: GenerationConfig
    ) -> ModelResponse:
        """
        Generate text from the LLM.
        
        This is the core method that all implementations must provide.
        
        Args:
            user_prompt: The primary input (e.g., question, attack prompt)
            system_prompt: Instructions/persona (e.g., "You are a medical AI")
            config: Generation hyperparameters (temperature, max_tokens, etc.)
            
        Returns:
            ModelResponse with raw_text and optional metadata
            
        Raises:
            ModelGenerationError: If the underlying API/vLLM fails
            APIKeyMissingError: If required credentials are missing
            
        Implementation Guidelines:
        - Always populate raw_text and model_id in ModelResponse
        - Set generation_config in response for logging
        - Populate metadata with provider-specific info (tokens, latency, etc.)
        - Handle errors gracefully (don't let raw API errors bubble up)
        """
        pass
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"