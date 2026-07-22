
"""
Models module - Unified interface for LLM interactions.

This module provides:
- ModelFactory for creating model instances
- Core abstractions (LLMInterface, ModelResponse, Configs)
- Automatic CoT parsing
- Support for API-based and open-source models

Usage:
    from med_red_team.models import ModelFactory, GenerationConfig
    
    model = ModelFactory.create("gpt-4o")
    response = model.generate(
        user_prompt="Question here",
        system_prompt="You are an expert",
        config=GenerationConfig(temperature=0.7)
    )
    
    print(response.final_answer)
"""

from med_red_team.models.interfaces import (
    LLMInterface,
    ModelResponse,
    GenerationConfig,
    InfrastructureConfig,
    ModelExecutionError,
    ModelGenerationError,
    APIKeyMissingError,
    ModelNotFoundError,
    MalformedResponseError,
)
from med_red_team.models.factory import ModelFactory
from med_red_team.models.registry import MODEL_CONFIGS, list_api_models, list_opensource_models


# Initialize the factory with the registry
ModelFactory.register_models(MODEL_CONFIGS)


# ==============================================================================
# Public API
# ==============================================================================

__all__ = [
    # Factory
    "ModelFactory",
    
    # Core Abstractions
    "LLMInterface",
    "ModelResponse",
    
    # Configs
    "GenerationConfig",
    "InfrastructureConfig",
    
    # Exceptions
    "ModelExecutionError",
    "ModelGenerationError",
    "APIKeyMissingError",
    "ModelNotFoundError",
    "MalformedResponseError",
    
    # Utilities
    "list_api_models",
    "list_opensource_models",
]