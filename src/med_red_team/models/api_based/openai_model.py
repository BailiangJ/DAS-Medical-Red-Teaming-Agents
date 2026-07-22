"""
OpenAI model implementations (GPT-4o, O3, etc.).

This module provides concrete implementations for OpenAI's API-based models.
"""

from typing import Dict, Any
from openai import OpenAI

from med_red_team.models.api_based.base import APIBasedModel
from med_red_team.models.interfaces import GenerationConfig, ModelGenerationError


class GPTModel(APIBasedModel):
    """
    OpenAI GPT models (gpt-4o, gpt-4.1, etc.).
    
    Supports:
    - gpt-4o
    - gpt-4.1
    - gpt-4.1-mini
    """
    
    def __init__(self, model_id: str):
        """
        Initialize GPT model.
        
        Args:
            model_id: Model identifier (e.g., "gpt-4o")
        """
        # Validate API key exists (optional, but good practice)
        super().__init__(model_id, api_key_env_var="OPENAI_API_KEY")
    
    def _initialize_client(self) -> OpenAI:
        """
        Initialize OpenAI client.
        
        The client automatically reads OPENAI_API_KEY from environment.
        """
        return OpenAI()
    
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Call OpenAI API for GPT models.
        
        Uses the new responses.create() API format.
        """
        # Format messages
        messages = self._format_messages(user_prompt, system_prompt)
        
        try:
            # Call API with temperature support
            response = self.client.responses.create(
                model=self.model_id,
                input=messages,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                # top_p=config.top_p,
            )

            # Extract text and metadata
            return {
                "text": response.output_text.strip(),
                "metadata": {
                    # Note: The new API may have different metadata fields
                    # Adjust based on actual response structure
                    # response doesn't have 'finish_reason'
                    # "finish_reason": getattr(response, "finish_reason", None),
                    "status": getattr(response, "status", None)
                }
            }
            
        except Exception as e:
            raise ModelGenerationError(
                f"OpenAI API call failed for {self.model_id}: {str(e)}"
            ) from e


class OGPTModel(APIBasedModel):
    """
    OpenAI O-series models (o3, o3-mini, o4-mini).
    
    These models do NOT support temperature parameter.
    """
    
    def __init__(self, model_id: str):
        """
        Initialize O-series model.
        
        Args:
            model_id: Model identifier (e.g., "o3", "o3-mini")
        """
        super().__init__(model_id, api_key_env_var="OPENAI_API_KEY")
    
    def _initialize_client(self) -> OpenAI:
        """Initialize OpenAI client."""
        return OpenAI()
    
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Call OpenAI API for O-series models.
        
        Note: Temperature is NOT supported for these models.

        Response object: https://platform.openai.com/docs/api-reference/responses/object
        """
        # Format messages
        messages = self._format_messages(user_prompt, system_prompt)
        
        try:
            # Call API WITHOUT temperature (not supported)
            response = self.client.responses.create(
                model=self.model_id,
                input=messages,
                max_output_tokens=config.max_tokens,
                reasoning=dict(effort=config.reasoning_effort),
            )
            
            return {
                "text": response.output_text.strip(),
                "metadata": {
                    "finish_reason": getattr(getattr(response, "incomplete_details", None), "reason", None),
                    "reasoning_summary": getattr(getattr(response, "reasoning", None), "summary", None),
                    "status": getattr(response, "status", None),
                    "note": "Temperature not supported for O-series models"
                }
            }
            
        except Exception as e:
            raise ModelGenerationError(
                f"OpenAI API call failed for {self.model_id}: {str(e)}"
            ) from e