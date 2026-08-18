"""
Base class for API-based LLM implementations.

This module provides shared functionality for all models that use HTTP APIs
(OpenAI, Anthropic, Google, DeepSeek, etc.).
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import time
import os

from med_red_team.models.interfaces import (
    LLMInterface,
    ModelResponse,
    GenerationConfig,
    ModelGenerationError,
    APIKeyMissingError
)


class APIBasedModel(LLMInterface):
    """
    Abstract base class for API-based models.
    
    Responsibilities:
    - Define the common pattern for API calls
    - Provide helper methods for error handling
    - Standardize metadata extraction
    
    Subclasses must implement:
    - _initialize_client(): Set up the API client
    - _call_api(): Make the actual API request
    
    Design Pattern: Template Method Pattern
    - generate() defines the algorithm structure
    - Subclasses fill in the specific steps
    """
    
    def __init__(self, model_id: str, api_key_env_var: Optional[str] = None):
        """
        Initialize API-based model.
        
        Args:
            model_id: The model identifier
            api_key_env_var: Environment variable name for API key (optional)
                           If provided, validates that the key exists
        """
        super().__init__(model_id)
        
        # Validate API key if specified
        if api_key_env_var:
            self._validate_api_key(api_key_env_var)
        
        # Initialize the client (provider-specific)
        self.client = self._initialize_client()
    
    def _validate_api_key(self, env_var: str) -> None:
        """
        Validate that an API key exists in environment.
        
        Args:
            env_var: Environment variable name
            
        Raises:
            APIKeyMissingError: If the key is not set
        """
        api_key = os.getenv(env_var)
        if not api_key:
            raise APIKeyMissingError(
                f"API key not found: {env_var}. "
                "Set this environment variable before starting the provider-backed run."
            )
    
    @abstractmethod
    def _initialize_client(self) -> Any:
        """
        Initialize the provider-specific API client.
        
        Returns:
            The initialized client object (OpenAI, Anthropic, etc.)
            
        Implementation Example:
            from openai import OpenAI
            return OpenAI()  # Reads key from environment
        """
        pass
    
    @abstractmethod
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Make the actual API call.
        
        Args:
            user_prompt: User message
            system_prompt: System instruction
            config: Generation parameters
            
        Returns:
            Dict containing:
                - "text": str (the generated text)
                - "metadata": Dict[str, Any] (optional provider-specific data)
                
        Raises:
            ModelGenerationError: If API call fails
            
        Implementation Example:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            return {
                "text": response.choices[0].message.content,
                "metadata": {
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens
                    },
                    "finish_reason": response.choices[0].finish_reason
                }
            }
        """
        pass
    
    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> ModelResponse:
        """
        Generate text via API call.
        
        This method implements the template:
        1. Record start time
        2. Call provider-specific API (_call_api)
        3. Parse CoT reasoning (auto)
        4. Build ModelResponse with metadata
        5. Handle errors gracefully
        
        Args:
            user_prompt: User input
            system_prompt: System instruction
            config: Generation config
            
        Returns:
            ModelResponse with raw text, reasoning, and final answer
            
        Raises:
            ModelGenerationError: If generation fails
        """
        start_time = time.time()
        
        try:
            # Call provider-specific API
            api_result = self._call_api(user_prompt, system_prompt, config)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Build metadata
            metadata = api_result.get("metadata", {})
            metadata["latency_ms"] = latency_ms
            
            # Parse CoT reasoning
            from med_red_team.models.utils.response_utils import parse_cot_response
            from med_red_team.models import MalformedResponseError

            raw_text = api_result["text"]

            # Try to parse, catching malformed response errors
            reasoning = ""
            final_answer = raw_text  # Fallback: use raw text as answer
            error_info = None

            try:
                reasoning, final_answer = parse_cot_response(raw_text, raise_on_malformed=True)
            except MalformedResponseError as e:
                # Store error info but don't crash - allow graceful handling
                reasoning = e.details.get("truncated_reasoning", "")
                final_answer = ""  # Empty since model never finished thinking
                error_info = {
                    "error_type": e.error_type,
                    "error_message": str(e),
                    "details": e.details
                }

            # Add error info to metadata if present
            if error_info:
                metadata["parse_error"] = error_info

            # Build response
            return ModelResponse(
                raw_text=raw_text,
                reasoning=reasoning,
                final_answer=final_answer,
                model_id=self.model_id,
                metadata=metadata,
                generation_config=config
            )
            
        except APIKeyMissingError:
            # Re-raise API key errors as-is
            raise
            
        except Exception as e:
            # Wrap all other errors in ModelGenerationError
            raise ModelGenerationError(
                f"Failed to generate response from {self.model_id}: {str(e)}"
            ) from e
    
    def _format_messages(
        self,
        user_prompt: str,
        system_prompt: str,
        image_url: Optional[str] = None
    ) -> list:
        """
        Helper: Format messages for OpenAI-style APIs.
        
        This is a convenience method that subclasses can use.
        
        Args:
            user_prompt: User message
            system_prompt: System message
            image_url: Optional image URL (for vision models)
            
        Returns:
            List of message dicts in OpenAI format
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        return messages