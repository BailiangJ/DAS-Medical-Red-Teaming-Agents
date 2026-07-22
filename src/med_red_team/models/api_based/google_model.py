"""
Google Gemini model implementations.
"""

from typing import Dict, Any
from google import genai
from google.genai import types

from med_red_team.models.api_based.base import APIBasedModel
from med_red_team.models.interfaces import GenerationConfig, ModelGenerationError


class GeminiModel(APIBasedModel):
    """
    Google Gemini models.
    
    Supports:
    - gemini-2.5-flash
    - gemini-2.5-pro
    """

    def __init__(self, model_id: str):
        """
        Initialize Gemini model.
        
        Args:
            model_id: Model identifier (e.g., "gemini-2.0-flash")
        """
        super().__init__(model_id, api_key_env_var="GEMINI_API_KEY")
    
    def _initialize_client(self):
        """
        Initialize Gemini client.
        
        Reads GEMINI_API_KEY from environment.
        """
        import os
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        return genai.Client(api_key=gemini_api_key)
    
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Call Google Gemini API.
        GenerationConfig: https://ai.google.dev/api/generate-content#generationconfig
        docs on Thinking: https://ai.google.dev/gemini-api/docs/thinking
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k if config.top_k >= 0 else None,
                    max_output_tokens=config.max_tokens,
                    stop_sequences=config.stop_sequences or None,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=config.use_thinking,
                        thinking_budget=config.max_thinking_tokens,
                    ),
                ),
                contents=user_prompt,
            )
            return {
                "text": response.text,
                "metadata": {
                    # Gemini may provide additional metadata
                    "finish_reason": response.candidates[0].finish_reason.name,
                }
            }
            
        except Exception as e:
            raise ModelGenerationError(
                f"Gemini API call failed for {self.model_id}: {str(e)}"
            ) from e