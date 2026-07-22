"""
DeepSeek model implementations.
"""

from typing import Dict, Any
from openai import OpenAI

from med_red_team.models.api_based.base import APIBasedModel
from med_red_team.models.interfaces import GenerationConfig, ModelGenerationError


class DeepSeekModel(APIBasedModel):
    """
    DeepSeek models (uses OpenAI-compatible API).
    
    Supports:
    - deepseek-chat (deepseek-v3)
    - deepseek-reasoner (deepseek-r1)
    """
    
    def __init__(self, model_id: str):
        """
        Initialize DeepSeek model.
        
        Args:
            model_id: Model identifier (e.g., "deepseek-v3")
        """
        super().__init__(model_id, api_key_env_var="DEEPSEEK_API_KEY")
    
    def _initialize_client(self) -> OpenAI:
        """
        Initialize DeepSeek client.
        
        DeepSeek uses OpenAI-compatible API with custom base URL.
        """
        import os
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        return OpenAI(
            api_key=deepseek_api_key,
            base_url="https://api.deepseek.com"
        )
    
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Call DeepSeek API (OpenAI-compatible).
        """
        # Format messages using OpenAI format
        messages = self._format_messages(user_prompt, system_prompt)
        
        request_kwargs = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "stop": config.stop_sequences or None,
        }
        if "reasoner" not in self.model_id:
            request_kwargs.update(
                temperature=config.temperature,
                top_p=config.top_p,
            )

        try:
            response = self.client.chat.completions.create(**request_kwargs)
            
            return {
                "text": response.choices[0].message.content,
                "metadata": {
                    "finish_reason": response.choices[0].finish_reason,
                    "token_usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            }
            
        except Exception as e:
            raise ModelGenerationError(
                f"DeepSeek API call failed for {self.model_id}: {str(e)}"
            ) from e