"""
Anthropic Claude model implementations.
"""

from typing import Dict, Any
import anthropic

from med_red_team.models.api_based.base import APIBasedModel
from med_red_team.models.interfaces import GenerationConfig, ModelGenerationError


class ClaudeModel(APIBasedModel):
    """
    Anthropic Claude models.
    
    Supports:
    - claude-sonnet-4
    - claude-sonnet-4.5
    """
    
    def __init__(self, model_id: str):
        """
        Initialize Claude model.
        
        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4")
        """
        super().__init__(model_id, api_key_env_var="ANTHROPIC_API_KEY")
    
    def _initialize_client(self) -> anthropic.Anthropic:
        """
        Initialize Anthropic client.
        
        Reads ANTHROPIC_API_KEY from environment.
        """
        return anthropic.Anthropic()
    
    def _call_api(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig
    ) -> Dict[str, Any]:
        """
        Call Anthropic API.
        https://docs.claude.com/en/api/messages
        docs on Thinking: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
        """
        try:
            response = self.client.messages.create(
                model=self.model_id,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ],
                **({"thinking": {"type": "enabled", "budget_tokens": config.max_thinking_tokens}} 
                if config.use_thinking else {})
            )
            
            # Extract content from response
            thinking_content = None
            text_content = None
            
            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                elif block.type == "text":
                    text_content = block.text
            
            return {
                "text": text_content,
                "thinking": thinking_content if config.use_thinking else None,
                "metadata": {
                    "finish_reason": response.stop_reason,
                    "token_usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            }
        # try:
        #     response = self.client.messages.create(
        #         model=self.model_id,
        #         max_tokens=config.max_tokens,
        #         temperature=config.temperature,
        #         system=system_prompt,
        #         messages=[
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {
        #                         "type": "text",
        #                         "text": user_prompt
        #                     }
        #                 ]
        #             }
        #         ]
        #     )
            
        #     return {
        #         "text": response.content[0].text,
        #         "metadata": {
        #             "finish_reason": response.stop_reason,
        #             "token_usage": {
        #                 "input_tokens": response.usage.input_tokens,
        #                 "output_tokens": response.usage.output_tokens
        #             }
        #         }
        #     }
            
        except Exception as e:
            raise ModelGenerationError(
                f"Anthropic API call failed for {self.model_id}: {str(e)}"
            ) from e