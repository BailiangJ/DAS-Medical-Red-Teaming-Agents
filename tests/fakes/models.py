"""Reusable provider-free model test doubles."""

from collections import deque
from typing import Any

from med_red_team.models import GenerationConfig, ModelResponse


class ScriptedModel:
    def __init__(self, model_id: str, responses: list[str | ModelResponse]):
        self.model_id = model_id
        self.responses = deque(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        config: GenerationConfig,
    ) -> ModelResponse:
        self.calls.append({
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "config": config,
        })
        if not self.responses:
            raise AssertionError(f"No scripted response remains for {self.model_id}")
        response = self.responses.popleft()
        if isinstance(response, ModelResponse):
            return response
        return ModelResponse(
            raw_text=response,
            final_answer=response,
            model_id=self.model_id,
            generation_config=config,
        )


class StaticModelPool:
    def __init__(self, models: dict[str, Any]):
        self.models = dict(models)
        self.requests = []

    def get_model(self, model_id: str, **kwargs):
        self.requests.append((model_id, kwargs))
        if model_id not in self.models:
            raise KeyError(f"No fake model registered for {model_id}")
        return self.models[model_id]

    def is_loaded(self, model_id: str, **kwargs) -> bool:
        return model_id in self.models

    def get_loaded_models(self) -> list[str]:
        return list(self.models)

    def clear(self) -> None:
        self.models.clear()
