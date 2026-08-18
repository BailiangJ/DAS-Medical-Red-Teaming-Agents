"""Lazy, per-process model instance reuse."""

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Tuple

from med_red_team.models import LLMInterface, ModelFactory


logger = logging.getLogger(__name__)


_GENERATION_ONLY_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "repetition_penalty",
    "stop_sequences",
    "use_thinking",
    "max_thinking_tokens",
    "reasoning_effort",
}


def _freeze(value: Any) -> Hashable:
    """Convert construction configuration into a deterministic cache-key value."""
    if is_dataclass(value):
        return _freeze(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(item) for item in value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


class ModelPool:
    """Cache constructed models for one single-threaded process.

    Generation settings are supplied to each ``generate`` call and do not
    participate in model identity. Only constructor and infrastructure overrides
    may be supplied to :meth:`get_model`.
    """

    def __init__(self) -> None:
        self._loaded_models: Dict[Tuple[Hashable, ...], LLMInterface] = {}

    def get_model(self, model_id: str, **override_kwargs: Any) -> LLMInterface:
        invalid = sorted(_GENERATION_ONLY_KEYS.intersection(override_kwargs))
        if invalid:
            joined = ", ".join(invalid)
            raise ValueError(
                f"Generation settings ({joined}) cannot be passed to ModelPool.get_model(); "
                "pass them through GenerationConfig when calling generate()."
            )

        model_class, effective_kwargs = ModelFactory.resolve_constructor(
            model_id,
            **override_kwargs,
        )
        cache_key = self._make_cache_key(model_id, effective_kwargs)
        if cache_key in self._loaded_models:
            logger.debug("Reusing cached model: %s", model_id)
            return self._loaded_models[cache_key]

        logger.debug("Loading model: %s", model_id)
        model = model_class(model_id=model_id, **effective_kwargs)
        self._loaded_models[cache_key] = model
        return model

    def _make_cache_key(
        self,
        model_id: str,
        kwargs: Dict[str, Any],
    ) -> Tuple[Hashable, ...]:
        """Identify a model by ID and normalized construction-time overrides."""
        return (model_id, _freeze(kwargs))

    def is_loaded(self, model_id: str, **override_kwargs: Any) -> bool:
        """Check whether an exact configuration, or any instance by ID, is loaded."""
        if override_kwargs:
            _, effective_kwargs = ModelFactory.resolve_constructor(
                model_id,
                **override_kwargs,
            )
            return (
                self._make_cache_key(model_id, effective_kwargs)
                in self._loaded_models
            )
        return any(key[0] == model_id for key in self._loaded_models)

    def get_loaded_models(self) -> list[str]:
        """Return unique loaded model IDs in insertion order."""
        return list(dict.fromkeys(str(key[0]) for key in self._loaded_models))

    def clear(self) -> None:
        """Best-effort model cleanup followed by cache clearing."""
        logger.debug("Clearing %d loaded model(s)", len(self._loaded_models))
        for model in self._loaded_models.values():
            for method_name in ("close", "shutdown", "unload"):
                method = getattr(model, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        logger.warning(
                            "%s failed for %s",
                            method_name,
                            model.model_id,
                            exc_info=True,
                        )
                    break
        self._loaded_models.clear()

    def __len__(self) -> int:
        return len(self._loaded_models)

    def __repr__(self) -> str:
        return f"ModelPool(loaded={len(self)} models: {self.get_loaded_models()})"
