"""
Factory for creating LLM instances from model_id strings.

This module routes model identifiers to their concrete implementations
and handles configuration merging.
"""

from copy import deepcopy
from importlib import import_module
from typing import Any, Dict

from med_red_team.models.interfaces import LLMInterface, InfrastructureConfig, ModelNotFoundError


class ModelFactory:
    """
    Creates LLM instances based on model_id.
    
    Design Pattern: Factory Pattern
    - Hides instantiation logic from clients
    - Centralizes model registry lookup
    - Handles configuration merging (defaults + overrides)
    
    Usage:
        # API model (minimal kwargs)
        model = ModelFactory.create("gpt-4o")
        
        # Open-source model with custom infra
        model = ModelFactory.create(
            "llama-3-70b", 
            gpu_memory_utilization=0.95,
            tensor_parallel_size=4
        )
    """
    
    # This will be populated by importing from registry.py
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_models(cls, registry: Dict[str, Dict[str, Any]]) -> None:
        """
        Register the model configuration dictionary.
        
        This is called once at module initialization.
        
        Args:
            registry: MODEL_CONFIGS from registry.py
        """
        cls._registry = registry

    @staticmethod
    def _resolve_model_class(model_class: Any) -> type[LLMInterface]:
        """Resolve a lazy ``module:Class`` reference when a model is created."""
        if not isinstance(model_class, str):
            return model_class
        module_name, separator, class_name = model_class.partition(":")
        if not separator:
            raise ValueError(f"Invalid model class reference: {model_class}")
        module = import_module(module_name)
        return getattr(module, class_name)

    @classmethod
    def create(cls, model_id: str, **override_kwargs: Any) -> LLMInterface:
        """
        Create an LLM instance.

        Args:
            model_id: Identifier for the model (e.g., "gpt-4o", "llama-3-70b")
            **override_kwargs: Override default config values from registry

        Returns:
            An instance implementing LLMInterface

        Raises:
            ModelNotFoundError: If model_id is not in registry

        Examples:
            # Use defaults from registry
            model = ModelFactory.create("llama-3-70b")

            # Override infrastructure config
            model = ModelFactory.create(
                "llama-3-70b",
                gpu_memory_utilization=0.95,
                tensor_parallel_size=4
            )

            # API model (overrides typically not needed)
            model = ModelFactory.create("gpt-4o")
        """
        if not cls.is_registered(model_id):
            raise ModelNotFoundError(
                f"Model '{model_id}' not found in registry. "
                f"Available models: {list(cls._registry.keys())}"
            )

        config = cls._registry[model_id]
        model_class = cls._resolve_model_class(config["class"])
        # Registry entries are process-global. Deep-copy nested dataclasses so
        # per-run infrastructure overrides cannot mutate later creations.
        default_kwargs = deepcopy(config["kwargs"])

        # Merge: defaults + overrides
        final_kwargs = cls._merge_kwargs(default_kwargs, override_kwargs)

        return model_class(model_id=model_id, **final_kwargs)
    
    @classmethod
    def _merge_kwargs(cls, defaults: dict, overrides: dict) -> dict:
        """
        Smart merge of kwargs, handling nested InfrastructureConfig.
        
        Logic:
        - If overrides contain infra params, update the infra_config object
        - Otherwise, directly merge kwargs
        
        Args:
            defaults: Default kwargs from registry
            overrides: User-provided overrides
            
        Returns:
            Merged kwargs dictionary
            
        Example:
            defaults = {"infra_config": InfrastructureConfig(gpu_memory_utilization=0.9)}
            overrides = {"gpu_memory_utilization": 0.95}
            
            Result: InfrastructureConfig.gpu_memory_utilization = 0.95
        """
        merged = defaults.copy()
        
        # Infrastructure config attribute names
        infra_attrs = {
            "gpu_memory_utilization",
            "tensor_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "dtype",
            "enable_tf32",
            "quantization",
            "seed"
        }
        
        # Check if we have an infra_config object and infra overrides
        if "infra_config" in merged:
            infra_overrides = {
                k: v for k, v in overrides.items() if k in infra_attrs
            }
            
            if infra_overrides:
                # Update the InfrastructureConfig instance
                infra_config = merged["infra_config"]
                for key, value in infra_overrides.items():
                    setattr(infra_config, key, value)
                
                # Remove infra overrides from remaining overrides
                remaining_overrides = {
                    k: v for k, v in overrides.items() if k not in infra_attrs
                }
                merged.update(remaining_overrides)
            else:
                # No infra overrides, just merge directly
                merged.update(overrides)
        else:
            # No infra_config in defaults, direct merge
            merged.update(overrides)
        
        return merged
    
    @classmethod
    def list_available_models(cls) -> list:
        """
        Get list of all registered model IDs.
        
        Returns:
            List of model_id strings
            
        Example:
            >>> ModelFactory.list_available_models()
            ['gpt-4o', 'claude-sonnet-4', 'llama-3-70b', ...]
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_model_info(cls, model_id: str) -> Dict[str, Any]:
        """
        Get configuration info for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with class name and default kwargs
            
        Raises:
            ModelNotFoundError: If model_id not in registry
            
        Example:
            >>> info = ModelFactory.get_model_info("llama-3-70b")
            >>> print(info["class"])
            'LlamaModel'
            >>> print(info["default_kwargs"])
            {'infra_config': InfrastructureConfig(...)}
        """
        if model_id not in cls._registry:
            raise ModelNotFoundError(f"Model '{model_id}' not found in registry.")
        
        config = cls._registry[model_id]
        class_ref = config["class"]
        class_name = class_ref.rsplit(":", 1)[-1] if isinstance(class_ref, str) else class_ref.__name__
        return {
            "model_id": model_id,
            "class": class_name,
            "default_kwargs": deepcopy(config["kwargs"])
        }
    
    @classmethod
    def is_registered(cls, model_id: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model is in registry, False otherwise
            
        Example:
            >>> ModelFactory.is_registered("gpt-4o")
            True
            >>> ModelFactory.is_registered("unknown-model")
            False
        """
        return model_id in cls._registry