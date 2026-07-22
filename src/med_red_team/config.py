"""
Base Configuration Classes
===========================

Shared configuration base classes for red-team testing modules.

This module provides common configuration functionality used across
privacy, robustness, bias, and other testing domains.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any
from med_red_team.models import GenerationConfig


class BaseConfig:
    """
    Base class for all red-team testing configurations.

    Provides shared serialization/deserialization logic for:
    - GenerationConfig objects
    - Nested attacker_strategies dictionaries
    - JSON file I/O

    Subclasses should define their own fields as dataclass fields.
    """

    @staticmethod
    def serialize_attacker_strategies(
        attacker_strategies: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert attacker_strategies dict, serializing nested GenerationConfig objects.

        This handles the pattern where strategy kwargs may contain GenerationConfig
        objects that need to be converted to dicts for JSON serialization.

        Args:
            attacker_strategies: Dict mapping strategy names to their kwargs

        Returns:
            Dict with GenerationConfig objects converted to dicts

        Example:
            >>> strategies = {
            ...     "implicit_disguise": {
            ...         "model_id": "gpt-4o",
            ...         "config": GenerationConfig(temperature=0.2)
            ...     }
            ... }
            >>> serialized = BaseConfig.serialize_attacker_strategies(strategies)
            >>> serialized["implicit_disguise"]["config"]
            {'temperature': 0.2, 'max_tokens': 2048, ...}
        """
        result = {}
        for strategy_name, kwargs in attacker_strategies.items():
            strategy_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, GenerationConfig):
                    strategy_kwargs[key] = asdict(value)
                else:
                    strategy_kwargs[key] = value
            result[strategy_name] = strategy_kwargs
        return result

    @staticmethod
    def deserialize_attacker_strategies(
        attacker_strategies: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert attacker_strategies dict, deserializing GenerationConfig dicts.

        This handles the reverse of serialize_attacker_strategies, reconstructing
        GenerationConfig objects from dicts loaded from JSON.

        Args:
            attacker_strategies: Dict with GenerationConfig as dicts

        Returns:
            Dict with GenerationConfig dicts converted to objects

        Example:
            >>> strategies = {
            ...     "implicit_disguise": {
            ...         "model_id": "gpt-4o",
            ...         "config": {"temperature": 0.2, "max_tokens": 2048}
            ...     }
            ... }
            >>> deserialized = BaseConfig.deserialize_attacker_strategies(strategies)
            >>> isinstance(deserialized["implicit_disguise"]["config"], GenerationConfig)
            True
        """
        result = {}
        for strategy_name, kwargs in attacker_strategies.items():
            strategy_kwargs = {}
            for key, value in kwargs.items():
                # Detect GenerationConfig dicts by checking for characteristic fields
                if isinstance(value, dict) and ('temperature' in value or 'max_tokens' in value):
                    strategy_kwargs[key] = GenerationConfig(**value)
                else:
                    strategy_kwargs[key] = value
            result[strategy_name] = strategy_kwargs
        return result

    @staticmethod
    def deserialize_generation_configs(
        config_dict: Dict[str, Any],
        config_keys: list
    ) -> Dict[str, Any]:
        """
        Convert GenerationConfig dicts to objects in config dict.

        Args:
            config_dict: Configuration dictionary
            config_keys: List of keys that should be GenerationConfig objects

        Returns:
            Modified config dict with GenerationConfig objects

        Example:
            >>> config = {
            ...     "testee_config": {"temperature": 0.0, "max_tokens": 1024},
            ...     "grader_config": {"temperature": 0.0, "max_tokens": 512}
            ... }
            >>> result = BaseConfig.deserialize_generation_configs(
            ...     config,
            ...     ["testee_config", "grader_config"]
            ... )
            >>> isinstance(result["testee_config"], GenerationConfig)
            True
        """
        for key in config_keys:
            if key in config_dict and isinstance(config_dict[key], dict):
                config_dict[key] = GenerationConfig(**config_dict[key])
        return config_dict

    def to_json(self, filepath: str):
        """
        Save configuration to JSON file.

        Subclasses must implement to_dict() method.

        Args:
            filepath: Path to save JSON file

        Example:
            >>> config = SomeConfig()
            >>> config.to_json("configs/my_config.json")
        """
        config_dict = self.to_dict()
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath_obj, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"[Config] Saved to: {filepath}")

    @classmethod
    def from_json(cls, filepath: str):
        """
        Load configuration from JSON file.

        Subclasses must implement from_dict() class method.

        Args:
            filepath: Path to JSON config file

        Returns:
            Config instance (subclass type)

        Example:
            >>> config = SomeConfig.from_json("configs/my_config.json")
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        print(f"[Config] Loaded from: {filepath}")
        return cls.from_dict(config_dict)
