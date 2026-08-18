"""
Robustness Evaluation Configuration
====================================

Configuration management for robustness red-teaming evaluation.

This module provides:
- RobustnessConfig: Centralized configuration dataclass
- Preset configurations (DEFAULT, FAST, DEV)
- JSON serialization/deserialization

Example:
    >>> from med_red_team.robustness import RobustnessConfig, FAST_CONFIG
    >>>
    >>> # Use preset
    >>> config = FAST_CONFIG
    >>>
    >>> # Custom config
    >>> config = RobustnessConfig(
    ...     testee_model="gpt-4o",
    ...     attack_strategies=["bias_manipulation", "add_distraction_sentence"],
    ...     max_samples=100
    ... )
    >>>
    >>> # Save/load
    >>> config.to_json("configs/my_config.json")
    >>> config = RobustnessConfig.from_json("configs/my_config.json")
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from med_red_team.models import GenerationConfig
from med_red_team.config import BaseConfig


# ==============================================================================
# Robustness Configuration
# ==============================================================================

@dataclass
class RobustnessConfig(BaseConfig):
    """
    Configuration for robustness evaluation.

    This dataclass centralizes all settings for robustness testing:
    - Model configurations (testee, attacker)
    - Generation parameters
    - Evaluation settings
    - Attack strategies
    - Data and output paths

    Attributes:
        testee_model: Model ID for the model being tested
        grader_type: Grader type ("simple" for rule-based, or LLM model ID)
        testee_config: Generation config for testee
        testee_system_prompt: System prompt for testee
        data_dir: Directory containing test data
        data_split: Data split to use ("test", "train", "dev")
        max_samples: Maximum number of samples to evaluate (None = all)
        skip_incorrect_first_round: Skip samples incorrect in baseline
        attacker_strategies: Dict mapping strategy names to their __init__ kwargs
        log_dir: Directory for saving results

    Example:
        >>> # Strategies with custom kwargs
        >>> config = RobustnessConfig(
        ...     testee_model="gpt-4o",
        ...     attacker_strategies={
        ...         "add_distraction_sentence": {
        ...             "model_id": "gpt-4o",
        ...             "config": GenerationConfig(temperature=0.2)
        ...         },
        ...         "bias_manipulation": {
        ...             "model_id": "gpt-4o",
        ...             "n_bias_styles": 3
        ...         }
        ...     },
        ...     max_samples=100
        ... )
        >>>
        >>> # Let strategies use __init__ defaults (empty dicts)
        >>> config = RobustnessConfig(
        ...     attacker_strategies={
        ...         "add_none_of_the_above": {}  # Rule-based, no LLM
        ...     }
        ... )
    """

    # Model IDs
    testee_model: str = "gpt-4o"
    grader_type: str = "simple"  # "simple" for SimpleGrader, or LLM model ID

    # Generation configs
    testee_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(
            temperature=0.0,  # Deterministic for consistency
            max_tokens=512
        )
    )

    # System prompt for testee
    testee_system_prompt: str = (
        "Answer the question with only the final letter choice (A, B, C, D, E, or F) "
        "with no explanation. More than one choice can be correct, in which case "
        "return all of them separated by commas."
    )

    # Data settings
    dataset_path: str = "data/medqa_test.jsonl"  # Path to JSONL file

    # Evaluation settings
    max_samples: Optional[int] = None  # None = evaluate all
    skip_incorrect_first_round: bool = True  # Skip samples incorrect in baseline

    # Attack strategies configuration
    # Maps strategy names to their __init__ kwargs (model_id, config, strategy-specific params)
    # Example: {
    #     "add_distraction_sentence": {"model_id": "gpt-4o", "config": GenerationConfig(...)},
    #     "bias_manipulation": {"model_id": "gpt-4o", "n_bias_styles": 3}
    # }
    # If kwargs not specified, strategy uses defaults from its __init__ method
    attacker_strategies: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Output settings
    log_dir: str = "logs"

    def __post_init__(self):
        """
        Validate configuration after initialization.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate max_samples using the shared None/zero/positive contract.
        from med_red_team.shared.config_loading import validate_sample_limit
        validate_sample_limit(self.max_samples)

        # Validate strategy names against registry (lazy import to avoid circular dependency)
        if self.attacker_strategies:
            self._validate_strategies()

    def _validate_strategies(self):
        """
        Validate strategy names against registry.

        This is separated to allow lazy import of STRATEGY_REGISTRY.
        """
        try:
            from med_red_team.attacker_registry import STRATEGY_REGISTRY

            for strategy_name in self.attacker_strategies.keys():
                if strategy_name not in STRATEGY_REGISTRY:
                    available = ", ".join(STRATEGY_REGISTRY.keys())
                    raise ValueError(
                        f"Unknown strategy: '{strategy_name}'. "
                        f"Available strategies: {available}\n"
                        f"Use list_available_strategies() to see all options."
                    )
        except ImportError:
            # Registry not yet loaded, skip validation
            pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of config

        Example:
            >>> config = RobustnessConfig()
            >>> config_dict = config.to_dict()
        """
        # Convert attacker_strategies using base class method
        attacker_strategies_dict = self.serialize_attacker_strategies(self.attacker_strategies)

        config_dict = {
            "testee_model": self.testee_model,
            "grader_type": self.grader_type,
            "testee_config": asdict(self.testee_config),
            "testee_system_prompt": self.testee_system_prompt,
            "dataset_path": self.dataset_path,
            "max_samples": self.max_samples,
            "skip_incorrect_first_round": self.skip_incorrect_first_round,
            "attacker_strategies": attacker_strategies_dict,
            "log_dir": self.log_dir
        }
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RobustnessConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with config fields

        Returns:
            RobustnessConfig instance

        Example:
            >>> config_dict = {...}
            >>> config = RobustnessConfig.from_dict(config_dict)
        """
        # Create a copy to avoid modifying original
        config_dict = config_dict.copy()

        # Convert testee_config dict back to GenerationConfig
        if 'testee_config' in config_dict and isinstance(config_dict['testee_config'], dict):
            config_dict['testee_config'] = GenerationConfig(**config_dict['testee_config'])

        # Convert attacker_strategies using base class method
        if 'attacker_strategies' in config_dict and isinstance(config_dict['attacker_strategies'], dict):
            config_dict['attacker_strategies'] = cls.deserialize_attacker_strategies(
                config_dict['attacker_strategies']
            )

        return cls(**config_dict)


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_config_from_dict(config_dict: Dict[str, Any]) -> RobustnessConfig:
    """
    Create configuration from dictionary with validation.

    This is a convenience wrapper around RobustnessConfig.from_dict().

    Args:
        config_dict: Dictionary with config fields

    Returns:
        RobustnessConfig instance

    Example:
        >>> config_dict = {"testee_model": "gpt-4o", "max_samples": 50}
        >>> config = create_config_from_dict(config_dict)
    """
    return RobustnessConfig.from_dict(config_dict)


def load_config(filepath: str) -> RobustnessConfig:
    """
    Load configuration from JSON file.

    This is a convenience wrapper around RobustnessConfig.from_json().

    Args:
        filepath: Path to JSON config file

    Returns:
        RobustnessConfig instance

    Example:
        >>> config = load_config("configs/my_config.json")
    """
    return RobustnessConfig.from_json(filepath)
